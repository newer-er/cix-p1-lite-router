"""Start both classifier profiles, their Routers and the frontend on loopback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener
import venv


BOARD = Path(__file__).resolve().parent
ROOT = BOARD.parent
TORCH = "torch==2.8.0+cpu"
HTTP = build_opener(ProxyHandler({}))


def ensure_python(prepared: Path | None) -> str:
    if prepared is not None:
        executable = prepared.expanduser().absolute()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError("--python 需要指向可执行的板端虚拟环境 Python")
        return str(executable)
    if sys.version_info[:2] != (3, 12):
        raise ValueError("板端首次安装使用 Python 3.12")
    environment = ROOT / ".venv-board"
    executable = environment / "bin" / "python"
    requirements = BOARD / "requirements.txt"
    fingerprint = hashlib.sha256(requirements.read_bytes() + TORCH.encode()).hexdigest()
    stamp = environment / ".dependencies"
    if not executable.is_file():
        venv.create(environment, with_pip=True)
    if not stamp.is_file() or stamp.read_text() != fingerprint:
        pip = [str(executable), "-m", "pip", "install", "--only-binary=:all:"]
        print("安装板端 CPU 推理依赖", flush=True)
        subprocess.run(pip + ["--index-url", "https://download.pytorch.org/whl/cpu", TORCH], check=True)
        subprocess.run(pip + ["-r", str(requirements)], check=True)
        subprocess.run([str(executable), "-m", "pip", "check"], check=True)
        stamp.write_text(fingerprint)
    return str(executable)


def require_assets(m47b: Path, intent: Path) -> None:
    required = [
        m47b / name for name in (
            "manifest.json", "config.json", "model.safetensors",
            "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
        )
    ]
    required += [
        intent / "models/embedding_intent_06b_final_20261116" / name
        for name in ("model_spec.json", "training_policy.json", "best_trainable.safetensors")
    ]
    required += [
        intent / "backbone/Qwen3-Embedding-0.6B" / name
        for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json")
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("缺少模型文件\n" + "\n".join(missing))


def wait_health(port: int, key: str, children: list, timeout: int) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if any(process.poll() is not None for process in children):
            raise RuntimeError("板端进程提前退出，请查看前面的进程输出")
        try:
            with HTTP.open(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                if json.load(response).get(key) is True:
                    return
        except (OSError, ValueError):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"回环端口 {port} 启动超时")


def run(args) -> None:
    require_assets(args.m47b_bundle, args.intent_assets)
    api_key = os.environ.get("ANYINT_API_KEY", "").strip()
    if not api_key:
        raise ValueError("启动实机模式前需要设置 ANYINT_API_KEY")

    primary, shared, router0, router1 = range(args.first_service_port, args.first_service_port + 4)
    ports = (args.port, primary, shared, router0, router1)
    if len(set(ports)) != len(ports) or any(not 1 <= port <= 65535 for port in ports):
        raise ValueError("网页端口和四个连续的板端端口需要互异且位于 1 至 65535")
    python = ensure_python(args.python)
    for port in ports:
        with socket.socket() as listener:
            try:
                listener.bind(("127.0.0.1", port))
            except OSError as error:
                raise ValueError(f"回环端口 {port} 无法使用") from error

    env = {**os.environ, "PYTHONPATH": str(BOARD), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    env.pop("ANYINT_API_KEY", None)
    gateway_env = {**env, "ANYINT_API_KEY": api_key}
    processes = []

    def start(arguments: list[str], *, gateway: bool = False):
        child = subprocess.Popen([python, "-B", *arguments], cwd=ROOT, env=gateway_env if gateway else env)
        processes.append(child)
        return child

    try:
        print("加载 0.8B 分类器", flush=True)
        start(["-m", "fusion.classifier_serve", "--bundle", str(args.m47b_bundle),
               "--port", str(primary), "--threads", str(args.threads)])
        wait_health(primary, "ready", processes, args.startup_timeout)
        print("加载 0.6B 意图模型", flush=True)
        start([str(BOARD / "intent_classifier.py"), "--assets", str(args.intent_assets),
               "--base-classifier-url", f"http://127.0.0.1:{primary}", "--port", str(shared), "--threads", str(args.threads)])
        wait_health(shared, "ready", processes, args.startup_timeout)
        for backend_port, router_port in ((primary, router0), (shared, router1)):
            start([str(BOARD / "edge_router.py"), "--classifier-url", f"http://127.0.0.1:{backend_port}",
                   "--port", str(router_port)], gateway=True)
            wait_health(router_port, "classification_ready", processes, args.startup_timeout)
        start([str(ROOT / "server.py"), "--port", str(args.port),
               "--router-url", f"http://127.0.0.1:{router0}", "--intent-router-url", f"http://127.0.0.1:{router1}"])
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        raise RuntimeError("演示进程退出")
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for process in reversed(processes):
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def stop(_signal, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m47b-bundle", type=lambda s: Path(s).expanduser().resolve(), required=True)
    parser.add_argument("--intent-assets", type=lambda s: Path(s).expanduser().resolve(), required=True)
    parser.add_argument("--port", type=int, default=18812)
    parser.add_argument("--first-service-port", type=int, default=18910)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--python", type=Path, help="板上已准备好的虚拟环境 Python")
    args = parser.parse_args()
    if args.threads < 1 or args.startup_timeout < 1:
        parser.error("--threads 和 --startup-timeout 需要为正数")
    for name in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(name, stop)
    try:
        run(args)
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"依赖安装失败，进程退出码 {error.returncode}") from error
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"部署失败。{error}") from error
