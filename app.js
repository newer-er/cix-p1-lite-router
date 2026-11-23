const examples = {
  inspection: "把以下巡检交接记录整理成两条待办：A区运行正常，B区指示灯偶尔闪烁。明早复查B区电源接头，下午把复查结果补进登记表。",
  debugging: "这段采样脚本遇到空列表会报 IndexError：def latest(samples): return samples[-1]。请给出能处理空列表的修复和两条最小测试。",
};

const taskNames = new Map(Object.entries({
  general: "通用",
  coding: "编程",
  debugging: "故障调试",
  math: "数学",
  science: "科学",
  research: "研究",
  multimodal: "多模态",
}));
const textTasks = ["general", "coding", "debugging", "math", "science", "research"];

const intentNames = new Map(Object.entries({
  compare: "比较",
  debug: "调试",
  explain: "解释",
  fix: "修复",
  generate: "生成",
  other: "其他",
  plan: "规划",
  review: "审查",
  search: "搜索",
  summarize: "总结",
  transform: "转换",
}));
const toolNames = new Map([["yes", "需要"], ["no", "不需要"], ["uncertain", "不确定"]]);
const signalDefinitions = [
  ["domain", "领域", taskNames],
  ["intent_primary", "意图", intentNames],
  ["requires_tools", "工具需求", toolNames],
];

const fallbackNames = new Map(Object.entries({
  domain_low_confidence: "领域置信度偏低",
  domain_unavailable: "领域预测不可用",
  backend_unavailable: "分类器暂不可用",
  classifier_response_invalid: "分类结果无效",
  no_prediction: "没有可用的分类结果",
  image_constraint: "图像输入约束",
  shadow_only: "影子模式",
  decision_mode_disabled: "分类采纳已关闭",
}));

const byId = (id) => document.getElementById(id);
const form = byId("request-form");
const promptInput = byId("prompt");
const sendButton = byId("send");
const exampleButtons = document.querySelectorAll("[data-example]");
const boardStatus = byId("board-status");
const boardState = byId("board-state");
const classifierSelect = byId("classifier-model");
const pendingClassifiers = byId("pending-classifiers");
const currentClassifier = byId("current-classifier");
const requestState = byId("request-state");
const features = byId("features");
const signals = byId("signals");
const signalEmpty = byId("signal-empty");
const signalDetail = byId("signal-detail");
const probabilities = byId("probabilities");
const task = byId("task");
const policy = byId("policy");
const worker = byId("worker");
const workerCalls = byId("worker-calls");
const details = byId("route-details");
const detailsList = byId("details-list");
const comparison = byId("route-comparison");
const comparisonList = byId("comparison-list");
const answer = byId("answer");
const responsePanel = byId("response-panel");
const exampleNames = new Map(Array.from(exampleButtons, (button) => [
  examples[button.dataset.example], button.querySelector(".example-title").textContent,
]));
const recentRoutes = [];
const classifiers = new Map();
let completedRequest = null;
let healthCheckId = 0;

const milliseconds = (value) => Number.isFinite(value) && value >= 0 ? `${(value / 1000).toFixed(2)} 秒` : "";
const taskName = (value) => taskNames.get(value) ?? value;
const probabilityValue = (value) => Number.isFinite(value) && value >= 0 && value <= 1;
const percent = (value) => probabilityValue(value) ? `${(value * 100).toFixed(1)}%` : "";
const classificationFor = (route) => route?.workflow?.classification ?? route?.classification;
const classifierLabel = (model) => `${model.name} ${model.parameters}`;

function element(tag, text, className = "") {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}

function setLabel(node, text = "") {
  node.textContent = text;
  node.hidden = !text;
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => null);
  if (response.ok && data) return data;

  const messages = {
    router_unreachable: "无法连接香橙派。请检查 SSH 隧道后重试。",
    router_timeout: "板端请求超时，请稍后重试。",
    router_invalid_response: "板端返回的响应无法读取。",
    classifier_unknown: "所选板端模型未配置，请刷新页面。",
    replay_example_required: "离线回放支持左侧的两条示例。载入一条示例后再发送。",
    prompt_required: "请输入一条请求。",
    prompt_too_long: "请求超过 12,000 字符。",
    request_too_large: "请求内容太大。",
  };
  if (data?.error === "router_rejected") {
    throw new Error(`板端返回错误（HTTP ${data.router_status}），请稍后重试。`);
  }
  throw new Error(messages[data?.error] ?? "请求失败，请重试。");
}

function setBoardState(state, text) {
  boardStatus.dataset.state = state;
  boardStatus.title = text;
  boardState.textContent = text;
}

async function checkBoard() {
  const checkId = ++healthCheckId;
  const query = new URLSearchParams({classifier: classifierSelect.value});
  try {
    const health = await requestJson(`/api/health?${query}`);
    if (checkId !== healthCheckId) return;
    const replay = health.mode === "replay";
    byId("live-intro").hidden = replay;
    byId("replay-intro").hidden = !replay;
    if (replay) {
      setBoardState("replay", "离线回放");
    } else if (health.classification_ready !== true) {
      setBoardState("warning", "分类未就绪");
    } else if (health.upstream_configured === false) {
      setBoardState("warning", "上游未配置");
    } else {
      setBoardState("online", "在线");
    }
  } catch {
    if (checkId === healthCheckId) setBoardState("offline", "连接中断");
  }
}

function showSelectedClassifier() {
  const model = classifiers.get(classifierSelect.value);
  if (!model) return;
  const sources = model.head_sources;
  const label = sources
    ? `${sources.intent_primary} 意图模型与 ${sources.domain} 领域和工具模型`
    : classifierLabel(model);
  currentClassifier.textContent = label;
  classifierSelect.title = label;
}

async function loadClassifiers() {
  const catalog = await requestJson("/api/classifiers");
  for (const model of catalog.models) classifiers.set(model.id, model);
  const previous = classifierSelect.value;
  const options = Array.from(classifiers.values(), (model) => {
    const option = new Option(model.parameters, model.id);
    option.title = classifierLabel(model);
    return option;
  });
  if (!classifiers.has(pendingClassifiers.firstElementChild.value)) options.push(pendingClassifiers);
  classifierSelect.replaceChildren(...options);
  classifierSelect.value = classifiers.has(previous) ? previous : catalog.default;
  classifierSelect.disabled = sendButton.disabled;
  showSelectedClassifier();
}

function classifierForVersion(version) {
  if (typeof version !== "string") return;
  for (const model of classifiers.values()) {
    if (version.startsWith(model.version_prefix)) return model;
  }
}

function setRequestState(text, tone = "") {
  setLabel(requestState, text);
  requestState.dataset.state = tone;
}

function setAnswerMessage(text, isError = false) {
  answer.replaceChildren(element("p", text, isError ? "answer-error" : "answer-placeholder"));
  answer.scrollTop = 0;
}

function renderAnswer(text) {
  answer.replaceChildren();
  const blocks = text.split(/^```[^\r\n]*(?:\r?\n|$)/mu);
  for (const [index, block] of blocks.entries()) {
    if (!block.trim()) continue;
    if (index % 2 === 1) {
      const pre = document.createElement("pre");
      pre.append(element("code", block.trimEnd()));
      answer.append(pre);
      continue;
    }

    const prose = block.trim();
    const paragraph = element("div", "", "answer-text");
    let start = 0;
    for (const match of prose.matchAll(/\*\*([^\n*]+)\*\*|`([^`\n]+)`/gu)) {
      paragraph.append(prose.slice(start, match.index));
      paragraph.append(match[1] ? element("strong", match[1]) : element("code", match[2]));
      start = match.index + match[0].length;
    }
    paragraph.append(prose.slice(start));
    answer.append(paragraph);
  }
  answer.scrollTop = 0;
}

function showRequestFeatures(prompt) {
  const tags = ["单轮", `${Array.from(prompt).length} 字符`];
  if (/\p{Script=Han}/u.test(prompt)) tags.unshift("含中文");
  if (/```|\b(?:def|function)\s+[A-Za-z_]\w*\s*\(/u.test(prompt)) tags.push("含代码");
  features.replaceChildren(...tags.map((tag) => element("li", tag)));
  features.hidden = false;
  byId("request-step").dataset.state = "complete";
}

function toggleSignal(button, prediction, label, names, used, source) {
  const open = button.getAttribute("aria-expanded") !== "true";
  for (const candidate of signals.querySelectorAll("button")) {
    candidate.setAttribute("aria-expanded", String(candidate === button && open));
  }
  signalDetail.hidden = !open;
  if (!open) return;

  byId("signal-note").textContent = [`${label}预测`, used ? "参与路由判定" : "仅供观察", source]
    .filter(Boolean).join(" · ");
  const scores = Object.entries(prediction.probabilities ?? {})
    .filter(([, score]) => probabilityValue(score))
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4);
  if (!scores.length && probabilityValue(prediction.confidence)) {
    scores.push([prediction.value, prediction.confidence]);
  }
  probabilities.replaceChildren();
  for (const [key, value] of scores) {
    const name = names.get(key) ?? key;
    const row = element("li", "", "probability");
    const bar = document.createElement("progress");
    bar.max = 1;
    bar.value = value;
    bar.setAttribute("aria-label", `${name}的预测概率`);
    row.append(element("span", name), bar, element("span", percent(value), "probability-score"));
    probabilities.append(row);
  }
  probabilities.hidden = !scores.length;
}

function showSignals(classification) {
  signals.replaceChildren();
  probabilities.replaceChildren();
  signalDetail.hidden = true;
  const predictions = Array.isArray(classification?.answers) ? classification.answers : [];
  const usedDimensions = classification?.applied && Array.isArray(classification.used_dimensions)
    ? classification.used_dimensions : [];
  const sources = classifierForVersion(classification?.model_version)?.head_sources;
  for (const [id, label, names] of signalDefinitions) {
    const prediction = predictions.find((item) => item?.question_id === id);
    if (typeof prediction?.value !== "string" || !prediction.value || prediction.abstained) continue;

    const used = usedDimensions.includes(id);
    const button = element("button", "", "signal");
    button.type = "button";
    button.dataset.used = String(used);
    button.setAttribute("aria-controls", "signal-detail");
    button.setAttribute("aria-expanded", "false");
    const title = element("span", "", "signal-top");
    title.append(element("span", label), element("span", percent(prediction.confidence)));
    const value = element("span", "", "signal-main");
    value.append(element("strong", names.get(prediction.value) ?? prediction.value, "signal-value"));
    const source = sources?.[id];
    if (source) value.append(element("span", source, "signal-source"));
    button.append(title, value);
    button.addEventListener("click", () => toggleSignal(button, prediction, label, names, used, source));
    signals.append(button);
  }
  signals.hidden = !signals.childElementCount;
  setLabel(signalEmpty, signals.hidden ? (classification ? "分类信号不可用" : "等待板端分类") : "");
  const time = milliseconds(classification?.latency_ms);
  setLabel(byId("signal-time"), time ? `板内 ${time}` : "");
  byId("signals-step").dataset.state = signals.hidden ? "idle" : "complete";
}

function showLanes(selected) {
  const lanes = textTasks.includes(selected) || !selected ? textTasks : [...textTasks, selected];
  byId("lanes").replaceChildren(...lanes.map((name) => {
    const lane = element("li", taskName(name));
    if (name === selected) {
      lane.dataset.selected = "true";
      lane.setAttribute("aria-current", "true");
    }
    return lane;
  }));
}

function showDecision(route, classification) {
  const selected = route?.task_type ?? classification?.task_type;
  const baseline = classification?.baseline_task_type;
  const version = classification?.model_version;
  const model = classifierForVersion(version);
  const dimensions = Array.isArray(classification?.used_dimensions) ? classification.used_dimensions : [];
  const adopted = [...new Set(dimensions.map((dimension) => model?.head_sources?.[dimension]).filter(Boolean))];
  const classifier = adopted.join(" + ") || (model ? classifierLabel(model) : version);
  const fallback = classification?.fallback_reason;

  showLanes(selected);
  task.replaceChildren();
  const changed = selected && baseline && selected !== baseline;
  task.hidden = !changed;
  if (changed) {
    task.append(element("span", taskName(baseline), "previous-task"));
    task.append(element("span", "→", "task-arrow"), taskName(selected));
  }
  const note = fallback
    ? `规则回退 · ${fallbackNames.get(fallback) ?? fallback}`
    : classification?.applied && classifier ? `已采纳 ${classifier} 分类` : "";
  setLabel(policy, note);
  policy.dataset.state = fallback ? "warning" : "";
  byId("decision-step").dataset.state = selected ? "complete" : "idle";
}

function showWorker(route) {
  const selected = route?.worker_model;
  const calls = Array.isArray(route?.calls) ? route.calls.filter((call) => typeof call?.model === "string" && call.model) : [];
  const count = route?.budget?.worker_calls ?? (Array.isArray(route?.calls) ? calls.length : null);
  const time = calls.length === 1 && calls[0].latency_ms > 0 ? milliseconds(calls[0].latency_ms) : "";
  const metadata = [Number.isInteger(count) ? `${count} 次调用` : "", time].filter(Boolean).join(" · ");
  setLabel(byId("call-count"), metadata);

  worker.textContent = selected || (route ? "未返回模型名称" : "等待选模");
  worker.dataset.selected = String(Boolean(selected));
  worker.classList.toggle("placeholder", !selected);
  workerCalls.replaceChildren();
  if (calls.length > 1) {
    for (const call of calls) {
      const time = call.latency_ms > 0 ? milliseconds(call.latency_ms) : "";
      workerCalls.append(element("li", [call.model, time].filter(Boolean).join(" · ")));
    }
  }
  workerCalls.hidden = !workerCalls.childElementCount;
  byId("worker-step").dataset.state = selected ? "complete" : "idle";
}

function showRoute(route) {
  const classification = classificationFor(route);
  showSignals(classification);
  if (route && !classification) setLabel(signalEmpty, "未返回分类记录");
  showDecision(route, classification);
  showWorker(route);

  detailsList.replaceChildren();
  const fields = [
    ["分类模型", classification?.model_version],
    ["策略", classification?.policy_version],
    ["采纳阈值", percent(classification?.minimum_confidence)],
  ];
  for (const [label, value] of fields) {
    if (!value) continue;
    detailsList.append(element("dt", label), element("dd", value));
  }
  details.hidden = !detailsList.childElementCount;
  if (details.hidden) details.open = false;
}

function rememberRoute(prompt, route) {
  if (!route?.worker_model || !route?.task_type) return;

  const chars = Array.from(prompt.replace(/\s+/gu, " "));
  const name = exampleNames.get(prompt) ?? (chars.length > 12 ? `${chars.slice(0, 12).join("")}…` : chars.join(""));
  const model = classifierForVersion(classificationFor(route)?.model_version);
  const description = [taskName(route.task_type), model?.parameters].filter(Boolean).join(" · ");
  recentRoutes.push({name, description, worker: route.worker_model});
  if (recentRoutes.length > 2) recentRoutes.shift();
  if (recentRoutes.length < 2) return;

  comparisonList.replaceChildren();
  for (const entry of recentRoutes) {
    const row = document.createElement("li");
    row.append(element("span", entry.name, "comparison-name"), element("span", entry.description, "comparison-task"));
    row.append(element("code", entry.worker, "comparison-worker"));
    comparisonList.append(row);
  }
  comparison.hidden = false;
}

function syncPrompt() {
  const value = promptInput.value.trim();
  for (const button of exampleButtons) {
    button.setAttribute("aria-pressed", String(value === examples[button.dataset.example]));
  }
  if (completedRequest) {
    const sameRequest = value === completedRequest.prompt && classifierSelect.value === completedRequest.classifier;
    setRequestState(sameRequest ? "" : "上次请求");
  }
}

function setBusy(busy) {
  sendButton.disabled = busy;
  classifierSelect.disabled = busy || !classifiers.size;
  promptInput.readOnly = busy;
  answer.setAttribute("aria-busy", String(busy));
  for (const button of exampleButtons) button.disabled = busy;
}

function loadExample(name, preventScroll = false) {
  promptInput.value = examples[name];
  syncPrompt();
  promptInput.focus({preventScroll});
}

for (const button of exampleButtons) {
  button.addEventListener("click", () => loadExample(button.dataset.example));
}

const heroExample = document.querySelector("[data-hero-example]");
heroExample.addEventListener("click", (event) => {
  if (!sendButton.disabled && !event.ctrlKey && !event.metaKey && !event.shiftKey && !event.altKey) {
    loadExample(heroExample.dataset.heroExample, true);
  }
});

classifierSelect.addEventListener("change", () => {
  showSelectedClassifier();
  syncPrompt();
  setBoardState("checking", "连接中");
  checkBoard();
});

promptInput.addEventListener("input", syncPrompt);
promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.isComposing && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (sendButton.disabled) return;

  const prompt = promptInput.value.trim();
  if (!prompt) {
    promptInput.value = "";
    promptInput.reportValidity();
    return;
  }

  const classifier = classifierSelect.value;
  promptInput.value = prompt;
  completedRequest = null;
  setBusy(true);
  showRequestFeatures(prompt);
  showRoute(null);
  byId("signals-step").dataset.state = "waiting";
  setLabel(byId("response-time"));
  setLabel(byId("response-usage"));
  responsePanel.dataset.state = "waiting";
  const replay = boardStatus.dataset.state === "replay";
  setAnswerMessage(replay ? "正在载入示例…" : "等待板端处理…");
  setRequestState(replay ? "回放中" : "板端处理中");

  const start = performance.now();
  try {
    const response = await requestJson("/api/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({prompt, classifier}),
    });
    const duration = milliseconds(performance.now() - start);
    const route = response.x1_route;
    showRoute(route);
    const content = response.choices?.[0]?.message?.content;
    if (typeof content === "string" && content.trim()) renderAnswer(content);
    else setAnswerMessage("本次请求没有返回文本。");
    responsePanel.dataset.state = "complete";
    setLabel(byId("response-time"), response.demo_mode === "replay" ? "离线回放" : `端到端 ${duration}`);
    const tokens = response.usage?.total_tokens;
    if (Number.isInteger(tokens) && tokens >= 0) setLabel(byId("response-usage"), `${tokens.toLocaleString()} tokens`);
    if (route) completedRequest = {prompt, classifier};
    rememberRoute(prompt, route);
    const version = classificationFor(route)?.model_version;
    const expected = classifiers.get(classifier);
    const mismatch = expected && typeof version === "string" && !version.startsWith(expected.version_prefix);
    setRequestState(mismatch ? "实际分类模型与所选项不同" : route ? "" : "路由记录未返回", "warning");
  } catch (error) {
    const message = error instanceof TypeError ? "本地演示服务连接中断，请重试。" : error.message;
    showRoute(null);
    responsePanel.dataset.state = "error";
    setAnswerMessage(message, true);
    setRequestState("请求失败", "error");
  } finally {
    setBusy(false);
  }
});

showLanes();
loadClassifiers().catch(() => {
  classifierSelect.title = "暂时无法读取板端模型列表";
});
checkBoard();
window.setInterval(checkBoard, 12000);
