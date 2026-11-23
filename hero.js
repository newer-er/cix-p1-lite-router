(() => {
  const canvas = document.querySelector("#hero-wave");
  const context = canvas?.getContext("2d");
  const mask = document.createElement("canvas");
  const maskContext = mask.getContext("2d", {willReadFrequently: true});
  if (!context || !maskContext) return;

  const hero = canvas.closest(".hero");
  const motion = matchMedia("(prefers-reduced-motion: reduce)");
  const wordmark = "anyint";
  const settleTime = 1800;
  const maskSamples = 4;
  const theme = getComputedStyle(hero);
  const colors = [
    [0, "--wave-bronze"],
    [.34, "--wave-copper"],
    [.67, "--wave-umber"],
    [1, "--wave-ink"],
  ].map(([position, token]) => [
    position,
    theme.getPropertyValue(token).trim().split(/\s+/).map(Number),
  ]);
  const pointer = {x: 0, y: 0, nextX: 0, nextY: 0, strength: 0, target: 0};
  let width = 0;
  let height = 0;
  let columns = [];
  let visible = false;
  let elapsed = 0;
  let lastPaint;
  let frame;

  function unit(value) {
    return Math.max(0, Math.min(1, value));
  }

  // Keep the scatter and the grain around each letter stable between frames.
  function noise(seed) {
    let bits = Math.imul(seed ^ (seed >>> 16), 0x45d9f3b);
    bits = Math.imul(bits ^ (bits >>> 16), 0x45d9f3b);
    return ((bits ^ (bits >>> 16)) >>> 0) / 4294967296;
  }

  function colorAt(position) {
    const right = colors.findIndex(([stop], index) => index > 0 && stop >= position);
    const [from, start] = colors[right - 1];
    const [to, end] = colors[right];
    const fraction = (position - from) / (to - from);
    const channels = start.map((channel, index) => Math.round(channel + (end[index] - channel) * fraction));
    return `rgb(${channels.join(" ")})`;
  }

  function makeMask(pitch) {
    const fontAt = (size) => `900 ${size}px "Arial Black", Arial, sans-serif`;
    const referenceSize = 100;
    const availableWidth = Math.min(width * (width < 640 ? .9 : .82), 950);
    const availableHeight = height * .65;
    maskContext.font = fontAt(referenceSize);
    const reference = maskContext.measureText(wordmark);
    const widthScale = Math.floor(availableWidth / pitch) / (reference.actualBoundingBoxLeft + reference.actualBoundingBoxRight);
    const heightScale = Math.floor(availableHeight / pitch) / (reference.actualBoundingBoxAscent + reference.actualBoundingBoxDescent);
    const font = fontAt(referenceSize * maskSamples * Math.min(widthScale, heightScale));
    maskContext.font = font;

    const bounds = maskContext.measureText(wordmark);
    const textWidth = bounds.actualBoundingBoxLeft + bounds.actualBoundingBoxRight;
    const textHeight = bounds.actualBoundingBoxAscent + bounds.actualBoundingBoxDescent;
    const gridWidth = Math.ceil(textWidth / maskSamples);
    const gridHeight = Math.ceil(textHeight / maskSamples);
    mask.width = gridWidth * maskSamples;
    mask.height = gridHeight * maskSamples;
    maskContext.font = font;
    maskContext.fillText(
      wordmark,
      bounds.actualBoundingBoxLeft + (mask.width - textWidth) / 2,
      bounds.actualBoundingBoxAscent + (mask.height - textHeight) / 2,
    );
    const alpha = maskContext.getImageData(0, 0, mask.width, mask.height).data;
    return {gridWidth, gridHeight, alpha};
  }

  function tileCoverage(alpha, stride, column, row) {
    let sum = 0;
    for (let y = 0; y < maskSamples; y++) {
      const offset = ((row * maskSamples + y) * stride + column * maskSamples) * 4 + 3;
      for (let x = 0; x < maskSamples; x++) {
        sum += alpha[offset + x * 4];
      }
    }
    return sum / (255 * maskSamples ** 2);
  }

  function makeTile(y, size, opacity, seed) {
    return {
      y,
      size,
      opacity,
      lag: noise(seed) * .23,
      offsetX: (noise(seed + 1) - .5) * Math.min(width * .56, 540),
      offsetY: (noise(seed + 2) - .5) * height * .52,
    };
  }

  function buildColumns(pitch) {
    const {gridWidth, gridHeight, alpha} = makeMask(pitch);
    const left = Math.round((width - gridWidth * pitch) / 2);
    const top = Math.round((height - gridHeight * pitch) * (width < 640 ? .42 : .48));
    const size = pitch - (pitch > 8 ? 2 : 1);
    const result = [];

    for (let column = 0; column < gridWidth; column++) {
      const position = (column + .5) / gridWidth;
      const seed = (column + 1) * 6197;
      const tiles = [];
      for (let row = 0; row < gridHeight; row++) {
        const coverage = tileCoverage(alpha, gridWidth * maskSamples, column, row);
        if (coverage < .48) continue;
        tiles.push(makeTile(top + row * pitch, size, .69 + .28 * coverage, seed + row * 17));
      }
      if (!tiles.length) continue;

      const topEdge = tiles[0].y;
      const bottomEdge = tiles.at(-1).y;
      const fleck = Math.max(2, Math.round(size * .65));
      for (let row = 1; row <= 7; row++) {
        const grain = seed + 1800 + row * 19;
        if (noise(grain) < .43 + row * .057) continue;
        tiles.push(makeTile(bottomEdge + row * pitch, fleck, .22 * (1 - row / 9), grain));
      }
      if (noise(seed + 3200) > .67) {
        const distance = 2 + Math.floor(noise(seed + 3201) * 4);
        tiles.push(makeTile(topEdge - distance * pitch, fleck, .19, seed + 3202));
      }

      result.push({x: left + column * pitch, pitch, position, tiles, color: colorAt(position)});
    }
    return result;
  }

  function paint() {
    const reduced = motion.matches;
    const time = reduced ? 0 : elapsed / 1000;
    const progress = reduced ? 1 : unit(elapsed / settleTime);
    const radius = Math.max(48, Math.min(118, width * .17));
    context.clearRect(0, 0, width, height);

    for (const {x, pitch, position, tiles, color} of columns) {
      const horizontal = x + pitch / 2 - pointer.x;
      const drift = reduced ? 0 : Math.sin(position * 15 - time * 1.6) * pitch * .13;
      const pulse = .77 + .25 * Math.cos(horizontal / 24 - time * 3.6);
      context.fillStyle = color;

      for (const {y, size, opacity, lag, offsetX, offsetY} of tiles) {
        const spread = (1 - unit((progress - lag) / (1 - lag))) ** 2;
        const vertical = (y + size / 2 - pointer.y) * 1.15;
        const proximity = reduced ? 0 : pointer.strength * Math.exp(-(horizontal ** 2 + vertical ** 2) / radius ** 2);
        const moveX = Math.round(x + offsetX * spread + horizontal * proximity * .045);
        const moveY = Math.round(y + offsetY * spread + drift - proximity * pitch * pulse);
        const fade = unit((height - moveY) / (pitch * 4));
        context.globalAlpha = unit(opacity * (1 - .7 * spread) * fade * (1 + proximity * .12));
        context.fillRect(moveX, moveY, size, size);
      }
    }
    context.globalAlpha = 1;
  }

  function tick(now) {
    if (lastPaint === undefined || now - lastPaint >= 31) {
      const delta = lastPaint === undefined ? 0 : Math.min(now - lastPaint, 75);
      elapsed += delta;
      lastPaint = now;
      const step = 1 - Math.exp(-delta / 120);
      pointer.x += (pointer.nextX - pointer.x) * step;
      pointer.y += (pointer.nextY - pointer.y) * step;
      pointer.strength += (pointer.target - pointer.strength) * step;
      paint();
    }
    frame = requestAnimationFrame(tick);
  }

  function updateAnimation() {
    const running = visible && !document.hidden && !motion.matches && columns.length > 0;
    if (running && frame === undefined) {
      frame = requestAnimationFrame(tick);
    } else if (!running && frame !== undefined) {
      cancelAnimationFrame(frame);
      frame = undefined;
      lastPaint = undefined;
    }
  }

  function resize() {
    const bounds = canvas.getBoundingClientRect();
    width = bounds.width;
    height = bounds.height;
    const density = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * density);
    canvas.height = Math.round(height * density);
    context.setTransform(density, 0, 0, density, 0, 0);
    const pitch = Math.max(5, Math.min(11, Math.round(width / 140)));
    columns = width && height ? buildColumns(pitch) : [];
    paint();
    updateAnimation();
  }

  hero.addEventListener("pointermove", (event) => {
    if (event.pointerType === "touch" || motion.matches) return;
    const bounds = canvas.getBoundingClientRect();
    pointer.nextX = event.clientX - bounds.left;
    pointer.nextY = event.clientY - bounds.top;
    pointer.target = Number(pointer.nextY >= 0 && pointer.nextY < height);
  }, {passive: true});
  hero.addEventListener("pointerleave", () => { pointer.target = 0; });
  document.addEventListener("visibilitychange", updateAnimation);
  motion.addEventListener("change", () => {
    pointer.target = 0;
    pointer.strength = 0;
    paint();
    updateAnimation();
  });
  new ResizeObserver(resize).observe(canvas);
  new IntersectionObserver(([entry]) => {
    visible = entry.intersectionRatio >= .3;
    updateAnimation();
  }, {threshold: .3}).observe(canvas);
})();
