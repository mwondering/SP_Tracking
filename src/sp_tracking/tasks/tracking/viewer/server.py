from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Adaptive Motion Sampling Viewer</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7fb;
      color: #17202a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(180deg, rgba(226, 234, 247, 0.8), rgba(255, 255, 255, 0) 360px),
        #f7f9fc;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 17px 22px;
      border-bottom: 1px solid #d8e1ee;
      background: rgba(255, 255, 255, 0.9);
      box-shadow: 0 10px 30px rgba(30, 44, 68, 0.08);
      backdrop-filter: blur(14px);
    }
    h1, h2, h3 { margin: 0; color: #101828; }
    h1 { font-size: 20px; }
    h2 { font-size: 16px; }
    h3 { font-size: 13px; }
    .meta, .hint, .status { color: #667085; font-size: 12px; }
    .summary-pill {
      padding: 7px 11px;
      border: 1px solid #d5dfec;
      border-radius: 999px;
      background: #fff;
      box-shadow: 0 5px 14px rgba(25, 39, 64, 0.06);
      white-space: nowrap;
    }
    #panels {
      display: grid;
      gap: 18px;
      padding: 18px;
    }
    .rank-panel {
      min-width: 0;
      overflow: hidden;
      border: 1px solid #d9e3f0;
      border-radius: 10px;
      background: linear-gradient(180deg, #fff, #f9fbff);
      box-shadow: 0 18px 42px rgba(25, 39, 64, 0.09);
    }
    .rank-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 13px 15px;
      border-bottom: 1px solid #e2e8f0;
      background: rgba(248, 250, 252, 0.9);
    }
    .rank-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) 310px;
      gap: 14px;
      padding: 14px;
    }
    .chart {
      min-width: 0;
    }
    .chart-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
    }
    .chart-shell {
      overflow: hidden;
      border: 1px solid #ced9e7;
      border-radius: 8px;
      background: #f7f9fc;
      box-shadow: inset 0 1px 3px rgba(15, 23, 42, 0.08);
    }
    canvas {
      display: block;
      width: 100%;
      height: 310px;
      image-rendering: pixelated;
      cursor: grab;
      touch-action: none;
    }
    canvas.dragging { cursor: grabbing; }
    .details {
      min-height: 354px;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
      padding: 11px;
      border: 1px solid #d8e1ed;
      border-radius: 8px;
      background: #fff;
      color: #263246;
      font: 12px/1.52 ui-monospace, SFMono-Regular, Menlo, monospace;
      box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.05);
    }
    button {
      height: 29px;
      padding: 0 10px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: linear-gradient(180deg, #fff, #edf3fa);
      color: #263246;
      font: inherit;
      font-size: 12px;
      cursor: pointer;
    }
    button:active { transform: translateY(1px); }
    .error { color: #b42318; }
    @media (max-width: 1180px) {
      .rank-grid { grid-template-columns: 1fr 1fr; }
      .details { grid-column: 1 / -1; min-height: 150px; }
    }
    @media (max-width: 760px) {
      header, .rank-head { align-items: flex-start; flex-direction: column; }
      .rank-grid { grid-template-columns: 1fr; }
      .details { grid-column: auto; }
      canvas { height: 280px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Adaptive Motion Sampling Viewer</h1>
      <div class="meta" id="summary">Waiting for snapshots...</div>
    </div>
    <div class="summary-pill" id="layoutSummary">discovering layout</div>
  </header>
  <main id="panels"></main>
<script>
const root = document.getElementById("panels");
const summary = document.getElementById("summary");
const layoutSummary = document.getElementById("layoutSummary");
const panelStates = new Map();
let currentLayoutKey = "";

async function fetchJson(path) {
  const separator = path.includes("?") ? "&" : "?";
  const response = await fetch(`${path}${separator}t=${Date.now()}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

async function fetchBinary(path, Type, key) {
  const separator = path.includes("?") ? "&" : "?";
  const response = await fetch(`${path}${separator}v=${encodeURIComponent(key)}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return new Type(await response.arrayBuffer());
}

async function discoverLayout() {
  try {
    const layout = await fetchJson("layout.json");
    if (layout.version === 2 && layout.layout === "per_rank_exact_motion") {
      return {
        mode: "per_rank",
        descriptors: layout.ranks.map((item) => {
          const slash = item.snapshot.lastIndexOf("/");
          return {
            id: `rank-${item.rank}`,
            rank: item.rank,
            metaPath: item.snapshot,
            basePath: slash >= 0 ? item.snapshot.slice(0, slash + 1) : "",
          };
        }),
      };
    }
  } catch (_) {
    // A v1 large-dataset directory has no layout.json.
  }
  return {
    mode: "legacy",
    descriptors: [{ id: "legacy", rank: null, metaPath: "latest.json", basePath: "" }],
  };
}

function layoutKey(layout) {
  return `${layout.mode}:${layout.descriptors.map((item) => item.id).join(",")}`;
}

function colorFailure(value, valid) {
  if (!valid) return [230, 235, 243, 255];
  const x = Math.max(0, Math.min(1, value));
  if (x < 0.5) {
    const t = x / 0.5;
    return [
      Math.floor(20 + 220 * t),
      Math.floor(145 + 55 * t),
      Math.floor(160 - 90 * t),
      255,
    ];
  }
  const t = (x - 0.5) / 0.5;
  return [
    Math.floor(240 - 45 * t),
    Math.floor(200 - 120 * t),
    Math.floor(70 - 30 * t),
    255,
  ];
}

function colorAccess(value, maxValue, valid) {
  if (!valid) return [230, 235, 243, 255];
  const x = maxValue > 0 ? Math.log1p(value) / Math.log1p(maxValue) : 0;
  return [
    Math.floor(245 - 205 * x),
    Math.floor(248 - 120 * x),
    Math.floor(252 - 40 * x),
    255,
  ];
}

function makePanel(descriptor, mode) {
  const article = document.createElement("article");
  article.className = "rank-panel";
  article.innerHTML = `
    <div class="rank-head">
      <div>
        <h2>${mode === "legacy" ? "Large-dataset aggregate" : `Rank ${descriptor.rank}`}</h2>
        <div class="status">Waiting for snapshot...</div>
      </div>
      <div>
        <button type="button" class="reset-view">Reset View</button>
        <span class="hint"> motion: short → long</span>
      </div>
    </div>
    <div class="rank-grid">
      <section class="chart">
        <div class="chart-head"><h3>Failure Rate</h3><span class="hint">linear scale</span></div>
        <div class="chart-shell"><canvas data-mode="failure"></canvas></div>
      </section>
      <section class="chart">
        <div class="chart-head"><h3>Access Count</h3><span class="hint">log scale</span></div>
        <div class="chart-shell"><canvas data-mode="access"></canvas></div>
      </section>
      <div class="details">Move over a heatmap cell.</div>
    </div>`;
  root.appendChild(article);
  const state = {
    descriptor,
    mode,
    article,
    status: article.querySelector(".status"),
    details: article.querySelector(".details"),
    canvases: {
      failure: article.querySelector('canvas[data-mode="failure"]'),
      access: article.querySelector('canvas[data-mode="access"]'),
    },
    meta: null,
    motions: null,
    access: null,
    failure: null,
    valid: null,
    offscreen: { failure: null, access: null },
    view: { x: 0, y: 0, w: 1, h: 1 },
    dragging: null,
    lastKey: "",
  };
  for (const [canvasMode, canvas] of Object.entries(state.canvases)) {
    canvas.addEventListener("pointerdown", (event) => onPointerDown(state, event));
    canvas.addEventListener("pointermove", (event) => onPointerMove(state, event));
    canvas.addEventListener("pointerup", (event) => onPointerUp(state, event));
    canvas.addEventListener("pointercancel", (event) => onPointerUp(state, event));
    canvas.addEventListener(
      "wheel",
      (event) => onWheel(state, event),
      { passive: false },
    );
    canvas.dataset.mode = canvasMode;
  }
  article.querySelector(".reset-view").addEventListener("click", () => setFullView(state));
  panelStates.set(descriptor.id, state);
  return state;
}

function ensurePanels(layout) {
  const key = layoutKey(layout);
  if (key === currentLayoutKey) return;
  currentLayoutKey = key;
  panelStates.clear();
  root.replaceChildren();
  for (const descriptor of layout.descriptors) makePanel(descriptor, layout.mode);
  layoutSummary.textContent =
    layout.mode === "per_rank"
      ? `${layout.descriptors.length} GPU/rank panels`
      : "large-dataset aggregate";
}

function columnCount(state) {
  if (!state.meta) return 0;
  return state.mode === "per_rank" ? state.meta.motion_count : state.meta.bucket_count;
}

function isValidCell(state, x, y) {
  if (state.mode === "per_rank") {
    const motion = state.motions?.motions?.[x];
    return Boolean(motion) && y < motion.valid_bin_count;
  }
  return Boolean(state.valid) && state.valid[x * state.meta.bin_count + y] > 0;
}

function buildOffscreen(state, mode) {
  const columns = columnCount(state);
  const bins = state.meta.bin_count;
  if (!columns || !bins || !state.access || !state.failure) return;
  const canvas = document.createElement("canvas");
  canvas.width = columns;
  canvas.height = bins;
  const context = canvas.getContext("2d");
  const image = context.createImageData(columns, bins);
  let maxAccess = 0;
  for (let index = 0; index < state.access.length; index++) {
    maxAccess = Math.max(maxAccess, state.access[index]);
  }
  for (let y = 0; y < bins; y++) {
    for (let x = 0; x < columns; x++) {
      const sourceIndex = x * bins + y;
      const targetIndex = ((bins - 1 - y) * columns + x) * 4;
      const valid = isValidCell(state, x, y);
      const access = state.access[sourceIndex];
      const failure = state.failure[sourceIndex];
      const color =
        mode === "failure"
          ? colorFailure(access > 0 ? failure / access : 0, valid)
          : colorAccess(access, maxAccess, valid);
      image.data[targetIndex] = color[0];
      image.data[targetIndex + 1] = color[1];
      image.data[targetIndex + 2] = color[2];
      image.data[targetIndex + 3] = color[3];
    }
  }
  context.putImageData(image, 0, 0);
  state.offscreen[mode] = canvas;
}

function setFullView(state) {
  if (!state.meta) return;
  state.view = {
    x: 0,
    y: 0,
    w: columnCount(state),
    h: state.meta.bin_count,
  };
  drawState(state);
}

function clampView(state) {
  if (!state.meta) return;
  const columns = columnCount(state);
  const bins = state.meta.bin_count;
  const minWidth = Math.min(columns, Math.max(1, columns / 256));
  const minHeight = Math.min(bins, Math.max(1, bins / 64));
  state.view.w = Math.max(minWidth, Math.min(columns, state.view.w));
  state.view.h = Math.max(minHeight, Math.min(bins, state.view.h));
  state.view.x = Math.max(0, Math.min(columns - state.view.w, state.view.x));
  state.view.y = Math.max(0, Math.min(bins - state.view.h, state.view.y));
}

function drawCanvas(state, canvas, mode) {
  const source = state.offscreen[mode];
  if (!source || !state.meta) return;
  const rectangle = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rectangle.width * ratio));
  const height = Math.max(1, Math.floor(rectangle.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  clampView(state);
  const context = canvas.getContext("2d");
  context.imageSmoothingEnabled = false;
  context.clearRect(0, 0, width, height);
  context.drawImage(
    source,
    state.view.x,
    state.view.y,
    state.view.w,
    state.view.h,
    0,
    0,
    width,
    height,
  );
  context.strokeStyle = "rgba(15, 23, 42, 0.14)";
  context.strokeRect(0.5, 0.5, width - 1, height - 1);
}

function drawState(state) {
  drawCanvas(state, state.canvases.failure, "failure");
  drawCanvas(state, state.canvases.access, "access");
}

function canvasPointToCell(state, canvas, event) {
  if (!state.meta) return null;
  const columns = columnCount(state);
  const bins = state.meta.bin_count;
  const rectangle = canvas.getBoundingClientRect();
  const relativeX = Math.max(0, Math.min(1, (event.clientX - rectangle.left) / rectangle.width));
  const relativeY = Math.max(0, Math.min(1, (event.clientY - rectangle.top) / rectangle.height));
  const x = Math.max(0, Math.min(columns - 1, Math.floor(state.view.x + relativeX * state.view.w)));
  const drawnY = Math.max(0, Math.min(bins - 1, Math.floor(state.view.y + relativeY * state.view.h)));
  return { x, y: bins - 1 - drawnY };
}

function showHover(state, event) {
  if (!state.meta || !state.access || !state.failure) return;
  const point = canvasPointToCell(state, event.currentTarget, event);
  if (!point) return;
  const { x, y } = point;
  const index = x * state.meta.bin_count + y;
  const access = state.access[index];
  const failure = state.failure[index];
  const rate = access > 0 ? failure / access : 0;
  const stepStart = y * state.meta.bin_width_steps;
  const stepEnd = (y + 1) * state.meta.bin_width_steps;
  if (state.mode === "per_rank") {
    const motion = state.motions.motions[x];
    state.details.textContent =
      `rank: ${state.meta.rank} | local rank: ${state.meta.local_rank}\n` +
      `sorted column: ${x}\n` +
      `local motion id: ${motion.local_motion_id}\n` +
      `duration: ${motion.duration_seconds.toFixed(3)} s\n` +
      `frames / fps: ${motion.length_steps} / ${motion.fps}\n` +
      `bin: ${y} | step range: [${stepStart}, ${stepEnd})\n` +
      `valid: ${isValidCell(state, x, y)}\n` +
      `access count: ${access.toFixed(4)}\n` +
      `failure count: ${failure.toFixed(4)}\n` +
      `failure rate: ${rate.toFixed(4)}\n\n` +
      `path:\n${motion.path}`;
    return;
  }
  state.details.textContent =
    `bucket: ${x}\n` +
    `motion ids: [${state.meta.bucket_start_motion_ids[x]}, ${state.meta.bucket_end_motion_ids[x]})\n` +
    `bin: ${y} | step range: [${stepStart}, ${stepEnd})\n` +
    `valid motions: ${state.valid[index]}\n` +
    `access count: ${access.toFixed(4)}\n` +
    `failure count: ${failure.toFixed(4)}\n` +
    `failure rate: ${rate.toFixed(4)}\n\n` +
    `first path:\n${state.meta.bucket_first_paths[x] || "(none)"}\n\n` +
    `last path:\n${state.meta.bucket_last_paths[x] || "(none)"}`;
}

function onPointerDown(state, event) {
  if (!state.meta) return;
  event.currentTarget.setPointerCapture(event.pointerId);
  event.currentTarget.classList.add("dragging");
  state.dragging = {
    id: event.pointerId,
    x: event.clientX,
    y: event.clientY,
    viewX: state.view.x,
    viewY: state.view.y,
  };
}

function onPointerMove(state, event) {
  showHover(state, event);
  if (!state.dragging || state.dragging.id !== event.pointerId) return;
  const rectangle = event.currentTarget.getBoundingClientRect();
  state.view.x =
    state.dragging.viewX -
    ((event.clientX - state.dragging.x) / rectangle.width) * state.view.w;
  state.view.y =
    state.dragging.viewY -
    ((event.clientY - state.dragging.y) / rectangle.height) * state.view.h;
  clampView(state);
  drawState(state);
}

function onPointerUp(state, event) {
  event.currentTarget.classList.remove("dragging");
  if (state.dragging?.id === event.pointerId) state.dragging = null;
}

function onWheel(state, event) {
  if (!state.meta) return;
  event.preventDefault();
  const rectangle = event.currentTarget.getBoundingClientRect();
  const relativeX = Math.max(0, Math.min(1, (event.clientX - rectangle.left) / rectangle.width));
  const relativeY = Math.max(0, Math.min(1, (event.clientY - rectangle.top) / rectangle.height));
  const anchorX = state.view.x + relativeX * state.view.w;
  const anchorY = state.view.y + relativeY * state.view.h;
  const scale = event.deltaY < 0 ? 0.82 : 1.22;
  state.view.w *= scale;
  state.view.h *= scale;
  state.view.x = anchorX - relativeX * state.view.w;
  state.view.y = anchorY - relativeY * state.view.h;
  clampView(state);
  drawState(state);
}

async function refreshPanel(state) {
  const descriptor = state.descriptor;
  try {
    const meta = await fetchJson(descriptor.metaPath);
    const key = `${meta.iteration}-${meta.updated_at_unix}`;
    if (key === state.lastKey) return true;
    const oldColumns = state.meta ? columnCount(state) : 0;
    const oldBins = state.meta?.bin_count || 0;
    state.meta = meta;
    if (state.mode === "per_rank") {
      const motionPath = `${descriptor.basePath}${meta.motion_metadata_file}`;
      if (!state.motions) state.motions = await fetchJson(motionPath);
      state.valid = null;
    } else {
      state.valid = await fetchBinary(
        `${descriptor.basePath}${meta.valid_file}`,
        Int32Array,
        key,
      );
    }
    state.access = await fetchBinary(
      `${descriptor.basePath}${meta.access_file}`,
      Float32Array,
      key,
    );
    state.failure = await fetchBinary(
      `${descriptor.basePath}${meta.failure_file}`,
      Float32Array,
      key,
    );
    const expectedLength = columnCount(state) * meta.bin_count;
    if (
      state.access.length !== expectedLength ||
      state.failure.length !== expectedLength
    ) {
      throw new Error(
        `snapshot shape mismatch: expected ${expectedLength}, ` +
        `access=${state.access.length}, failure=${state.failure.length}`,
      );
    }
    state.lastKey = key;
    buildOffscreen(state, "failure");
    buildOffscreen(state, "access");
    if (
      state.view.w === 1 ||
      oldColumns !== columnCount(state) ||
      oldBins !== meta.bin_count
    ) {
      setFullView(state);
    } else {
      drawState(state);
    }
    state.status.className = "status";
    if (state.mode === "per_rank") {
      const windowText =
        meta.failure_rate_window_iterations == null
          ? "cumulative window"
          : `window ${meta.failure_rate_window_iterations} iters`;
      state.status.textContent =
        `iteration ${meta.iteration} | motions ${meta.motion_count} | ` +
        `bins ${meta.bin_count} | ${windowText} | ${meta.hostname}`;
    } else {
      state.status.textContent =
        `iteration ${meta.iteration} | motions ${meta.num_files} | ` +
        `buckets ${meta.bucket_count} | bins ${meta.bin_count}`;
    }
    return true;
  } catch (error) {
    state.status.textContent = `Waiting: ${String(error)}`;
    state.status.className = "status error";
    return false;
  }
}

async function refreshAll() {
  const layout = await discoverLayout();
  ensurePanels(layout);
  const results = await Promise.all(
    Array.from(panelStates.values()).map((state) => refreshPanel(state)),
  );
  const ready = results.filter(Boolean).length;
  summary.textContent =
    layout.mode === "per_rank"
      ? `ready ${ready}/${results.length} rank snapshots`
      : ready
        ? "large-dataset snapshot ready"
        : "waiting for large-dataset snapshot";
  summary.className = ready ? "meta" : "meta error";
}

window.addEventListener("resize", () => {
  for (const state of panelStates.values()) drawState(state);
});
refreshAll();
setInterval(refreshAll, 2000);
</script>
</body>
</html>
"""


class ViewerRequestHandler(SimpleHTTPRequestHandler):
  def do_GET(self) -> None:
    if self.path in {"/", "/index.html"}:
      body = INDEX_HTML.encode("utf-8")
      self.send_response(200)
      self.send_header("Content-Type", "text/html; charset=utf-8")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)
      return
    super().do_GET()


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Serve adaptive motion sampling snapshots."
  )
  parser.add_argument("snapshot_dir", type=Path)
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8765)
  args = parser.parse_args()

  snapshot_dir = args.snapshot_dir.expanduser().resolve()
  snapshot_dir.mkdir(parents=True, exist_ok=True)
  handler = functools.partial(ViewerRequestHandler, directory=str(snapshot_dir))
  server = ThreadingHTTPServer((args.host, args.port), handler)
  print(f"Adaptive motion sampling viewer: http://{args.host}:{args.port}")
  print(f"Serving snapshots from: {snapshot_dir}")
  server.serve_forever()


if __name__ == "__main__":
  main()
