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
  <title>Adaptive Bin Pool Viewer</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f8fb;
      color: #17202a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(180deg, rgba(232, 238, 247, 0.72), rgba(255, 255, 255, 0) 320px),
        #f7f9fc;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 18px;
      padding: 18px 22px;
      background: rgba(255, 255, 255, 0.86);
      border-bottom: 1px solid #d9e2ef;
      box-shadow: 0 12px 32px rgba(32, 47, 71, 0.08);
      backdrop-filter: blur(16px);
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 18px;
      padding: 18px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 720;
      letter-spacing: 0;
      color: #101828;
    }
    h2 {
      margin: 0;
      font-size: 15px;
      font-weight: 700;
      letter-spacing: 0;
      color: #1f2937;
    }
    .panel {
      background: linear-gradient(180deg, #ffffff, #f9fbff);
      border: 1px solid #dce5f2;
      border-radius: 8px;
      box-shadow:
        0 22px 48px rgba(25, 39, 64, 0.10),
        0 2px 4px rgba(25, 39, 64, 0.05),
        inset 0 1px 0 rgba(255, 255, 255, 0.95);
    }
    .chart-panel { padding: 14px; }
    .chart-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 10px;
    }
    .chart-shell {
      position: relative;
      overflow: hidden;
      border: 1px solid #cfd9e8;
      border-radius: 8px;
      background: #f8fafc;
      box-shadow:
        inset 0 1px 2px rgba(15, 23, 42, 0.08),
        inset 0 -12px 28px rgba(15, 23, 42, 0.04);
    }
    canvas {
      display: block;
      width: 100%;
      height: 340px;
      image-rendering: pixelated;
      cursor: grab;
      touch-action: none;
    }
    canvas.dragging { cursor: grabbing; }
    .stack { display: grid; gap: 18px; min-width: 0; }
    .meta, .hint, .toolbar { color: #667085; font-size: 12px; }
    .summary-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border: 1px solid #d7e0ec;
      border-radius: 999px;
      background: #ffffff;
      box-shadow: 0 6px 16px rgba(25, 39, 64, 0.06);
      white-space: nowrap;
    }
    .toolbar { display: flex; align-items: center; gap: 8px; }
    button {
      height: 30px;
      padding: 0 10px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: linear-gradient(180deg, #ffffff, #eef4fb);
      color: #1f2937;
      font: inherit;
      font-size: 12px;
      box-shadow: 0 4px 10px rgba(25, 39, 64, 0.08);
      cursor: pointer;
    }
    button:active {
      transform: translateY(1px);
      box-shadow: inset 0 2px 4px rgba(25, 39, 64, 0.12);
    }
    aside.panel {
      position: sticky;
      top: 88px;
      align-self: start;
      padding: 14px;
    }
    .details {
      margin-top: 12px;
      padding: 12px;
      min-height: 280px;
      border: 1px solid #d9e2ef;
      border-radius: 8px;
      background: #ffffff;
      box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.06);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      line-height: 1.55;
      color: #243042;
    }
    .error { color: #b42318; border-color: #fecdca; background: #fff6f5; }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      aside.panel { position: static; }
      canvas { height: 300px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Adaptive Bin Pool Viewer</h1>
      <div class="meta" id="summary">Waiting for snapshot...</div>
    </div>
    <div class="summary-pill" id="viewSummary">full view</div>
  </header>
  <main>
    <div class="stack">
      <section class="panel chart-panel">
        <div class="chart-head">
          <h2>Failure Rate</h2>
          <div class="toolbar"><button id="resetView" type="button">Reset View</button></div>
        </div>
        <div class="chart-shell"><canvas id="failure"></canvas></div>
      </section>
      <section class="panel chart-panel">
        <div class="chart-head">
          <h2>Access Count</h2>
          <div class="toolbar">log scale</div>
        </div>
        <div class="chart-shell"><canvas id="access"></canvas></div>
      </section>
    </div>
    <aside class="panel">
      <h2>Hover Details</h2>
      <div class="details" id="details">Move over a heatmap cell.</div>
    </aside>
  </main>
<script>
const state = {
  meta: null,
  access: null,
  failure: null,
  valid: null,
  lastKey: "",
  offscreen: { failure: null, access: null },
  view: { x: 0, y: 0, w: 1, h: 1 },
  dragging: null,
};
const summary = document.getElementById("summary");
const viewSummary = document.getElementById("viewSummary");
const details = document.getElementById("details");
const failureCanvas = document.getElementById("failure");
const accessCanvas = document.getElementById("access");
const resetView = document.getElementById("resetView");

function colorFailure(v, valid) {
  if (!valid) return [232, 237, 244, 255];
  const x = Math.max(0, Math.min(1, v));
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

function colorAccess(v, vmax, valid) {
  if (!valid) return [232, 237, 244, 255];
  const x = vmax > 0 ? Math.log1p(v) / Math.log1p(vmax) : 0;
  return [
    Math.floor(245 - 205 * x),
    Math.floor(248 - 120 * x),
    Math.floor(252 - 40 * x),
    255,
  ];
}

function buildOffscreen(mode) {
  const meta = state.meta;
  if (!meta || !state.access || !state.failure || !state.valid) return;
  const w = meta.bucket_count;
  const h = meta.bin_count;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(w, h);
  let maxAccess = 0;
  for (let i = 0; i < state.access.length; i++) maxAccess = Math.max(maxAccess, state.access[i]);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const src = x * h + y;
      const dst = ((h - 1 - y) * w + x) * 4;
      const valid = state.valid[src] > 0;
      const access = state.access[src];
      const failure = state.failure[src];
      const color = mode === "failure"
        ? colorFailure(access > 0 ? failure / access : 0, valid)
        : colorAccess(access, maxAccess, valid);
      image.data[dst] = color[0];
      image.data[dst + 1] = color[1];
      image.data[dst + 2] = color[2];
      image.data[dst + 3] = color[3];
    }
  }
  ctx.putImageData(image, 0, 0);
  state.offscreen[mode] = canvas;
}

function setFullView() {
  if (!state.meta) return;
  state.view = { x: 0, y: 0, w: state.meta.bucket_count, h: state.meta.bin_count };
  drawAll();
}

function clampView() {
  const meta = state.meta;
  if (!meta) return;
  const minW = Math.max(8, meta.bucket_count / 256);
  const minH = Math.max(4, meta.bin_count / 64);
  state.view.w = Math.max(minW, Math.min(meta.bucket_count, state.view.w));
  state.view.h = Math.max(minH, Math.min(meta.bin_count, state.view.h));
  state.view.x = Math.max(0, Math.min(meta.bucket_count - state.view.w, state.view.x));
  state.view.y = Math.max(0, Math.min(meta.bin_count - state.view.h, state.view.y));
}

function drawCanvas(canvas, mode) {
  const source = state.offscreen[mode];
  if (!source || !state.meta) return;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * ratio));
  const height = Math.max(1, Math.floor(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, width, height);
  clampView();
  ctx.drawImage(
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
  ctx.strokeStyle = "rgba(15, 23, 42, 0.14)";
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
}

function drawAll() {
  drawCanvas(failureCanvas, "failure");
  drawCanvas(accessCanvas, "access");
  if (state.meta) {
    const x0 = Math.floor(state.view.x);
    const x1 = Math.ceil(state.view.x + state.view.w);
    const y0 = Math.max(0, state.meta.bin_count - Math.ceil(state.view.y + state.view.h));
    const y1 = Math.min(state.meta.bin_count, state.meta.bin_count - Math.floor(state.view.y));
    viewSummary.textContent = `bucket ${x0}-${x1} | bin ${y0}-${y1}`;
  }
}

async function fetchBinary(name, type, key) {
  const res = await fetch(`${name}?v=${key}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`failed to fetch ${name}: ${res.status}`);
  return new type(await res.arrayBuffer());
}

async function refresh() {
  try {
    const res = await fetch(`latest.json?t=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) throw new Error("latest.json not found");
    const meta = await res.json();
    const key = `${meta.iteration}-${meta.updated_at_unix}`;
    if (key === state.lastKey) return;
    state.meta = meta;
    state.access = await fetchBinary(meta.access_file, Float32Array, key);
    state.failure = await fetchBinary(meta.failure_file, Float32Array, key);
    state.valid = await fetchBinary(meta.valid_file, Int32Array, key);
    state.lastKey = key;
    buildOffscreen("failure");
    buildOffscreen("access");
    if (state.view.w === 1 && state.view.h === 1) {
      setFullView();
    } else {
      drawAll();
    }
    summary.textContent = `iteration ${meta.iteration} | motions ${meta.num_files} | buckets ${meta.bucket_count} | bins ${meta.bin_count}`;
    summary.className = "meta";
  } catch (err) {
    summary.textContent = String(err);
    summary.className = "meta error";
  }
}

function canvasPointToCell(canvas, evt) {
  const meta = state.meta;
  if (!meta) return null;
  const rect = canvas.getBoundingClientRect();
  const relX = Math.max(0, Math.min(1, (evt.clientX - rect.left) / rect.width));
  const relY = Math.max(0, Math.min(1, (evt.clientY - rect.top) / rect.height));
  const drawX = Math.max(0, Math.min(meta.bucket_count - 1, Math.floor(state.view.x + relX * state.view.w)));
  const drawY = Math.max(0, Math.min(meta.bin_count - 1, Math.floor(state.view.y + relY * state.view.h)));
  return { x: drawX, y: meta.bin_count - 1 - drawY };
}

function showHover(evt) {
  const meta = state.meta;
  if (!meta || !state.access || !state.failure || !state.valid) return;
  const point = canvasPointToCell(evt.currentTarget, evt);
  if (!point) return;
  const x = point.x;
  const y = point.y;
  const idx = x * meta.bin_count + y;
  const access = state.access[idx];
  const failure = state.failure[idx];
  const rate = access > 0 ? failure / access : 0;
  const step0 = y * meta.bin_width_steps;
  const step1 = (y + 1) * meta.bin_width_steps;
  details.textContent =
    `bucket: ${x}\n` +
    `motion ids: [${meta.bucket_start_motion_ids[x]}, ${meta.bucket_end_motion_ids[x]})\n` +
    `bin: ${y}\n` +
    `step range: [${step0}, ${step1})\n` +
    `valid motions in cell: ${state.valid[idx]}\n` +
    `access count: ${access.toFixed(4)}\n` +
    `failure count: ${failure.toFixed(4)}\n` +
    `failure rate: ${rate.toFixed(4)}\n\n` +
    `first path:\n${meta.bucket_first_paths[x] || "(none)"}\n\n` +
    `last path:\n${meta.bucket_last_paths[x] || "(none)"}`;
}

function onPointerDown(evt) {
  if (!state.meta) return;
  evt.currentTarget.setPointerCapture(evt.pointerId);
  evt.currentTarget.classList.add("dragging");
  state.dragging = {
    id: evt.pointerId,
    x: evt.clientX,
    y: evt.clientY,
    viewX: state.view.x,
    viewY: state.view.y,
  };
}

function onPointerMove(evt) {
  showHover(evt);
  if (!state.dragging || state.dragging.id !== evt.pointerId) return;
  const rect = evt.currentTarget.getBoundingClientRect();
  const dx = (evt.clientX - state.dragging.x) / rect.width * state.view.w;
  const dy = (evt.clientY - state.dragging.y) / rect.height * state.view.h;
  state.view.x = state.dragging.viewX - dx;
  state.view.y = state.dragging.viewY - dy;
  clampView();
  drawAll();
}

function onPointerUp(evt) {
  evt.currentTarget.classList.remove("dragging");
  if (state.dragging && state.dragging.id === evt.pointerId) state.dragging = null;
}

function onWheel(evt) {
  if (!state.meta) return;
  evt.preventDefault();
  const rect = evt.currentTarget.getBoundingClientRect();
  const relX = Math.max(0, Math.min(1, (evt.clientX - rect.left) / rect.width));
  const relY = Math.max(0, Math.min(1, (evt.clientY - rect.top) / rect.height));
  const anchorX = state.view.x + relX * state.view.w;
  const anchorY = state.view.y + relY * state.view.h;
  const scale = evt.deltaY < 0 ? 0.82 : 1.22;
  state.view.w *= scale;
  state.view.h *= scale;
  state.view.x = anchorX - relX * state.view.w;
  state.view.y = anchorY - relY * state.view.h;
  clampView();
  drawAll();
}

for (const canvas of [failureCanvas, accessCanvas]) {
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointercancel", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });
}
resetView.addEventListener("click", setFullView);
window.addEventListener("resize", drawAll);
refresh();
setInterval(refresh, 2000);
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
  parser = argparse.ArgumentParser(description="Serve adaptive bin pool snapshots.")
  parser.add_argument("snapshot_dir", type=Path)
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8765)
  args = parser.parse_args()

  snapshot_dir = args.snapshot_dir.expanduser().resolve()
  snapshot_dir.mkdir(parents=True, exist_ok=True)
  handler = functools.partial(ViewerRequestHandler, directory=str(snapshot_dir))
  server = ThreadingHTTPServer((args.host, args.port), handler)
  print(f"Adaptive bin viewer: http://{args.host}:{args.port}")
  print(f"Serving snapshots from: {snapshot_dir}")
  server.serve_forever()


if __name__ == "__main__":
  main()
