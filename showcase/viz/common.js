// Shared by the layer view and its fly-through: the traced tensors,
// colour maps, small 2D plots, and the voxel blocks the 3D views are built from.

import * as THREE from 'three';
import { OrbitControls } from '../vendor/three/OrbitControls.js';

export { THREE };

// ---------------------------------------------------------------- data

// the data files are checked with the host on every load (a quick 304 when unchanged), so a new
// copy shows at once instead of after the host's ten-minute cache
const FRESH = { cache: 'no-cache' };

export async function loadTrace() {
  const [manifest, buffer] = await Promise.all([
    fetch('../assets/trace.json', FRESH).then((r) => r.json()),
    fetch('../assets/trace.bin', FRESH).then((r) => r.arrayBuffer()),
  ]);
  const all = new Float32Array(buffer);
  const tensor = (key) => {
    const meta = manifest.tensors[key];
    if (!meta) throw new Error(`no tensor ${key}`);
    return { key, ...meta, data: all.subarray(meta.offset, meta.offset + meta.length) };
  };
  return { manifest, tensor };
}

// the imagery and road-network pipelines upstream of the inputs, from trace_upstream.py: floats as
// for the trace, and the language model's hidden states as bytes
export async function loadUpstream() {
  const [manifest, floats, bytes] = await Promise.all([
    fetch('../assets/upstream.json', FRESH).then((r) => r.json()),
    fetch('../assets/upstream.bin', FRESH).then((r) => r.arrayBuffer()),
    fetch('../assets/upstream_u8.bin', FRESH).then((r) => r.arrayBuffer()),
  ]);
  const all = new Float32Array(floats), raw = new Uint8Array(bytes);
  const tensor = (key) => { const m = manifest.tensors[key]; return { key, ...m, data: all.subarray(m.offset, m.offset + m.length) }; };
  const block = (key) => { const m = manifest.bytes[key]; return { key, ...m, data: raw.subarray(m.offset, m.offset + m.length) }; };
  return { manifest, tensor, block };
}

export function sampleLabel(sample) {
  const date = new Date(`${sample.date}T00:00:00`);
  const day = date.toLocaleDateString('en-AU', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' });
  return `${sample.junction} ${sample.name} · ${sample.road_class} · ${day}`;
}

export const KIND_COLOUR = {
  input: '#5cc8e6', embedding: '#c3a3f2', linear: '#6ea8fe', reshape: '#9aa0ad',
  conv: '#b388ff', norm: '#62c99a', add: '#ffb35c', output: '#ff7a7a',
  source: '#4cbf8c', pretrained: '#f0a73a', reduce: '#e98bbd',
};

export const shapeText = (shape) => shape.join('×');

// ---------------------------------------------------------------- colour

export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

// Google's Turbo, as the published polynomial fit
export function turbo(t) {
  t = clamp(t, 0, 1);
  const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return [clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1)];
}

// 2nd to 98th percentile, so a few extreme values do not wash out the rest
export function robustRange(data) {
  const step = Math.max(1, Math.floor(data.length / 4000));
  const sample = [];
  for (let i = 0; i < data.length; i += step) if (Number.isFinite(data[i])) sample.push(data[i]);
  if (!sample.length) return [0, 1];
  sample.sort((a, b) => a - b);
  let lo = sample[Math.floor(sample.length * 0.02)];
  let hi = sample[Math.min(sample.length - 1, Math.floor(sample.length * 0.98))];
  if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
  return [lo, hi];
}

export const normaliser = ([lo, hi]) => (v) => clamp((v - lo) / (hi - lo), 0, 1);

export function fmt(v, digits = 3) {
  if (!Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return v.toFixed(digits);
}

// ---------------------------------------------------------------- 2D plots

// rows x cols matrix as a pixel image; a 3D weight is shown slice by slice side by side
export function asMatrix(t) {
  const s = t.shape;
  if (s.length === 1) return { rows: 1, cols: s[0], at: (r, c) => t.data[c] };
  if (s.length === 2) return { rows: s[0], cols: s[1], at: (r, c) => t.data[r * s[1] + c] };
  const [a, b, k] = s;                                        // conv weight: out x in x taps
  return { rows: a, cols: b * k + (k - 1) * 2, at: (r, c) => {
    const slice = Math.floor(c / (b + 2)), col = c % (b + 2);
    return col >= b ? NaN : t.data[(r * b + col) * k + slice];
  } };
}

export function paintHeatmap(canvas, t, range = robustRange(t.data)) {
  const m = asMatrix(t);
  const norm = normaliser(range);
  canvas.width = m.cols;
  canvas.height = m.rows;
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(m.cols, m.rows);
  for (let r = 0; r < m.rows; r++) {
    for (let c = 0; c < m.cols; c++) {
      const v = m.at(r, c);
      const i = (r * m.cols + c) * 4;
      if (!Number.isFinite(v)) { image.data[i + 3] = 0; continue; }
      const [cr, cg, cb] = turbo(norm(v));
      image.data[i] = cr * 255; image.data[i + 1] = cg * 255; image.data[i + 2] = cb * 255; image.data[i + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
}

export function paintHistogram(canvas, data, { bins = 48, background, ink } = {}) {
  const width = canvas.clientWidth || 260, height = canvas.clientHeight || 110;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio; canvas.height = height * ratio;
  const ctx = canvas.getContext('2d');
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);
  const [lo, hi] = robustRange(data);
  const span = hi - lo, counts = new Array(bins).fill(0);
  for (const v of data) {
    if (!Number.isFinite(v)) continue;
    counts[clamp(Math.floor(((v - lo) / span) * bins), 0, bins - 1)]++;
  }
  const top = Math.max(...counts, 1), w = width / bins;
  counts.forEach((n, i) => {
    const h = (n / top) * (height - 14);
    const [r, g, b] = turbo((i + 0.5) / bins);
    ctx.fillStyle = `rgb(${r * 255},${g * 255},${b * 255})`;
    ctx.fillRect(i * w + 0.5, height - 12 - h, Math.max(1, w - 1), h);
  });
  ctx.fillStyle = ink || '#8b93a7';
  ctx.font = '10px Consolas, monospace';
  ctx.textBaseline = 'bottom';
  ctx.fillText(fmt(lo, 2), 0, height);
  ctx.textAlign = 'right';
  ctx.fillText(fmt(hi, 2), width, height);
  return [lo, hi];
}

export function paintColorbar(el) {
  const stops = [];
  for (let i = 0; i <= 10; i++) {
    const [r, g, b] = turbo(i / 10);
    stops.push(`rgb(${r * 255},${g * 255},${b * 255}) ${i * 10}%`);
  }
  el.style.background = `linear-gradient(90deg, ${stops.join(',')})`;
}

// predicted, profile and recorded volumes over the 96 intervals of the day
export function paintDayChart(canvas, series, { ink = '#8b93a7', grid = 'rgba(128,128,128,0.18)' } = {}) {
  const width = canvas.clientWidth || 320, height = canvas.clientHeight || 150;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio; canvas.height = height * ratio;
  const ctx = canvas.getContext('2d');
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);
  const left = 34, bottom = 16, top = 6;
  let hi = 1;
  for (const s of series) for (const v of s.data) if (Number.isFinite(v)) hi = Math.max(hi, v);
  hi *= 1.08;
  const x = (i) => left + (i / 95) * (width - left - 4);
  const y = (v) => top + (1 - v / hi) * (height - top - bottom);
  ctx.strokeStyle = grid; ctx.lineWidth = 1; ctx.fillStyle = ink;
  ctx.font = '10px Consolas, monospace';
  for (const frac of [0, 0.5, 1]) {
    const v = hi * frac;
    ctx.beginPath(); ctx.moveTo(left, y(v)); ctx.lineTo(width - 4, y(v)); ctx.stroke();
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(Math.round(v), left - 4, y(v));
  }
  ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
  for (const h of [0, 6, 12, 18]) ctx.fillText(`${String(h).padStart(2, '0')}:00`, x(h * 4), height - 3);
  for (const s of series) {
    ctx.strokeStyle = s.colour; ctx.lineWidth = s.width || 1.6; ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    let pen = false;
    s.data.forEach((v, i) => {
      if (!Number.isFinite(v)) { pen = false; return; }
      if (pen) ctx.lineTo(x(i), y(v)); else ctx.moveTo(x(i), y(v));
      pen = true;
    });
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

// ---------------------------------------------------------------- 3D

export function makeStage(container, { background, fov = 38 } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.domElement.className = 'gl';
  container.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(background);
  const camera = new THREE.PerspectiveCamera(fov, 1, 0.1, 8000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  const resize = () => {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(container);
  resize();
  return { renderer, scene, camera, controls, resize };
}

export function addLights(scene, { dark = true } = {}) {
  scene.add(new THREE.HemisphereLight(0xffffff, dark ? 0x202634 : 0xcfd8e4, dark ? 1.1 : 1.35));
  const key = new THREE.DirectionalLight(0xffffff, dark ? 1.5 : 1.6);
  key.position.set(-120, 220, 160);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.45);
  fill.position.set(160, 80, -140);
  scene.add(fill);
}

// Point the camera at a box from a direction, far enough back to see all of it.
export function frame(camera, controls, box, direction = new THREE.Vector3(-0.9, 0.6, 1.1), margin = 1.25) {
  const size = box.getSize(new THREE.Vector3());
  const centre = box.getCenter(new THREE.Vector3());
  const radius = size.length() / 2;
  // the narrower of the two fields of view decides, so a tall window does not cut off the sides
  const vertical = (camera.fov * Math.PI) / 180, horizontal = 2 * Math.atan(Math.tan(vertical / 2) * camera.aspect);
  const distance = (radius * margin) / Math.sin(Math.min(vertical, horizontal) / 2);
  return { position: centre.clone().add(direction.clone().normalize().multiplyScalar(distance)), target: centre };
}

// Ease the camera and orbit target to a pose over `ms`.
export function flyTo(camera, controls, pose, ms = 1400) {
  const fromPos = camera.position.clone(), fromTarget = controls.target.clone();
  const start = performance.now();
  return new Promise((resolve) => {
    const tick = () => {
      const k = clamp((performance.now() - start) / ms, 0, 1);
      const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
      camera.position.lerpVectors(fromPos, pose.position, e);
      controls.target.lerpVectors(fromTarget, pose.target, e);
      if (k < 1) requestAnimationFrame(tick); else resolve();
    };
    tick();
  });
}

const BOX = new THREE.BoxGeometry(1, 1, 1);

// A tensor as a block of cubes. Layouts:
//   column  1D along y (a vector of channels)       row   1D along z (a day of intervals)
//   plane   2D, first axis down y, second along z   bars  1D along z, height from the value
//   field   2D laid flat: second axis along x, first along z, height from the value
//   slices  3D conv weight: each tap a plane (out down y, in along z), taps stepped along x
//   order: 'columns' puts a plane's instances column by column (interval by interval), so that
//   drawing only the first mesh.count of them reveals the day from left to right
//   thick widens columns, rows and bars across their length, so a vector reads as a solid bar
export function voxels(t, { layout, cell = 1, fill = 0.82, colour, height = 10, material, order, thick = 1 } = {}) {
  const s = t.shape;
  const range = robustRange(t.data);
  const norm = normaliser(range);
  const zero = normaliser(range)(0);
  const count = t.data.length;
  const mesh = new THREE.InstancedMesh(BOX, material || new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.05 }), count);
  const m = new THREE.Matrix4(), c = new THREE.Color(), q = new THREE.Quaternion();
  const pos = new THREE.Vector3(), scl = new THREE.Vector3();
  const size = cell * fill;
  let i = 0;
  const put = (x, y, z, sx, sy, sz, v) => {
    pos.set(x, y, z); scl.set(sx, sy, sz);
    m.compose(pos, q, scl);
    mesh.setMatrixAt(i, m);
    const [r, g, b] = colour ? colour(v, norm(v)) : turbo(norm(v));
    c.setRGB(r, g, b, THREE.SRGBColorSpace);
    mesh.setColorAt(i, c);
    i++;
  };
  const wide = size * thick;
  if (layout === 'column') {
    const n = s[0];
    for (let k = 0; k < n; k++) put(0, ((n - 1) / 2 - k) * cell, 0, wide, size, wide, t.data[k]);
  } else if (layout === 'row' || layout === 'bars') {
    const n = s[s.length - 1];
    for (let k = 0; k < n; k++) {
      const v = t.data[k];
      if (layout === 'bars') {
        const h = Math.max(0.05, (Number.isFinite(v) ? v : 0) / Math.max(range[1], 1e-6)) * height;
        put(0, h / 2, (k - (n - 1) / 2) * cell, wide, h, size, v);
      } else put(0, 0, (k - (n - 1) / 2) * cell, wide, wide, size, v);
    }
  } else if (layout === 'plane') {
    const [rows, cols] = s;
    const cube = (r, k) => put(0, ((rows - 1) / 2 - r) * cell, (k - (cols - 1) / 2) * cell, size, size, size, t.data[r * cols + k]);
    if (order === 'columns') { for (let k = 0; k < cols; k++) for (let r = 0; r < rows; r++) cube(r, k); }
    else { for (let r = 0; r < rows; r++) for (let k = 0; k < cols; k++) cube(r, k); }
  } else if (layout === 'field') {
    const [rows, cols] = s;
    for (let r = 0; r < rows; r++)
      for (let k = 0; k < cols; k++) {
        const v = t.data[r * cols + k];
        const h = Math.max(0.12 * cell, Math.abs(norm(v) - zero) * height);
        put((k - (cols - 1) / 2) * cell, (norm(v) >= zero ? h / 2 : -h / 2), (r - (rows - 1) / 2) * cell, size, h, size, v);
      }
  } else if (layout === 'slices') {
    const [out, inn, taps] = s;
    for (let tap = 0; tap < taps; tap++)
      for (let o = 0; o < out; o++)
        for (let k = 0; k < inn; k++)
          put((tap - (taps - 1) / 2) * cell * 14, ((out - 1) / 2 - o) * cell, (k - (inn - 1) / 2) * cell, size * 0.9, size, size, t.data[(o * inn + k) * taps + tap]);
  }
  mesh.count = i;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.userData.range = range;
  return mesh;
}

// dots that travel along a polyline, for showing data moving between layers
export class Flow {
  constructor(scene, points, { colour = 0x58a6ff, count = 6, speed = 0.18, size = 1.1 } = {}) {
    this.curve = new THREE.CatmullRomCurve3(points, false, 'centripetal');
    this.speed = speed;
    this.offsets = Array.from({ length: count }, (_, k) => k / count);
    const geometry = new THREE.SphereGeometry(size, 12, 10);
    this.material = new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: 0.9 });
    this.dots = this.offsets.map(() => { const d = new THREE.Mesh(geometry, this.material); scene.add(d); return d; });
    const line = new THREE.BufferGeometry().setFromPoints(this.curve.getPoints(60));
    this.lineMaterial = new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity: 0.35 });
    this.line = new THREE.Line(line, this.lineMaterial);
    scene.add(this.line);
  }
  update(time) {
    this.dots.forEach((d, k) => {
      const u = (((this.offsets[k] + time * this.speed) % 1) + 1) % 1;
      d.position.copy(this.curve.getPointAt(u));
    });
  }
  set visible(v) { this.dots.forEach((d) => { d.visible = v; }); this.line.visible = v; }
  set opacity(v) { this.material.opacity = 0.9 * v; this.lineMaterial.opacity = 0.35 * v; }
}

// HTML labels pinned to points in the scene. While some objects are in focus, the dimmed labels
// that fall behind one of them are hidden rather than drawn across it.
export class Labels {
  constructor(container, camera) {
    this.container = container; this.camera = camera; this.items = [];
    this.occluders = [];
    this.ray = new THREE.Ray();
  }
  add(point, html) {
    const el = document.createElement('div');
    el.className = 'label3d';
    el.innerHTML = html;
    this.container.appendChild(el);
    this.items.push({ point: point.clone(), el });
    return el;
  }
  update() {
    const w = this.container.clientWidth, h = this.container.clientHeight, v = new THREE.Vector3(), hit = new THREE.Vector3();
    const eye = this.camera.position;
    for (const { point, el } of this.items) {
      v.copy(point).project(this.camera);
      const hidden = v.z > 1 || v.x < -1.2 || v.x > 1.2 || v.y < -1.2 || v.y > 1.2;
      el.style.display = hidden ? 'none' : '';
      el.style.left = `${((v.x + 1) / 2) * w}px`;
      el.style.top = `${((1 - v.y) / 2) * h}px`;
      if (hidden || el.dataset.dim !== '1') { el.classList.remove('behind'); continue; }
      const far = eye.distanceTo(point) - 1;
      this.ray.origin.copy(eye);
      this.ray.direction.copy(point).sub(eye).normalize();
      el.classList.toggle('behind', this.occluders.some((box) =>
        !box.containsPoint(point) && this.ray.intersectBox(box, hit) !== null && eye.distanceTo(hit) < far));
    }
  }
}
