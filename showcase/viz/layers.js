// 3D layers: the network's tensors as blocks in their real shapes, input to output, with the
// imagery, road-network and land-use pipelines upstream of its inputs. Explored freely, a wave of
// light marks the order of the forward pass. The fly-through instead carries one junction-day
// through it step by step, the camera following each stage as it plays out -- the satellite image
// segmented and cut into regions, the road network written as text and read by a language model,
// vectors filling, the description laid across the day, the convolution kernel sweeping the
// intervals at its dilation -- ending on the day's volumes.
//
// ?step=N opens the fly-through on step N, &at=0..1 holds it paused at that point of the step, and
// ?theme=dark gives the dark, value-coloured look the overview's Model card is taken in.

import { THREE, loadTrace, loadUpstream, sampleLabel, makeStage, addLights, frame, flyTo, clamp, KIND_COLOUR, paintDayChart } from './common.js';
import { buildModelScene, LAYOUT, CELL } from './model-scene.js';
import { upstreamKey } from './upstream-scene.js';

const $ = (id) => document.getElementById(id);
const [{ manifest, tensor }, upstream] = await Promise.all([loadTrace(), loadUpstream()]);
const params = new URLSearchParams(location.search);
// the junction-days whose imagery and road-network text were traced upstream
const samples = manifest.samples.filter((s) => upstream.manifest.samples[upstreamKey(s)]);
const sampleIndex = clamp(Number(params.get('sample') || 0), 0, samples.length - 1);
const sample = samples[sampleIndex];
const U = upstream.manifest.samples[upstreamKey(sample)];
const dark = params.get('theme') === 'dark';
document.body.className = dark ? 'dark' : 'light';
const THEME = dark
  ? { ground: 0x0b0e14, tap: 0xffffff, tapOpacity: 0.22, out: 0x58a6ff, link: 0x9ecbff, ink: '#8b93a7',
      grid: 'rgba(128,128,128,0.18)', predicted: '#58a6ff', profile: '#f5b041', recorded: '#e6e9f0' }
  : { ground: 0xf4f7fa, tap: 0xf59e0b, tapOpacity: 0.5, out: 0x1c5cab, link: 0xd97706, ink: '#6b778a',
      grid: 'rgba(20,33,61,0.12)', predicted: '#1c5cab', profile: '#c27c0e', recorded: '#46536a' };

samples.forEach((s, i) => {
  const o = document.createElement('option');
  o.value = String(i);
  o.textContent = sampleLabel(s);
  o.selected = i === sampleIndex;
  $('sample').appendChild(o);
});
// a different day rebuilds the whole scene; a reload is the simplest clean way to do it
$('sample').addEventListener('change', () => {
  params.set('sample', $('sample').value);
  params.delete('step'); params.delete('at');
  location.search = params.toString();
});
$('weights-chip').hidden = manifest.weights === 'trained';

const stage = makeStage($('view'), { background: `#${THEME.ground.toString(16).padStart(6, '0')}` });
addLights(stage.scene, { dark });
const model = buildModelScene(stage, { sample, tensor, colouring: dark ? 'value' : 'kind', dark,
  upstream: { entry: U, tensor: upstream.tensor, block: upstream.block } });
const O = (id) => model.objects.get(id);

const home = frame(stage.camera, stage.controls, model.bounds, new THREE.Vector3(-0.3, 0.62, 1), 0.5);
stage.camera.position.copy(home.position);
stage.controls.target.copy(home.target);
// the haze starts behind the middle of the scene, however far back the whole of it needs the camera;
// the dark look keeps the closer haze it was first made with
const distance = home.position.distanceTo(home.target);
stage.scene.fog = dark ? new THREE.Fog(THEME.ground, 420, 900) : new THREE.Fog(THEME.ground, distance * 0.95, distance * 2.1);

$('params').textContent = `${manifest.parameters.toLocaleString()} parameters · ${LAYOUT.length} tensors`;
const kinds = {
  source: 'Source data', pretrained: 'Pretrained model', reduce: 'Pooling and PCA', input: 'Input',
  linear: 'Linear', reshape: 'Repeat', conv: 'Convolution', output: 'Output',
};
// in the dark look colour follows the values, so the kinds have no key
$('legend').innerHTML = dark ? '' : Object.entries(kinds)
  .map(([k, name]) => `<span><i style="background:${KIND_COLOUR[k]}"></i>${name}</span>`).join('');

// ---------------------------------------------------------------- the convolution kernel

const kernel = new THREE.Group();
const tapMaterial = new THREE.MeshBasicMaterial({ color: THEME.tap, transparent: true, opacity: THEME.tapOpacity, depthWrite: false });
const outMaterial = new THREE.MeshBasicMaterial({ color: THEME.out, transparent: true, opacity: 0.35, depthWrite: false });
const taps = [0, 1, 2].map(() => new THREE.Mesh(new THREE.BoxGeometry(1.6, 1, CELL * 1.1), tapMaterial));
const outBox = new THREE.Mesh(new THREE.BoxGeometry(1.6, 1, CELL * 1.1), outMaterial);
const links = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: THEME.link, transparent: true, opacity: 0.8 }));
links.geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(18), 3));
taps.forEach((m) => kernel.add(m));
kernel.add(outBox, links);
kernel.visible = false;
stage.scene.add(kernel);

const columnZ = (o, k) => o.def.z + (k - 47.5) * CELL;
const middle = (o) => (o.bounds.min.y + o.bounds.max.y) / 2;
const tall = (o) => o.bounds.max.y - o.bounds.min.y + 1;

function sweep(from, to, dilation, p) {
  const k = Math.min(95, Math.floor(p * 96));
  kernel.visible = true;
  const A = O(from), B = O(to);
  const points = [];
  [-dilation, 0, dilation].forEach((offset, i) => {
    const col = clamp(k + offset, 0, 95);
    taps[i].visible = k + offset >= 0 && k + offset <= 95;
    taps[i].scale.y = tall(A);
    taps[i].position.set(A.def.x, middle(A), columnZ(A, col));
    if (taps[i].visible) points.push(A.def.x + 0.8, middle(A), columnZ(A, col), B.def.x - 0.8, middle(B), columnZ(B, k));
  });
  while (points.length < 18) points.push(...points.slice(-6));
  links.geometry.attributes.position.array.set(points.slice(0, 18));
  links.geometry.attributes.position.needsUpdate = true;
  outBox.scale.y = tall(B);
  outBox.position.set(B.def.x, middle(B), columnZ(B, k));
  B.mesh.count = (k + 1) * B.tensor.shape[0];                 // the output fills in as the kernel passes
}

const reveal = (id, p) => { const o = O(id); o.mesh.count = Math.max(1, Math.round(o.full * p)); };

// ---------------------------------------------------------------- the fly-through's steps

const day = new Date(`${sample.date}T00:00:00`).toLocaleDateString('en-AU', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });
const LEFT = new THREE.Vector3(-0.75, 0.5, 1), SIDE = new THREE.Vector3(-0.8, 0.36, 0.75), HIGH = new THREE.Vector3(-0.5, 0.9, 1);
const up = model.upstream, im = U.imagery, nw = U.network, qwen = upstream.manifest.qwen;
const FLAT = new THREE.Vector3(-0.25, 1.5, 1), FRONT = new THREE.Vector3(-0.12, 0.28, 1);
const STEPS = [
  { title: 'One junction, from above', text: `${sample.name}, ${U.year}: a satellite image ${im.patch_m} m across, centred on the junction.`,
    focus: ['aerial'], dir: FLAT, margin: 0.72, run: () => up.lift(0) },
  { title: 'Segmented by SAM 3', text: 'SAM 3 marks the roads, building roofs, vegetation and parking it finds for short text prompts.',
    focus: ['aerial', 'segmentation'], dir: new THREE.Vector3(-0.4, 0.75, 1), margin: 0.78, run: (p) => up.lift(p) },
  { title: 'Cut into regions', text: `Each connected patch of one class is a region, ${im.region_count} in this image. The ${im.regions.length} largest are shown, each in its box.`,
    focus: ['aerial', 'segmentation', 'regions'], dir: new THREE.Vector3(-0.3, 0.6, 1), margin: 0.8, run: (p) => up.cut(p) },
  { title: 'Described by DINOv3', text: 'A 12-layer vision transformer turns each box into 768 numbers, and principal components keep 16. Drawn as structure only.',
    focus: ['regions', 'dino', 'region_features'], dir: LEFT, margin: 0.9, run: (p) => up.dino(p) },
  { title: 'Pooled into 135', text: 'Averaged over all regions, by area and by class, with roof areas and distances from the junction: 135 numbers for the image.',
    focus: ['region_features', 'imagery'], dir: SIDE, margin: 1, run: (p) => reveal('imagery', p) },
  { title: 'The junction in its network', text: `${nw.graph.neighbours.length} neighbours, the closer the stronger, and ${nw.graph.hop2.length} junctions two steps away, among the ${nw.graph.nodes.length} of the network.`,
    focus: ['graph'], box: 'graph_local', dir: FLAT, margin: 0.95, run: (p) => up.graph(p) },
  { title: 'Written as text', text: `Its position and its neighbours as compact JSON: ${nw.text.length} characters, cut into ${nw.tokens.length} tokens.`,
    focus: ['text'], dir: FRONT, margin: 0.62, run: (p) => up.tokens(p) },
  { title: 'Read by Qwen3', text: `Qwen3-Embedding-0.6B carries every token through ${qwen.layers} layers of ${qwen.hidden_size.toLocaleString()} numbers; every 8th is drawn here.`,
    focus: ['text', 'qwen'], dir: new THREE.Vector3(-0.5, 0.7, 1), margin: 0.82, run: (p) => up.layers(p) },
  { title: 'Averaged over tokens', text: `The second-to-last layer, averaged over all ${nw.tokens.length} tokens: ${qwen.hidden_size.toLocaleString()} numbers for the junction.`,
    focus: ['qwen', 'text_pooled'], box: 'pool', dir: new THREE.Vector3(0.8, 0.85, 0.75), margin: 0.9,
    run: (p) => { up.pool(p); reveal('text_pooled', clamp((p - 0.78) / 0.22, 0.01, 1)); } },
  { title: 'Standardised, then 64 components', text: `Principal components fitted on all ${qwen.junction_years.toLocaleString()} junction-years keep 64, with ${(qwen.pca_explained * 100).toFixed(1)}% of the variance.`,
    focus: ['text_pooled', 'text_standardised', 'network'], dir: LEFT, margin: 0.95,
    run: (p) => { reveal('text_standardised', clamp(p * 2, 0.01, 1)); reveal('network', clamp(p * 2 - 1, 0.01, 1)); } },
  { title: 'Buildings in the census', text: `The City of Melbourne's land-use census (CLUE) records every building each year: ${U.landuse.count} stand in this ${U.landuse.patch_m} m square in ${U.landuse.year}, drawn to scale and coloured by their main use.`,
    focus: ['clue'], dir: new THREE.Vector3(-0.3, 0.75, 1), margin: 0.75, run: (p) => up.buildings(p) },
  { title: 'Summarised into 43', text: 'Building counts, floors, ages, bicycle spaces and distances to tall buildings (17), and the share of 13 uses by buildings and by floors (26).',
    focus: ['clue', 'use_shares', 'landuse'], dir: LEFT, margin: 0.9,
    run: (p) => { up.shares(clamp(p / 0.6, 0, 1)); reveal('landuse', clamp((p - 0.6) / 0.4, 0.01, 1)); } },
  { title: 'One junction-day', text: `${sample.name}, ${day}. Everything the model is given about this place and this day.`,
    focus: ['imagery', 'network', 'landuse', 'time', 'profile'], dir: HIGH, margin: 0.85 },
  { title: 'The calendar', text: '24 features for each of the 96 fifteen-minute intervals of the day.',
    focus: ['time'], dir: LEFT, margin: 1.2 },
  { title: 'Describing the place', text: 'Satellite imagery, road network and land use are each projected to a short vector.',
    focus: ['imagery', 'network', 'landuse', 'proj_imagery', 'proj_network', 'proj_landuse'], dir: LEFT, margin: 0.9,
    run: (p) => ['proj_imagery', 'proj_network', 'proj_landuse'].forEach((id) => reveal(id, p)) },
  { title: 'Static fusion', text: 'Joined and mapped to 192 values.',
    focus: ['proj_imagery', 'proj_network', 'proj_landuse', 'static'], dir: LEFT, margin: 0.95,
    run: (p) => reveal('static', p) },
  { title: 'Laid across the day', text: 'The same description is repeated for all 96 intervals.',
    focus: ['static', 'repeat'], dir: SIDE, margin: 0.9, run: (p) => reveal('repeat', p) },
  { title: 'Joined with the calendar', text: 'Each interval now carries the place and its time.',
    focus: ['repeat', 'proj_time', 'time', 'fusion'], dir: LEFT, margin: 0.85, run: (p) => reveal('fusion', p) },
  { title: 'Temporal block 1 · dilation 1', text: 'Each output interval reads its neighbours one interval apart.',
    focus: ['fusion', 'tcn1.out'], dir: SIDE, margin: 1.15, run: (p) => sweep('fusion', 'tcn1.out', 1, p) },
  { title: 'Temporal block 2 · dilation 2', text: 'Now two intervals apart, so each output sees further along the day.',
    focus: ['tcn1.out', 'tcn2.out'], dir: SIDE, margin: 1.15, run: (p) => sweep('tcn1.out', 'tcn2.out', 2, p) },
  { title: 'Temporal block 3 · dilation 4', text: 'Four apart. Together the three blocks let each interval draw on 14 intervals either side, three and a half hours.',
    focus: ['tcn2.out', 'tcn3.out'], dir: SIDE, margin: 1.15, run: (p) => sweep('tcn2.out', 'tcn3.out', 4, p) },
  { title: 'One number per interval', text: 'The head turns 192 channels into how far each interval departs from the 2022 profile.',
    focus: ['tcn3.out', 'head.hidden', 'log_ratio'], dir: LEFT, margin: 0.9,
    run: (p) => { reveal('head.hidden', Math.min(1, p * 1.6)); reveal('log_ratio', clamp(p * 1.6 - 0.6, 0.01, 1)); } },
  { title: 'Back to vehicles', text: `Scaled by the junction's 2022 profile and blended with it (α = ${manifest.alpha.toFixed(2)}): vehicles per 15 minutes.`,
    focus: ['log_ratio', 'profile', 'volume'], dir: LEFT, margin: 1.05,
    enter: () => { $('chart').hidden = false; drawChart(); },
    run: (p) => { O('volume').mesh.scale.y = Math.max(0.01, 1 - Math.pow(1 - p, 3)); } },
];

$('chart-legend').innerHTML = [['predicted', THEME.predicted], ['2022 profile', THEME.profile], ['recorded', THEME.recorded]]
  .map(([name, colour]) => `<span><i style="background:${colour}"></i>${name}</span>`).join('');
function drawChart() {
  const trained = manifest.weights === 'trained';
  paintDayChart($('chart-canvas'), [
    { data: tensor(sample.tensors.profile).data, colour: THEME.profile, dash: [4, 3] },
    { data: tensor(sample.tensors.recorded).data, colour: THEME.recorded, width: 1.3 },
    { data: tensor(sample.tensors.volume).data, colour: THEME.predicted, width: trained ? 2 : 1.2, dash: trained ? [] : [2, 2] },
  ], { ink: THEME.ink, grid: THEME.grid });
}

function resetAll() {
  model.objects.forEach((o) => { if (o.full !== undefined) { o.mesh.count = o.full; o.mesh.scale.y = 1; } });
  up.reset();
  kernel.visible = false;
  $('chart').hidden = true;
}

// ---------------------------------------------------------------- the two modes

const HOLD = 1300, DURATION = 7200;
let fly = false;                      // false: free exploration under the wave of light
let waving = true, playing = true;    // the wave in exploration, the playback in the fly-through
let index = -1, started = 0, pausedAt = 0;
STEPS.forEach((_, i) => {
  const b = document.createElement('button');
  b.type = 'button';
  b.setAttribute('aria-label', `Step ${i + 1}`);
  b.addEventListener('click', () => go(i));
  $('dots').appendChild(b);
});

function showPlay() {
  const on = fly ? playing : waving;
  $('play').textContent = on ? 'Pause' : 'Play';
  $('play').classList.toggle('on', on);
}

function go(i) {
  index = (i + STEPS.length) % STEPS.length;
  const step = STEPS[index];
  resetAll();
  model.focus(step.focus);
  step.run?.(0);
  step.enter?.();
  const box = step.box ? model.boxes[step.box] : model.boxOf(step.focus);
  flyTo(stage.camera, stage.controls, frame(stage.camera, stage.controls, box, step.dir, step.margin), HOLD);
  started = performance.now();
  pausedAt = started;
  $('count').textContent = `${index + 1} / ${STEPS.length}`;
  $('step-title').textContent = step.title;
  $('step-text').textContent = step.text;
  document.querySelectorAll('#dots button').forEach((b, k) => b.classList.toggle('on', k === index));
}

function enterFly(step) {
  fly = true;
  playing = true;
  model.objects.forEach((o) => o.glow(0));
  $('steps').hidden = false;
  $('hint').hidden = $('reset').hidden = true;
  $('fly').textContent = 'Exit fly-through';
  $('fly').classList.add('on');
  showPlay();
  go(step);
}

function exitFly() {
  fly = false;
  resetAll();
  model.focus([]);
  $('steps').hidden = true;
  $('hint').hidden = $('reset').hidden = false;
  $('fly').textContent = 'Fly-through';
  $('fly').classList.remove('on');
  showPlay();
  flyTo(stage.camera, stage.controls, home, 1200);
}

$('fly').addEventListener('click', () => (fly ? exitFly() : enterFly(0)));
$('prev').addEventListener('click', () => go(index - 1));
$('next').addEventListener('click', () => go(index + 1));
$('play').addEventListener('click', () => {
  if (fly) {
    playing = !playing;
    if (playing) started += performance.now() - pausedAt; else pausedAt = performance.now();
  } else {
    waving = !waving;
  }
  showPlay();
});
$('reset').addEventListener('click', () => flyTo(stage.camera, stage.controls, home, 1200));
window.addEventListener('keydown', (e) => {
  if (!fly) return;
  if (e.key === 'ArrowRight') go(index + 1);
  if (e.key === 'ArrowLeft') go(index - 1);
  if (e.key === ' ') { e.preventDefault(); $('play').click(); }
  if (e.key === 'Escape') exitFly();
});
window.addEventListener('resize', () => { if (!$('chart').hidden) drawChart(); });

// ---------------------------------------------------------------- hover and click

const ray = new THREE.Raycaster(), pointer = new THREE.Vector2();
const meshes = [...model.objects.values()].map((o) => o.mesh);
function pick(event) {
  const rect = stage.renderer.domElement.getBoundingClientRect();
  pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
  ray.setFromCamera(pointer, stage.camera);
  const hit = ray.intersectObjects(meshes, true).find((h) => h.object.visible);
  return hit ? model.objects.get(hit.object.userData.id) : null;
}
stage.renderer.domElement.addEventListener('pointermove', (event) => {
  const o = pick(event);
  const tip = $('tip');
  if (!o) { tip.hidden = true; return; }
  const rect = $('view').getBoundingClientRect();
  tip.hidden = false;
  tip.innerHTML = `${o.def.label}<small>${o.detail ?? o.shape} · ${kinds[o.def.kind]}</small>`;
  tip.style.left = `${event.clientX - rect.left + 14}px`;
  tip.style.top = `${event.clientY - rect.top + 14}px`;
});
// in the fly-through the camera follows the steps, so a click does not take it elsewhere
stage.renderer.domElement.addEventListener('click', (event) => {
  const o = fly ? null : pick(event);
  if (!o) return;
  flyTo(stage.camera, stage.controls, frame(stage.camera, stage.controls, o.bounds, new THREE.Vector3(-0.8, 0.45, 1), 1.6), 1100);
});

// ---------------------------------------------------------------- one loop for both

if (params.has('step')) {
  enterFly(clamp(Number(params.get('step')), 1, STEPS.length) - 1);
  if (params.has('at')) {
    $('play').click();
    pausedAt = started + HOLD + clamp(Number(params.get('at')), 0, 1) * (DURATION - HOLD - 900);
  }
}
const clockStart = performance.now();
let last = clockStart, waveClock = 0;
const span = [model.bounds.min.x, model.bounds.max.x];
(function loop(now) {
  const dt = Math.max(0, (now - last) / 1000);
  last = now;
  if (fly) {
    const elapsed = (playing ? now : pausedAt) - started;
    STEPS[index].run?.(clamp((elapsed - HOLD) / (DURATION - HOLD - 900), 0, 1));
    $('track').style.width = `${clamp(elapsed / DURATION, 0, 1) * 100}%`;
    if (playing && elapsed > DURATION) go(index + 1);
  } else {
    if (waving) waveClock += dt;
    // a band of light travels from the sources to the output every few seconds
    const front = span[0] + ((waveClock * 80) % (span[1] - span[0] + 80)) - 40;
    model.objects.forEach((o) => o.glow(waving ? Math.max(0, 1 - Math.abs(o.def.x - front) / 26) * 0.55 : 0));
  }
  model.update(fly ? (now - clockStart) / 1000 : waveClock);
  stage.controls.update();
  stage.renderer.render(stage.scene, stage.camera);
  requestAnimationFrame(loop);
})(performance.now());
