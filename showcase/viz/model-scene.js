// The network laid out in 3D, shared by the layer view and the fly-through.
//
// x runs from input to output and z separates the branches: the static inputs on one side, the
// time branch on the other, meeting at the fusion layer. A vector of channels stands as a column,
// a day of 96 intervals lies along z, and a channel-by-interval tensor stands as a wall with
// channels down y and intervals along z. Everything stands on the floor at y = 0. Given a sample's
// upstream trace, the imagery and road-network pipelines are laid out to the left (upstream-scene.js).

import { THREE, voxels, KIND_COLOUR, turbo, Flow, Labels, shapeText } from './common.js';
import { buildUpstream } from './upstream-scene.js';

export const CELL = 0.5;

export const LAYOUT = [
  { id: 'imagery', x: 0, z: -110, layout: 'column', kind: 'input', label: 'Imagery features' },
  { id: 'network', x: 0, z: -92, layout: 'column', kind: 'input', label: 'Road network' },
  { id: 'landuse', x: 0, z: -74, layout: 'column', kind: 'input', label: 'Land use' },
  { id: 'time', x: 0, z: 46, layout: 'plane', kind: 'input', label: 'Time features' },
  { id: 'proj_imagery', x: 30, z: -110, layout: 'column', kind: 'linear', label: 'Imagery projection' },
  { id: 'proj_network', x: 30, z: -92, layout: 'column', kind: 'linear', label: 'Network projection' },
  { id: 'proj_landuse', x: 30, z: -74, layout: 'column', kind: 'linear', label: 'Land-use projection' },
  { id: 'proj_time', x: 30, z: 46, layout: 'plane', kind: 'linear', label: 'Time projection' },
  { id: 'static', x: 62, z: -83, layout: 'column', kind: 'linear', label: 'Static fusion' },
  { id: 'repeat', x: 96, z: -40, layout: 'plane', kind: 'reshape', label: 'Repeat over the day', order: 'columns' },
  { id: 'fusion', x: 132, z: 0, layout: 'plane', kind: 'linear', label: 'Fusion', order: 'columns' },
  { id: 'tcn1.out', x: 166, z: 0, layout: 'plane', kind: 'conv', label: 'Temporal block 1', order: 'columns' },
  { id: 'tcn2.out', x: 200, z: 0, layout: 'plane', kind: 'conv', label: 'Temporal block 2', order: 'columns' },
  { id: 'tcn3.out', x: 234, z: 0, layout: 'plane', kind: 'conv', label: 'Temporal block 3', order: 'columns' },
  { id: 'head.hidden', x: 266, z: 0, layout: 'plane', kind: 'conv', label: 'Head', order: 'columns' },
  { id: 'log_ratio', x: 292, z: 0, layout: 'row', kind: 'conv', label: 'Log ratio' },
  { id: 'profile', x: 292, z: 46, layout: 'row', kind: 'input', label: '2022 profile' },
  { id: 'volume', x: 318, z: 0, layout: 'bars', kind: 'output', label: 'Vehicles per 15 min' },
];

export const EDGES = [
  ['imagery', 'proj_imagery'], ['network', 'proj_network'], ['landuse', 'proj_landuse'],
  ['time', 'proj_time'], ['proj_imagery', 'static'], ['proj_network', 'static'],
  ['proj_landuse', 'static'], ['static', 'repeat'], ['repeat', 'fusion'],
  ['proj_time', 'fusion'], ['fusion', 'tcn1.out'], ['tcn1.out', 'tcn2.out'], ['tcn2.out', 'tcn3.out'],
  ['tcn3.out', 'head.hidden'], ['head.hidden', 'log_ratio'], ['log_ratio', 'volume'], ['profile', 'volume'],
];

function hexRGB(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

// the kind's hue, lighter or darker with the value: a texture of the data in the layer's colour
function kindColour(kind) {
  const [r, g, b] = hexRGB(KIND_COLOUR[kind]);
  return (v, n) => {
    const k = 0.5 + 0.75 * (Number.isFinite(n) ? n : 0.5);
    return [Math.min(1, r * k), Math.min(1, g * k), Math.min(1, b * k)];
  };
}

export function buildModelScene(stage, { sample, tensor, colouring = 'kind', dark = false, labels = true, flows = true, upstream = null }) {
  const objects = new Map();
  const labelLayer = labels ? new Labels(stage.renderer.domElement.parentElement, stage.camera) : null;
  for (const def of LAYOUT) {
    const t = tensor(sample.tensors[def.id]);
    const material = new THREE.MeshStandardMaterial({
      roughness: 0.5, metalness: 0.05, transparent: true, opacity: 1,
      emissive: new THREE.Color(KIND_COLOUR[def.kind]), emissiveIntensity: 0,
    });
    const colour = colouring === 'kind' ? kindColour(def.kind) : undefined;
    const mesh = voxels(t, { layout: def.layout, cell: CELL, fill: 0.84, colour, height: 40, material, order: def.order,
                             thick: def.layout === 'plane' ? 1 : 3 });
    const box = new THREE.Box3().setFromObject(mesh);
    const lift = -box.min.y;                                   // stand on the floor
    mesh.position.set(def.x, lift, def.z);
    stage.scene.add(mesh);
    const bounds = new THREE.Box3().setFromObject(mesh);
    mesh.userData = { ...mesh.userData, id: def.id };
    const object = { def, mesh, material, materials: [material], bounds, tensor: t, full: mesh.count, shape: shapeText(t.shape),
                     glow: (v) => { material.emissiveIntensity = v; } };
    objects.set(def.id, object);
    if (labelLayer) {
      const top = new THREE.Vector3(def.x, bounds.max.y + 3, def.z);
      object.label = labelLayer.add(top, `${def.label}<small>${shapeText(t.shape)}</small>`);
    }
  }
  const up = upstream ? buildUpstream(stage, { ...upstream, colouring, dark, labelLayer, kindColour }) : null;
  up?.objects.forEach((o, id) => objects.set(id, o));
  const flowList = [];
  if (flows) {
    // upstream stages are wider than a column, so they name where their data leaves and arrives
    const outX = (o) => o.port?.outX ?? o.def.x + 1.2;
    const inX = (o) => o.port?.inX ?? o.def.x - 1.2;
    const portY = (o, end) => o.port?.[end] ?? o.port?.y ?? Math.min(o.bounds.max.y, 24) * 0.5 + 2;
    for (const [a, b, note] of [...EDGES, ...(up?.edges ?? [])]) {
      const A = objects.get(a), B = objects.get(b);
      const start = new THREE.Vector3(outX(A), portY(A, 'outY'), A.def.z);
      const end = new THREE.Vector3(inX(B), portY(B, 'inY'), B.def.z);
      const mid = start.clone().lerp(end, 0.5);
      mid.y += 6;
      const colour = new THREE.Color(KIND_COLOUR[B.def.kind]);
      const flow = new Flow(stage.scene, [start, mid, end], { colour, count: 5, speed: 0.22, size: dark ? 0.7 : 0.65 });
      // a note on the flow names the step it stands for, placed on the way in, clear of the source's label
      const label = note && labelLayer ? labelLayer.add(flow.curve.getPointAt(0.72).add(new THREE.Vector3(0, 2, 0)), note) : null;
      label?.classList.add('flow');
      flowList.push({ from: a, to: b, label, flow });
    }
  }
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(1500, 700),
    new THREE.MeshStandardMaterial({ color: dark ? 0x0d1119 : 0xeef2f7, roughness: 1 }));
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(0, -0.02, -70);
  stage.scene.add(floor);
  const all = new THREE.Box3();
  objects.forEach((o) => all.union(o.bounds));
  return {
    objects, flows: flowList, labels: labelLayer, bounds: all, upstream: up, boxes: up?.boxes ?? {},
    update(time) { flowList.forEach(({ flow }) => flow.update(time)); labelLayer?.update(); },
    focus(ids) {
      const keep = new Set(ids);
      objects.forEach((o, id) => {
        const on = !ids.length || keep.has(id);
        // translucent parts keep their own opacity and never hide what lies behind them
        o.materials.forEach((m) => {
          const base = m.userData.base ?? 1;
          m.opacity = (on ? 1 : m.userData.dim ?? 0.1) * base;       // stacked sheets dim further, or they add up
          m.depthWrite = on && base === 1;
        });
        if (o.label) { o.label.style.opacity = on ? '1' : '0.25'; o.label.dataset.dim = on ? '' : '1'; }
      });
      flowList.forEach(({ from, to, flow, label }) => {
        const on = !ids.length || (keep.has(from) && keep.has(to));
        flow.opacity = on ? 1 : 0.12;
        if (label) { label.style.opacity = on ? '1' : '0.25'; label.dataset.dim = on ? '' : '1'; }
      });
      if (labelLayer) labelLayer.occluders = ids.map((id) => objects.get(id).bounds);
    },
    boxOf(ids) {
      const b = new THREE.Box3();
      ids.forEach((id) => b.union(objects.get(id).bounds));
      return b;
    },
  };
}
