// Upstream of the network's inputs, laid out to its left (x < 0) in three lanes that end in the
// input columns they feed.
//
// Behind, the imagery: the satellite image lies on the floor with SAM 3's segmentation lifted above
// it, the regions cut from it stand in a grid, then DINOv3 and the region features -- drawn as
// outlines, because their values are not on this machine -- pooled into the 135 imagery features.
// In front, the road network: the junction among the 143 of the network with its text standing
// behind it as tokens, Qwen3's layers stacked as slices of token-by-channel values, and the
// 1,024-number mean over the tokens before and after standardising, reduced to the 64 network
// features. Nearest the viewer, the land use: the census buildings standing in the same 500 m
// square as the image, summarised as the shares of each use that go into the 43 attributes.

import { THREE, voxels, KIND_COLOUR, turbo, clamp } from './common.js';

export const upstreamKey = (sample) => `${sample.junction}_${sample.date.slice(0, 4)}`;

const IMAGERY_Z = -200, NETWORK_Z = -92, LANDUSE_Z = -15;
const X = { aerial: -225, regions: -150, dino: -100, features: -72,
            graph: -222, qwen: -145, pooled: -92, standardised: -68, clue: -150, shares: -95 };
const IMAGE = 60, LIFT = 16;                 // the 500 m image, 8.3 m to a unit
const TILE = 6.2, GAP = 0.8, COLS = 8;       // region tiles
const MAP = 56;                              // the road network's longer side
const TOKEN = 0.2, CHANNEL = 0.3, LAYER = 1.5;
const PANEL = 52;                            // the text panel's width
const CLASS_NAMES = { road: 'road', building_roof: 'roof', vegetation: 'vegetation', parking_paved: 'parking' };
// the census's 13 uses; office, apartments and retail keep the colours of the report's Chapter 5
const USES = {
  house_townhouse: ['house', '#e8a87c'], residential_apartment: ['apartment', '#c0662b'], office: ['office', '#1c5cab'],
  retail__shop: ['retail', '#d9a93b'], entertainment_recreation__indoor: ['entertainment', '#b35c9e'],
  educational_research: ['education', '#2a9d8f'], storage: ['storage', '#7d6b58'], commercial_accommodation: ['hotel', '#7a5cb8'],
  hospital_clinic: ['hospital', '#d0455a'], student_accommodation: ['student housing', '#9c4a2a'],
  workshop_studio: ['workshop', '#6b8e3a'], unoccupied__unused: ['unoccupied', '#dcd7cf'], other: ['other', '#a9a49b'],
};

const ease = (k) => (k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2);
const phase = (p, a, b) => clamp((p - a) / (b - a), 0, 1);
const loader = new THREE.TextureLoader();

function picture(url, mipmaps = true) {
  const t = loader.load(url);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  if (!mipmaps) { t.generateMipmaps = false; t.minFilter = THREE.LinearFilter; }
  return t;
}

// every material remembers its resting opacity, so that focus can dim it and bring it back
function material(Kind, options, base = 1) {
  const m = new Kind({ transparent: true, ...options });
  m.opacity = base;
  m.userData.base = base;
  return m;
}

// a rectangle lying in the x-z plane
function rectangle(width, depth, m) {
  const w = width / 2, d = depth / 2;
  const line = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-w, 0, -d), new THREE.Vector3(w, 0, -d), new THREE.Vector3(w, 0, d), new THREE.Vector3(-w, 0, d)]), m);
  line.computeLineDistances();
  return line;
}

export function buildUpstream(stage, { entry, tensor, block, colouring, dark, labelLayer, kindColour }) {
  const objects = new Map();
  const colourOf = (kind) => new THREE.Color(KIND_COLOUR[kind]);
  const im = entry.imagery, net = entry.network;
  const sideColour = dark ? 0x1b2331 : 0xdde4ee;

  function add(def, object, materials, shape, { port, labelAt, detail } = {}) {
    object.position.set(def.x, def.y ?? 0, def.z);
    object.traverse((n) => { n.userData.id = def.id; });
    stage.scene.add(object);
    object.updateMatrixWorld(true);
    const bounds = new THREE.Box3().setFromObject(object);
    const lines = materials.filter((m) => m.isLineBasicMaterial || m.isLineDashedMaterial);
    const lit = materials.filter((m) => m.emissive);
    const o = {
      def, mesh: object, materials, bounds, shape, detail: detail ?? shape, port: port?.(bounds),
      glow(v) {
        lit.forEach((m) => { m.emissiveIntensity = v; });
        lines.forEach((m) => { m.opacity = Math.min(1, m.userData.base + v); });
      },
    };
    objects.set(def.id, o);
    if (labelLayer) {
      const c = bounds.getCenter(new THREE.Vector3());
      const at = labelAt ? labelAt(bounds, c) : new THREE.Vector3(c.x, bounds.max.y + 3, c.z);
      o.label = labelLayer.add(at, `${def.label}<small>${shape}</small>`);
    }
    return o;
  }
  const frontEdge = (b, c) => new THREE.Vector3(c.x, b.max.y + 1, b.max.z + 1);

  // ---------------------------------------------------------------- imagery

  const aerialTop = material(THREE.MeshBasicMaterial, { map: picture(im.image) });
  const aerialSide = material(THREE.MeshStandardMaterial, { color: sideColour, roughness: 0.9 });
  {
    const slab = new THREE.Mesh(new THREE.BoxGeometry(IMAGE, 0.6, IMAGE),
      [aerialSide, aerialSide, aerialTop, aerialSide, aerialSide, aerialSide]);
    slab.position.y = 0.3;
    add({ id: 'aerial', x: X.aerial, z: IMAGERY_Z, kind: 'source', label: 'Satellite image' }, new THREE.Group().add(slab),
      [aerialTop, aerialSide], `${im.image_px}×${im.image_px}`,
      { labelAt: frontEdge, detail: `${im.image_px}×${im.image_px} px · ${im.patch_m} m across · ${entry.year}` });
  }

  const lifted = new THREE.Group();                     // the segmentation, moved as one
  {
    const overlay = material(THREE.MeshBasicMaterial, { map: picture(im.segmentation), side: THREE.DoubleSide, alphaTest: 0.02 });
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(IMAGE, IMAGE), overlay);
    plane.rotation.x = -Math.PI / 2;
    const edge = material(THREE.LineBasicMaterial, { color: colourOf('pretrained') }, 0.9);
    lifted.add(plane, rectangle(IMAGE, IMAGE, edge));
    lifted.position.y = LIFT;
    const present = Object.entries(im.class_shares).filter(([, share]) => share > 0);
    const dot = (name) => `<i class="dot" style="background:${im.class_colours[name]}"></i>${CLASS_NAMES[name]}`;
    add({ id: 'segmentation', x: X.aerial, z: IMAGERY_Z, kind: 'pretrained', label: 'SAM 3 segmentation' },
      new THREE.Group().add(lifted), [overlay, edge], present.map(([name]) => dot(name)).join(''),
      { port: (b) => ({ outX: b.max.x + 0.6, y: LIFT }),
        detail: present.map(([name, share]) => `${dot(name)} ${(share * 100).toFixed(1)}%`).join('') });
  }

  // the regions stand in a grid; each remembers where it lies on the image, to be cut from it
  const tiles = [];
  {
    const face = material(THREE.MeshBasicMaterial, { map: picture(im.regions_atlas, false), side: THREE.DoubleSide });
    const group = new THREE.Group();
    const step = TILE + GAP, rows = Math.ceil(im.regions.length / COLS), px = im.image_px, unit = TILE / 122;   // 122 px: a tile's longer side
    im.regions.forEach((r, i) => {
      const w = r.size[0] * unit, h = r.size[1] * unit;
      const geometry = new THREE.PlaneGeometry(w, h);
      const uv = geometry.attributes.uv;
      for (let k = 0; k < uv.count; k++)
        uv.setXY(k, r.uv[0] + uv.getX(k) * (r.uv[2] - r.uv[0]), r.uv[1] + uv.getY(k) * (r.uv[3] - r.uv[1]));
      const mesh = new THREE.Mesh(geometry, face);
      const grid = new THREE.Vector3((i % COLS - (COLS - 1) / 2) * step, 2 + (rows - 1 - Math.floor(i / COLS) + 0.5) * step, 0);
      const [x0, y0, x1, y1] = r.box;
      const onImage = new THREE.Vector3(X.aerial - X.regions + ((x0 + x1) / 2 / px - 0.5) * IMAGE, LIFT + 0.1 + i * 0.03,
                                        ((y0 + y1) / 2 / px - 0.5) * IMAGE);           // smaller ones on top
      mesh.position.copy(grid);
      group.add(mesh);
      tiles.push({ mesh, grid, onImage, small: ((x1 - x0) / px) * IMAGE / w });
    });
    add({ id: 'regions', x: X.regions, z: IMAGERY_Z, kind: 'source', label: 'Regions' }, group, [face],
      `${im.region_count}`, { port: (b) => ({ inX: b.min.x - 0.6, outX: b.max.x + 0.6, y: (b.min.y + b.max.y) / 2 }),
                               detail: `${im.region_count} regions · the ${tiles.length} largest shown, each in its box` });
  }

  const dinoFills = [];
  const dinoColour = colourOf('pretrained'), dinoLit = dinoColour.clone().lerp(new THREE.Color(0xffffff), 0.55);
  {
    const group = new THREE.Group(), materials = [];
    const box = new THREE.BoxGeometry(14, 0.5, 14), edges = new THREE.EdgesGeometry(box);
    for (let k = 0; k < 12; k++) {
      const fill = material(THREE.MeshBasicMaterial, { color: dinoColour, depthWrite: false }, 0.16);
      const line = material(THREE.LineBasicMaterial, { color: dinoColour }, 0.75);
      const slab = new THREE.Mesh(box, fill), outline = new THREE.LineSegments(edges, line);
      slab.position.y = outline.position.y = 1 + k * 1.6;
      group.add(slab, outline);
      materials.push(fill, line);
      dinoFills.push(fill);
    }
    add({ id: 'dino', x: X.dino, z: IMAGERY_Z, kind: 'pretrained', label: 'DINOv3' }, group, materials, '12 layers',
      { port: (b) => ({ inX: b.min.x - 0.6, outX: b.max.x + 0.6, y: 9 }),
        detail: 'ViT-B/16 · 12 layers · 768 numbers per region · structure only' });
  }

  const lattice = new THREE.Group();                    // regions along z, 16 components up y
  {
    const n = im.region_count, cz = Math.min(0.6, 50 / n), depth = n * cz, height = 16 * 0.6;
    const points = [];
    for (let k = 0; k <= n; k++) points.push(0, 0, k * cz, 0, height, k * cz);
    for (let r = 0; r <= 16; r++) points.push(0, r * 0.6, 0, 0, r * 0.6, depth);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    const line = material(THREE.LineBasicMaterial, { color: colourOf('reduce') }, 0.7);
    const fill = material(THREE.MeshBasicMaterial, { color: colourOf('reduce'), depthWrite: false, side: THREE.DoubleSide }, 0.12);
    const sheet = new THREE.Mesh(new THREE.PlaneGeometry(depth, height), fill);
    sheet.rotation.y = Math.PI / 2;
    sheet.position.set(0, height / 2, depth / 2);
    lattice.add(new THREE.LineSegments(geometry, line), sheet);
    lattice.position.z = -depth / 2;
    add({ id: 'region_features', x: X.features, z: IMAGERY_Z, kind: 'reduce', label: 'Region features' },
      new THREE.Group().add(lattice), [line, fill], `${n}×16`,
      { port: (b) => ({ inX: b.min.x - 1, outX: b.max.x + 1, y: 4 }), labelAt: frontEdge,
        detail: `${n}×768 → PCA → ${n}×16 · structure only` });
  }

  // ---------------------------------------------------------------- road network

  const G = net.graph;
  const palette = dark ? { plate: 0x121925, edge: 0x2f3a4e, node: 0x56627a, near: 0x58a6ff, far: 0x9ecbff }
                       : { plate: 0xffffff, edge: 0xc3cedd, node: 0x9aa6b8, near: 0x1c5cab, far: 0x6ea8fe };
  const base = new THREE.Color(palette.node), near = new THREE.Color(palette.near), far = new THREE.Color(palette.far);
  const index = new Map(G.nodes.map(([id], i) => [id, i]));
  const links = [], hop2 = G.hop2.map((id) => index.get(id)).filter((i) => i !== undefined);
  let nodes, localBox, graphDepth;
  {
    const xs = G.nodes.map((n) => n[1]), ys = G.nodes.map((n) => n[2]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const s = MAP / Math.max(maxX - minX, maxY - minY), mx = (minX + maxX) / 2, my = (minY + maxY) / 2;
    const at = new Map(G.nodes.map(([id, x, y]) => [id, new THREE.Vector3((x - mx) * s, 0, (my - y) * s)]));   // north is -z
    const group = new THREE.Group();
    const plateMat = material(THREE.MeshStandardMaterial, { color: palette.plate, roughness: 0.95 });
    graphDepth = (maxY - minY) * s + 5;
    const plate = new THREE.Mesh(new THREE.BoxGeometry((maxX - minX) * s + 5, 0.4, graphDepth), plateMat);
    plate.position.y = 0.2;
    const points = [];
    G.edges.forEach(([a, b]) => { const A = at.get(a), B = at.get(b); points.push(A.x, 0.45, A.z, B.x, 0.45, B.z); });
    const edgeGeometry = new THREE.BufferGeometry();
    edgeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    const edgeMat = material(THREE.LineBasicMaterial, { color: palette.edge }, 0.9);
    const nodeMat = material(THREE.MeshStandardMaterial, { roughness: 0.6 });
    nodes = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.42, 0.42, 0.5, 12), nodeMat, G.nodes.length);
    const m4 = new THREE.Matrix4();
    G.nodes.forEach(([id], i) => { const p = at.get(id); nodes.setMatrixAt(i, m4.makeTranslation(p.x, 0.65, p.z)); nodes.setColorAt(i, base); });
    const self = at.get(G.self);
    const pinMat = material(THREE.MeshStandardMaterial, { color: colourOf('source'), roughness: 0.4, emissive: colourOf('source'), emissiveIntensity: 0 });
    const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.9, 0.9, 2.4, 20), pinMat);
    pin.position.set(self.x, 1.6, self.z);
    const linkMat = material(THREE.MeshStandardMaterial, { color: palette.near, roughness: 0.5 });
    G.neighbours.forEach(([id, w]) => {
      const d = at.get(id).clone().sub(self);
      const geometry = new THREE.BoxGeometry(1, 0.3, 0.2 + (0.6 * w) / 0.008);   // wider for closer neighbours
      geometry.translate(0.5, 0, 0);
      const mesh = new THREE.Mesh(geometry, linkMat);
      mesh.position.set(self.x, 0.9, self.z);
      mesh.rotation.y = Math.atan2(-d.z, d.x);
      mesh.scale.x = d.length();
      group.add(mesh);
      links.push({ mesh, length: d.length(), node: index.get(id) });
    });
    group.add(plate, new THREE.LineSegments(edgeGeometry, edgeMat), nodes, pin);
    add({ id: 'graph', x: X.graph, z: NETWORK_Z, kind: 'source', label: 'Junction network' }, group,
      [plateMat, edgeMat, nodeMat, pinMat, linkMat], `${G.nodes.length} junctions`,
      { labelAt: frontEdge,
        detail: `${G.nodes.length} junctions · ${G.edges.length} links · ${links.length} neighbours · ${hop2.length} two steps away` });
    // the junction's neighbourhood, at least 30 units across, for framing
    localBox = new THREE.Box3();
    [G.self, ...G.neighbours.map((n) => n[0]), ...G.hop2].forEach((id) => { if (at.has(id)) localBox.expandByPoint(at.get(id)); });
    localBox.expandByPoint(self.clone().addScalar(15)).expandByPoint(self.clone().subScalar(15));
    localBox.min.y = 0; localBox.max.y = 4;
    localBox.translate(group.position);
  }
  const paintGraph = (neighbours, twoHop) => {
    G.nodes.forEach((_, i) => nodes.setColorAt(i, base));
    links.forEach((l, k) => { if (k < neighbours) nodes.setColorAt(l.node, near); });
    hop2.forEach((i, k) => { if (k < twoHop) nodes.setColorAt(i, far); });
    nodes.instanceColor.needsUpdate = true;
  };
  paintGraph(links.length, hop2.length);

  // the text, drawn token by token in alternating tints as a tokenizer shows it, standing behind the
  // map it was written from
  const tokens = net.tokens;
  let drawTokens;
  {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const width = 1024, pad = 22, line = 40, font = '500 25px "Cascadia Mono", Consolas, "Courier New", monospace';
    ctx.font = font;
    const place = [];
    let x = pad, y = pad;
    tokens.forEach((t) => {
      const w = ctx.measureText(t).width;
      if (x + w > width - pad) { x = pad; y += line; }
      place.push([x, y, w]);
      x += w;
    });
    canvas.width = width;
    canvas.height = Math.ceil(y + line + pad);
    const chips = dark ? ['#1f3a5f', '#4a2231', '#1a4a31', '#4a3a12', '#362a5c'] : ['#dbeafe', '#fde2e4', '#d9f5e4', '#fdf0c4', '#ebe4ff'];
    const ink = dark ? '#e6e9f0' : '#14213d', faint = dark ? '#4b5468' : '#b3bccb', paper = dark ? '#111621' : '#ffffff';
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = 8;
    let drawn = -1;
    drawTokens = (k) => {
      if (k === drawn) return;
      drawn = k;
      ctx.fillStyle = paper;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.font = font;
      ctx.textBaseline = 'middle';
      tokens.forEach((t, i) => {
        const [tx, ty, w] = place[i];
        if (i < k) { ctx.fillStyle = chips[i % chips.length]; ctx.fillRect(tx, ty + 3, w, line - 6); }
        ctx.fillStyle = i < k ? ink : faint;
        ctx.fillText(t, tx, ty + line / 2);
      });
      texture.needsUpdate = true;
    };
    drawTokens(tokens.length);
    const h = (PANEL * canvas.height) / width;
    const face = material(THREE.MeshBasicMaterial, { map: texture });
    const back = material(THREE.MeshStandardMaterial, { color: sideColour, roughness: 0.9 });
    const panel = new THREE.Mesh(new THREE.BoxGeometry(PANEL, h, 0.4), [back, back, back, back, face, back]);
    panel.position.y = 2 + h / 2;
    add({ id: 'text', x: X.graph, z: NETWORK_Z - graphDepth / 2 - 1.5, kind: 'source', label: 'Network text' }, new THREE.Group().add(panel),
      [face, back], `${tokens.length} tokens`,
      { port: (b) => ({ inX: b.min.x - 0.6, outX: b.max.x + 0.6, y: 2 + h / 2 }),
        detail: `${net.text.length} characters · ${tokens.length} tokens` });
  }

  // Qwen3's hidden states: one slice per layer, tokens along x, every 8th channel along z. The
  // top slice, the last layer, is only outlined: the features are taken one layer below it.
  const slices = [];
  const ghost = new THREE.Mesh();
  let usedY = 0, towerTop, aboveTower;
  {
    const t = block(net.tower);
    const [L, T, C] = t.shape;
    const W = T * TOKEN, D = C * CHANNEL, used = L - 2;             // hidden_states[-2]
    const tint = colouring === 'kind' ? kindColour('pretrained') : null;
    const group = new THREE.Group(), materials = [];
    const plane = new THREE.PlaneGeometry(W, D);
    for (let l = 0; l < L; l++) {
      const y = 0.8 + l * LAYER;
      let sheet = null;
      if (l < L - 1) {
        const rgba = new Uint8Array(T * C * 4);
        for (let c = 0; c < C; c++)
          for (let k = 0; k < T; k++) {
            const v = t.data[(l * T + k) * C + c] / 255;
            const [r, g, b] = tint ? tint(v, v) : turbo(v);
            const i = (c * T + k) * 4;
            rgba[i] = r * 255; rgba[i + 1] = g * 255; rgba[i + 2] = b * 255; rgba[i + 3] = 255;
          }
        const texture = new THREE.DataTexture(rgba, T, C);
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.magFilter = THREE.NearestFilter;
        texture.needsUpdate = true;
        const face = material(THREE.MeshBasicMaterial, { map: texture, side: THREE.DoubleSide });
        face.userData.dim = 0.005;
        sheet = new THREE.Mesh(plane, face);
        sheet.rotation.x = -Math.PI / 2;
        sheet.position.y = y;
        group.add(sheet);
        materials.push(face);
        if (l === used) {
          usedY = y;
          ghost.geometry = plane;
          ghost.material = material(THREE.MeshBasicMaterial, { map: texture, side: THREE.DoubleSide }, 0.95);
          ghost.rotation.x = -Math.PI / 2;
        }
      }
      const edgeMat = l === L - 1
        ? material(THREE.LineDashedMaterial, { color: colourOf('pretrained'), dashSize: 1.2, gapSize: 0.8 }, 0.9)
        : material(THREE.LineBasicMaterial, { color: l === used ? (dark ? 0xffffff : 0x14213d) : colourOf('pretrained') }, l === used ? 1 : 0.3);
      const edge = rectangle(W, D, edgeMat);
      edge.position.y = y + 0.02;
      group.add(edge);
      materials.push(edgeMat);
      slices.push({ sheet, edge });
    }
    add({ id: 'qwen', x: X.qwen, z: NETWORK_Z, kind: 'pretrained', label: 'Qwen3-Embedding-0.6B' }, group, materials,
      `${L - 1} layers`, { port: (b) => ({ inX: b.min.x - 0.6, outX: b.max.x + 0.6, inY: 12, outY: usedY }),
                           detail: `${L - 1} layers · ${T} tokens × 1024 · every ${t.channel_step}th channel drawn` });
    ghost.visible = false;
    stage.scene.add(ghost);
    aboveTower = new THREE.Vector3(X.qwen, 0.8 + L * LAYER + 6, NETWORK_Z);
    towerTop = new THREE.Box3(new THREE.Vector3(X.qwen - W / 2, usedY - 6, NETWORK_Z - D / 2),
                              new THREE.Vector3(X.qwen + W / 2, aboveTower.y + 2, NETWORK_Z + D / 2));
  }

  function wall(id, key, x, label, labelAt) {
    const t = tensor(key);
    const m = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.05, transparent: true, opacity: 1,
      emissive: colourOf('reduce'), emissiveIntensity: 0 });
    m.userData.base = 1;
    const mesh = voxels({ shape: [32, 32], data: t.data }, { layout: 'plane', cell: 0.5, fill: 0.84, material: m, order: 'columns',
      colour: colouring === 'kind' ? kindColour('reduce') : undefined });
    const lift = -new THREE.Box3().setFromObject(mesh).min.y;
    const o = add({ id, x, y: lift, z: NETWORK_Z, kind: 'reduce', label }, mesh, [m], `${t.shape[0]}`,
      { port: (b) => ({ inX: b.min.x - 1, outX: b.max.x + 1, y: 8 }), labelAt });
    o.tensor = t;
    o.full = mesh.count;
  }
  wall('text_pooled', net.pooled, X.pooled, 'Mean');
  wall('text_standardised', net.standardised, X.standardised, 'Standardised',
       (b, c) => new THREE.Vector3(c.x, b.min.y + 1, b.max.z + 3));            // in front, clear of the mean's label

  // ---------------------------------------------------------------- land use

  // each building where the census puts it, to scale (a floor taken as 3.5 m), coloured by its
  // main use, on the satellite image of the same square, faded; the inner square dashed
  const lu = entry.landuse;
  const perMetre = IMAGE / lu.patch_m, FLOOR = 3.5 * perMetre, FOOT = 1.1;
  const useColour = lu.use_keys.map((key) => new THREE.Color(USES[key][1]));
  const useDot = (k) => `<i class="dot" style="background:${USES[lu.use_keys[k]][1]}"></i>${USES[lu.use_keys[k]][0]}`;
  const topUses = (shares, n) => shares.map((v, k) => [v, k]).sort((a, b) => b[0] - a[0]).slice(0, n);
  const blocks = lu.buildings.map(([east, north, floors]) => ({ x: east * perMetre, z: -north * perMetre, h: floors * FLOOR }));
  const towers = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1),
    material(THREE.MeshStandardMaterial, { roughness: 0.55, emissive: colourOf('source'), emissiveIntensity: 0 }), blocks.length);
  const raise = (k) => {                                 // k(i): how much of building i is standing
    const m4 = new THREE.Matrix4(), at = new THREE.Vector3(), size = new THREE.Vector3(), turn = new THREE.Quaternion();
    blocks.forEach((b, i) => {
      const h = Math.max(0.001, b.h * k(i));
      towers.setMatrixAt(i, m4.compose(at.set(b.x, 0.42 + h / 2, b.z), turn, size.set(FOOT, h, FOOT)));
    });
    towers.instanceMatrix.needsUpdate = true;
  };
  {
    lu.buildings.forEach(([, , , use], i) => towers.setColorAt(i, useColour[use]));
    raise(() => 1);
    const plateMat = material(THREE.MeshStandardMaterial, { color: sideColour, roughness: 0.95 });
    const plate = new THREE.Mesh(new THREE.BoxGeometry(IMAGE, 0.4, IMAGE), plateMat);
    plate.position.y = 0.2;
    const photo = material(THREE.MeshBasicMaterial, { map: aerialTop.map }, dark ? 0.3 : 0.4);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(IMAGE, IMAGE), photo);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = 0.41;
    const innerMat = material(THREE.LineDashedMaterial, { color: colourOf('source'), dashSize: 1.2, gapSize: 0.8 }, 0.9);
    const inner = rectangle(lu.inner_m * perMetre, lu.inner_m * perMetre, innerMat);
    inner.position.y = 0.45;
    add({ id: 'clue', x: X.clue, z: LANDUSE_Z, kind: 'source', label: 'CLUE buildings' },
      new THREE.Group().add(plate, ground, inner, towers), [plateMat, photo, innerMat, towers.material],
      topUses(lu.shares.floors, 3).map(([, k]) => useDot(k)).join(''),
      { port: (b) => ({ outX: b.max.x + 0.6, y: 6 }), labelAt: (b, c) => new THREE.Vector3(c.x, 1.5, b.max.z + 1),
        detail: `${lu.count} buildings in the ${lu.patch_m} m square, ${lu.year} · to scale, by floors · `
          + topUses(lu.shares.floors, 4).map(([v, k]) => `${useDot(k)} ${(v * 100).toFixed(0)}%`).join('') });
  }

  // the shares of the 13 uses, stacked: by buildings on the left, by floors on the right
  const segments = [];
  {
    const group = new THREE.Group(), H = 24, W = 2.6, gap = 0.12;
    const mats = useColour.map((c) => material(THREE.MeshStandardMaterial, { color: c, roughness: 0.5, emissive: c, emissiveIntensity: 0 }));
    [['buildings', -2.2], ['floors', 2.2]].forEach(([which, dx]) => {
      let y = 0;
      lu.shares[which].forEach((share, k) => {
        if (share <= 0) return;
        const h = share * H;
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(W, 1, W), mats[k]);
        segments.push({ mesh, x: dx, y, h: Math.max(0.001, h - gap), k });
        group.add(mesh);
        y += h;
      });
    });
    const stack = (q) => segments.forEach((s) => {
      const h = s.h * q(s);
      s.mesh.visible = h > 0.002;
      s.mesh.scale.y = Math.max(0.001, h);
      s.mesh.position.set(s.x, s.y + h / 2, 0);
    });
    stack(() => 1);
    segments.stack = stack;
    add({ id: 'use_shares', x: X.shares, z: LANDUSE_Z, kind: 'reduce', label: 'Use shares' }, group, mats, '13 uses',
      { port: (b) => ({ inX: b.min.x - 0.6, outX: b.max.x + 0.6, y: 12 }),
        detail: 'share of buildings (left) and of floors (right) for 13 uses · '
          + topUses(lu.shares.floors, 4).map(([v, k]) => `${useDot(k)} ${(v * 100).toFixed(0)}%`).join('') });
  }

  // ---------------------------------------------------------------- how the stages play out

  const pooledAt = new THREE.Vector3(X.pooled, 8, NETWORK_Z);
  const sliceAt = new THREE.Vector3(X.qwen, usedY, NETWORK_Z);
  return {
    objects, boxes: { graph_local: localBox, pool: towerTop.clone().union(objects.get('text_pooled').bounds) },
    edges: [
      ['segmentation', 'regions'], ['regions', 'dino'], ['dino', 'region_features'],
      ['region_features', 'imagery', 'pooled → 135'],
      ['text', 'qwen'], ['qwen', 'text_pooled'],
      ['text_pooled', 'text_standardised'], ['text_standardised', 'network', 'PCA → 64'],
      ['clue', 'use_shares'], ['use_shares', 'landuse', '17 + 26 → 43'],
    ],
    // everything at rest, as the layer view shows it
    reset() {
      lifted.visible = true;
      objects.get('segmentation').label?.classList.remove('gone');
      lifted.position.y = LIFT;
      tiles.forEach((t) => { t.mesh.position.copy(t.grid); t.mesh.rotation.x = 0; t.mesh.scale.setScalar(1); });
      dinoFills.forEach((f) => f.color.copy(dinoColour));
      lattice.scale.z = 1;
      links.forEach((l) => { l.mesh.scale.x = l.length; });
      paintGraph(links.length, hop2.length);
      drawTokens(tokens.length);
      slices.forEach((s) => { if (s.sheet) s.sheet.visible = true; s.edge.visible = true; });
      ghost.visible = false;
      raise(() => 1);
      segments.stack(() => 1);
    },
    // the segmentation rises off the image
    lift(p) {
      lifted.visible = p > 0;
      objects.get('segmentation').label?.classList.toggle('gone', p <= 0);
      lifted.position.y = 0.7 + (LIFT - 0.7) * ease(p);
    },
    // each region leaves its place on the segmentation for the grid, largest first
    cut(p) {
      tiles.forEach((t, i) => {
        const start = (i / tiles.length) * 0.55;
        const q = ease(phase(p, start, start + 0.4));
        t.mesh.position.lerpVectors(t.onImage, t.grid, q);
        t.mesh.position.y += Math.sin(Math.PI * q) * 8;
        t.mesh.rotation.x = (-Math.PI / 2) * (1 - q);
        t.mesh.scale.setScalar(t.small + (1 - t.small) * q);
      });
    },
    // the transformer's layers light in turn, then the table of region features fills
    dino(p) {
      const k = phase(p, 0, 0.6) * 12;
      dinoFills.forEach((f, i) => f.color.copy(i < k ? dinoLit : dinoColour));
      lattice.scale.z = Math.max(0.001, ease(phase(p, 0.55, 1)));
    },
    // the links to the neighbours grow, then the junctions two steps away light up
    graph(p) {
      const a = phase(p, 0.05, 0.6), n = links.length;
      let reached = 0;
      links.forEach((l, k) => {
        const q = ease(phase(a, (k / n) * 0.5, (k / n) * 0.5 + 0.5));
        l.mesh.scale.x = Math.max(0.001, l.length * q);
        if (q >= 1) reached++;
      });
      paintGraph(reached, Math.floor(phase(p, 0.6, 0.95) * hop2.length + 1e-6));
    },
    tokens(p) { drawTokens(Math.round(p * tokens.length)); },
    // the layers stack up, from the token embeddings to the last
    layers(p) {
      const k = Math.ceil(p * slices.length);
      slices.forEach((s, l) => { if (s.sheet) s.sheet.visible = l < k; s.edge.visible = l < k; });
    },
    // the layer the features are taken from is lifted out, averaged over its tokens (squeezed
    // along them to a single row) and carried to the mean
    pool(p) {
      const rise = ease(phase(p, 0.02, 0.3)), squash = ease(phase(p, 0.3, 0.55)), carry = ease(phase(p, 0.55, 0.8));
      ghost.visible = p > 0.02 && p < 0.8;
      ghost.position.lerpVectors(sliceAt, aboveTower, rise);
      // across first and down after, so that it clears the tower
      ghost.position.x += (pooledAt.x - aboveTower.x) * (1 - Math.pow(1 - carry, 3));
      ghost.position.y += (pooledAt.y - aboveTower.y) * Math.pow(carry, 3);
      ghost.scale.set(1 - 0.97 * squash, 1 - 0.5 * carry, 1);
    },
    // the buildings rise from the ground, nearest the junction first
    buildings(p) {
      const n = blocks.length;
      raise((i) => ease(phase(p, (i / n) * 0.6, (i / n) * 0.6 + 0.4)));
    },
    // the shares stack up one use at a time
    shares(p) { segments.stack((s) => ease(phase(p, (s.k / 13) * 0.7, (s.k / 13) * 0.7 + 0.3))); },
  };
}
