// Project showcase: five views on one page, chosen by path so /report, /model, /imagery and /code
// can be linked directly (by hash in the static copy). The report list comes from whatever PDFs the
// report build has produced, so a newly finished chapter appears here without touching this file.

const VIEWS = { '/': 'overview', '/report': 'report', '/imagery': 'imagery', '/model': 'model', '/code': 'code' };
const PATHS = Object.fromEntries(Object.entries(VIEWS).map(([path, view]) => [view, path]));

// The page also runs as a static copy (GitHub Pages) written by export_static.py, which sets
// window.SHOWCASE_STATIC. There every address server.py answers is a file beside the page, and the
// view rides in the hash, since a static host has no page at /report.
const STATIC = Boolean(window.SHOWCASE_STATIC);
const STATIC_FILES = [
  [/^\/api\/showcase\/reports$/, () => 'data/reports.json'],
  [/^\/api\/showcase\/code$/, () => 'data/code.json'],
  [/^\/api\/showcase\/code\/file\?path=(.+)$/, (m) => `data/code/${decodeURIComponent(m[1])}.json`],
  [/^\/files\/report-thumb\/(.+)$/, (m) => `files/report-thumb/${m[1]}.png`],
  [/^\/api\/examples$/, () => 'data/examples.json'],
  [/^\/api\/examples\/(.+)$/, (m) => `examples/${m[1]}`],
  [/^\/tool$/, () => 'tool.html'],
  [/^\/(.*)$/, (m) => m[1]],              // anything else keeps its path, made relative
];
// an address on the server, or the file standing in for it in the static copy
function at(path) {
  if (!STATIC) return path;
  for (const [pattern, file] of STATIC_FILES) {
    const m = path.match(pattern);
    if (m) return file(m);
  }
  return path;
}
// a view (and its query) as the address bar shows it
const route = (view, query = '') => (STATIC ? `#${view}` : PATHS[view]) + (query ? `?${query}` : '');
const query = () => new URLSearchParams(STATIC ? location.hash.split('?')[1] || '' : location.search);
const currentView = () => (STATIC ? location.hash.slice(1).split('?')[0] : VIEWS[location.pathname]) || 'overview';

// the interactive 3D view of the settled network, then two tabs of the earlier web demo
// (https://cjsrc.github.io/melbourne-traffic-demo/, copied unchanged to showcase/demo/; the part
// after # names the demo's own tab), then the figures of Chapter 4
const DEMO = at('/showcase/demo/');
const VIEWS3D = {
  'Model framework in 3D': at('/showcase/viz/layers.html'),
  'Network map by time': `${DEMO}index.html#map`,
  'Study area & SAM segmentation': `${DEMO}index.html#sam`,
};
const FIGURES = [
  ...Object.keys(VIEWS3D).map((title) => [null, title]),
  ['fig4_graph.png', 'Junction network'],
  ['fig4_network_example.png', 'Network features'],
  ['fig4_configurations.png', 'Configurations compared'],
];

const $ = (id) => document.getElementById(id);
let reports = null;           // loaded once
let toolLoaded = false;
let figuresOpened = false;

// ---------------------------------------------------------------- routing

function show(view, push) {
  if (!Object.values(VIEWS).includes(view)) view = 'overview';
  document.querySelectorAll('.view').forEach((el) => el.classList.toggle('on', el.id === view));
  document.querySelectorAll('.bar nav a').forEach((a) => {
    if (a.dataset.view === view) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
  if (push) history.pushState({ view }, '', route(view));
  // the tool is a full application; load it the first time it is asked for, not on arrival
  if (view === 'imagery' && !toolLoaded) {
    $('tool-frame').src = at('/tool');
    toolLoaded = true;
  }
  if (view === 'report') loadReports();
  if (view === 'code') loadCode();
  // the first view is opened when the page is first shown, not on arrival: a 3D view laid out
  // while hidden has no size to draw at
  if (view === 'model' && !figuresOpened) {
    openFigure(0);
    figuresOpened = true;
  }
  window.scrollTo(0, 0);
}

document.addEventListener('click', (event) => {
  const link = event.target instanceof Element && event.target.closest('a[data-view]');
  if (!link || event.metaKey || event.ctrlKey || event.shiftKey) return;
  event.preventDefault();
  show(link.dataset.view, true);
});
window.addEventListener('popstate', () => show(currentView(), false));

// ---------------------------------------------------------------- report

async function fetchReports() {
  if (!reports) {
    const response = await fetch(at('/api/showcase/reports'));
    reports = await response.json();
  }
  return reports;
}

const LANGUAGE = { zh: 'Chinese', en: 'English' };

async function loadReports() {
  const toc = $('toc');
  if (toc.querySelector('button')) return;
  let data;
  try {
    data = await fetchReports();
  } catch (error) {
    $('viewer-empty').textContent = 'The report list could not be loaded.';
    return;
  }
  // one entry per chapter; where both languages exist, English wins
  const byChapter = {};
  let full = null;
  for (const item of data.items) {
    if (item.chapter === null) {
      if (!full || item.language === 'en') full = item;
      continue;
    }
    const held = byChapter[item.chapter];
    if (!held || item.language === 'en') byChapter[item.chapter] = item;
  }
  if (full && full.outline && full.outline.length) {
    listParts(full);
    return;
  }
  const entries = [];
  if (full) entries.push({ n: '', title: 'Full report', item: full });
  for (const [n, title] of Object.entries(data.chapters)) {
    entries.push({ n, title, item: byChapter[n] || null });
  }
  for (const entry of entries) {
    const li = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    button.disabled = !entry.item;
    const meta = entry.item
      ? `${LANGUAGE[entry.item.language]} draft · ${entry.item.pages ?? '?'} pages · ${entry.item.updated}`
      : 'In preparation';
    button.innerHTML = `<span class="n">${entry.n}</span><span class="t"></span><span class="meta"></span>`;
    button.querySelector('.t').textContent = entry.title;
    button.querySelector('.meta').textContent = meta;
    if (entry.item) {
      button.dataset.file = entry.item.file;
      button.addEventListener('click', () => openReport(entry, button));
    }
    li.appendChild(button);
    toc.appendChild(li);
  }
  const wanted = query().get('doc');
  const first = [...toc.querySelectorAll('button:not(:disabled)')];
  const pick = first.find((b) => b.dataset.file === wanted) || first[0];
  if (pick) pick.click();
  else $('viewer-empty').textContent = 'No chapter is finished yet.';
}

// the merged report (docs/report/scripts/make_report.py): one file, and the list moves within it
function listParts(full) {
  const toc = $('toc');
  for (const part of full.outline) {
    if (part.appendix && !toc.querySelector('.appendix-head')) {
      const head = document.createElement('li');
      head.className = 'list-head appendix-head';
      head.setAttribute('aria-hidden', 'true');
      head.textContent = 'Appendices';
      toc.appendChild(head);
    }
    const li = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.page = part.page;
    button.innerHTML = '<span class="n"></span><span class="t"></span><span class="meta"></span>';
    button.querySelector('.n').textContent = part.n;
    button.querySelector('.t').textContent = part.title;
    button.querySelector('.meta').textContent = `p. ${part.label}`;
    button.addEventListener('click', () => openPart(full, part.page, part.heading, button));
    li.appendChild(button);
    toc.appendChild(li);
  }
  const wanted = Number(query().get('page'));
  const pick = [...toc.querySelectorAll('button')].find((b) => Number(b.dataset.page) === wanted);
  if (pick) pick.click();
  else openPart(full, 1, `Full report · ${LANGUAGE[full.language]} draft · ${full.pages} pages · ${full.updated}`, null);
}

function openPart(full, page, heading, button) {
  document.querySelectorAll('#toc button').forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
  const url = at(`/files/report/${full.file}`);
  $('viewer-title').textContent = heading;
  // the PDF viewer reads #page only when it loads a file, not when the address changes after,
  // so each jump loads the file into a new frame -- from the browser's cache after the first
  const frame = $('viewer-frame').cloneNode(false);
  frame.src = `${url}#page=${page}`;
  frame.hidden = false;
  $('viewer-frame').replaceWith(frame);
  $('viewer-empty').hidden = true;
  $('viewer-open').href = `${url}#page=${page}`;
  $('viewer-download').href = url;
  for (const id of ['viewer-open', 'viewer-download']) $(id).hidden = false;
  history.replaceState({ view: 'report' }, '', route('report', page > 1 ? `page=${page}` : ''));
}

function openReport(entry, button) {
  document.querySelectorAll('#toc button').forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
  const url = at(`/files/report/${entry.item.file}`);
  $('viewer-title').textContent = entry.n ? `Chapter ${entry.n} · ${entry.title}` : entry.title;
  $('viewer-frame').src = url;
  $('viewer-frame').hidden = false;
  $('viewer-empty').hidden = true;
  for (const id of ['viewer-open', 'viewer-download']) {
    $(id).href = url;
    $(id).hidden = false;
  }
  history.replaceState({ view: 'report' }, '', route('report', `doc=${encodeURIComponent(entry.item.file)}`));
}

// ---------------------------------------------------------------- model figures

function buildFigures() {
  const list = $('figures');
  FIGURES.forEach(([file, title], i) => {
    const li = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    button.innerHTML = `<span class="n">${String(i + 1).padStart(2, '0')}</span><span class="t"></span>`;
    button.querySelector('.t').textContent = title;
    button.addEventListener('click', () => openFigure(i));
    li.appendChild(button);
    list.appendChild(li);
  });
}

let figureIndex = 0;
function openFigure(i) {
  figureIndex = (i + FIGURES.length) % FIGURES.length;
  const [file, title] = FIGURES[figureIndex];
  const interactive = file === null;
  const url = interactive ? VIEWS3D[title] : at(`/files/figure/${file}`);
  // each interactive view loads the first time it is opened and keeps its state after that;
  // the figures swap in and out beside them
  const frames = $('frames');
  let frame = frames.querySelector(`iframe[data-title="${title}"]`);
  if (interactive && !frame) {
    frame = document.createElement('iframe');
    frame.className = 'view-frame';
    frame.dataset.title = title;
    frame.title = title;
    frame.src = url;
    if (url.startsWith(DEMO)) frame.addEventListener('load', () => demoTab(frame, url.split('#')[1]));
    frames.appendChild(frame);
  }
  frames.querySelectorAll('iframe').forEach((f) => { f.hidden = f !== frame || !interactive; });
  frames.hidden = !interactive;
  // a demo map hidden in the meantime has lost its size; Leaflet measures again on resize
  if (interactive && url.startsWith(DEMO)) frame.contentWindow.dispatchEvent(new Event('resize'));
  $('figure-stage').hidden = interactive;
  if (!interactive) {
    $('figure-img').src = url;
    $('figure-img').alt = title;
  }
  $('figure-title').textContent = title;
  $('figure-open').href = url;
  // opened on its own, the demo page would show all six of its tabs and its broken basemap
  $('figure-open').hidden = interactive && url.startsWith(DEMO);
  document.querySelectorAll('#figures button').forEach((b, k) => b.setAttribute('aria-pressed', String(k === figureIndex)));
}
// The demo page is shown one tab at a time: its title bar and tab row are hidden by a style added
// from here, so the copy stays identical to the published page. Also hidden: the study-area note,
// which promises per-site examples below that this version of the page does not have. The maps
// are fitted to the frame's height, so neither has to be scrolled to be seen whole.
function demoTab(frame, tab) {
  const page = frame.contentDocument;
  const style = page.createElement('style');
  style.textContent = 'header, .tabs, #v-sam .callout { display: none !important; }'
    + '#map { height: clamp(400px, calc(100vh - 260px), 560px) !important; }'
    + '#sammap { height: clamp(400px, calc(100vh - 230px), 620px) !important; }'
    + '#map .leaflet-tile-pane { filter: grayscale(1) contrast(.9) brightness(1.06); }';
  page.head.appendChild(style);
  if (tab === 'map') {
    // the published page's CARTO basemap now answers every tile with "API KEY REQUIRED";
    // OpenStreetMap's own tiles need no key (greyed above, as the light CARTO map was)
    const L = frame.contentWindow.L, map = frame.contentWindow.eval('map');
    map.eachLayer((layer) => { if (layer instanceof L.TileLayer) map.removeLayer(layer); });
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
      { maxZoom: 19, attribution: '© OpenStreetMap contributors' }).addTo(map);
  }
  page.querySelector(`.tab[data-v="${tab}"]`).click();
}
$('figure-img').addEventListener('click', () => window.open($('figure-open').href, '_blank', 'noopener'));
document.addEventListener('keydown', (event) => {
  const typing = event.target instanceof Element && event.target.closest('input, textarea');
  if (!$('model').classList.contains('on') || typing) return;
  if (event.key === 'ArrowDown' || event.key === 'ArrowRight') { openFigure(figureIndex + 1); event.preventDefault(); }
  if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') { openFigure(figureIndex - 1); event.preventDefault(); }
});

// ---------------------------------------------------------------- code

let code = null;              // the code release's manifest, loaded once
let codeToken = 0;            // the newest file asked for; slower answers for older ones are dropped
const LANG = { py: 'py', js: 'js', html: 'html', css: 'css', ps1: 'ps1', cmd: 'cmd' };

async function fetchCode() {
  if (!code) code = await (await fetch(at('/api/showcase/code'))).json();
  return code;
}

// each stage's files under its title, as the manifest groups them; the READMEs are not shown
async function loadCode() {
  const list = $('code-files');
  if (list.querySelector('button')) return;
  let data;
  try {
    data = await fetchCode();
  } catch (error) {
    $('code-body').innerHTML = '<div class="empty">The code release could not be loaded.</div>';
    return;
  }
  const head = (text) => {
    const li = document.createElement('li');
    li.className = 'list-head';
    li.textContent = text;
    list.appendChild(li);
  };
  const add = (entry, tag, title, meta) => {
    const li = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    button.innerHTML = '<span class="n"></span><span class="t"></span><span class="meta"></span>';
    button.querySelector('.n').textContent = tag;
    button.querySelector('.t').textContent = title;
    button.querySelector('.meta').textContent = meta;
    button.dataset.path = entry.file.path;
    button.addEventListener('click', () => openCode(entry, button));
    li.appendChild(button);
    list.appendChild(li);
  };
  for (const stage of data.stages) {
    head(stage.title);
    for (const file of stage.files) add({ file }, LANG[file.name.split('.').pop()] || 'txt', file.name, file.role);
  }
  const wanted = query().get('file');
  const buttons = [...list.querySelectorAll('button')];
  (buttons.find((b) => b.dataset.path === wanted) || buttons[0]).click();
}

async function openCode(entry, button) {
  document.querySelectorAll('#code-files button').forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
  button.scrollIntoView({ block: 'nearest' });
  const body = $('code-body');
  const token = ++codeToken;
  const { file } = entry;
  const url = at(`/files/code/${file.path.split('/').map(encodeURIComponent).join('/')}`);
  $('code-title').textContent = file.path;
  for (const id of ['code-raw', 'code-download']) { $(id).href = url; $(id).hidden = false; }
  $('code-download').setAttribute('download', file.path.split('/').pop());
  body.scrollTop = 0;
  history.replaceState({ view: 'code' }, '', route('code', `file=${encodeURIComponent(file.path)}`));
  body.innerHTML = '<div class="empty">Loading…</div>';
  let data;
  try {
    data = await (await fetch(at(`/api/showcase/code/file?path=${encodeURIComponent(file.path)}`))).json();
  } catch (error) {
    if (token === codeToken) body.innerHTML = '<div class="empty">This file could not be loaded.</div>';
    return;
  }
  if (token !== codeToken) return;
  const meta = document.createElement('div');
  meta.className = 'code-meta';
  meta.innerHTML = '<span class="role"></span><span class="facts"></span><span class="src"></span>';
  meta.querySelector('.role').textContent = data.role;
  meta.querySelector('.facts').textContent = `${data.language} · ${data.lines.toLocaleString('en')} lines`;
  meta.querySelector('.src').textContent = `from ${data.source}`;
  const source = document.createElement('div');
  source.innerHTML = data.html;
  body.replaceChildren(meta, source);
}

// ---------------------------------------------------------------- overview thumbnails

async function loadThumbs() {
  const views = Object.keys(VIEWS3D).length;
  $('sub-model').textContent = `${views} interactive views · ${FIGURES.length - views} figures`;
  try {
    const data = await fetchReports();
    const full = data.items.find((item) => item.chapter === null);
    const first = full || data.items[0];
    if (first) {
      $('thumb-report').src = at(`/files/report-thumb/${first.file}`);
      $('thumb-report').hidden = false;
    }
    const chapters = new Set(data.items.filter((item) => item.chapter !== null).map((item) => item.chapter));
    $('sub-report').textContent = full ? `${full.pages} pages · PDF`
      : chapters.size === 1 ? '1 chapter · PDF' : `${chapters.size} chapters · PDF`;
  } catch (error) { /* the card still works without a preview */ }
  try {
    const examples = (await (await fetch(at('/api/examples'))).json()).items;
    if (examples.length) {
      $('thumb-imagery').src = at(`/api/examples/${encodeURIComponent(examples[0].id)}/image.png`);
      $('thumb-imagery').hidden = false;
    }
  } catch (error) { /* likewise */ }
  try {
    const data = await fetchCode();
    $('sub-code').textContent = `${data.files} files · ${data.lines.toLocaleString('en')} lines`;
  } catch (error) { /* likewise */ }
}

buildFigures();
loadThumbs();
show(currentView(), false);
