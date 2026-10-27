// CogBattery site: task guide, panel summary, and per-model results from data/leaderboard.json.
const TASK_GUIDE = {
  DS: ['Repeat digits forward or backward.', 'No timing or adaptive length.'],
  NB: ['Check whether the current letter matches 2 or 3 steps back.', 'MATCH/NO MATCH replaces keys; history visible; no timing, practice, or adaptive load.'],
  OS: ['Judge equations, then recall letters.', 'Equations from public pools; no processing deadline.'],
  ST: ['Name the color label; ignore the word.', 'Color labels replace ink; no reaction time.'],
  FL: ['Find the center symbol among matching, conflicting, or neutral neighbors.', 'Text arrows and cues; LEFT/RIGHT answers; no cue timing or deadline.'],
  GN: ['PRESS for the GO color; otherwise WITHHOLD.', 'Color words; recent history visible; no response window.'],
  CVLT: ['Recall list words repeatedly, then recognize them.', 'Written list stays visible; no delays.'],
  DRM: ['Recognize studied words, related lures, and unrelated words, with or without warning.', 'Both warning conditions per model; study words visible; no delay.'],
  SM: ['Identify who provided information in quoted stories.', 'Three stories per model; stories visible.'],
  FB: ['Answer location questions in false- and true-belief stories.', 'Fixed scoring rules; both lists per model.'],
  EP: ['Answer mental-state and control questions at levels 1 to 7.', 'A/B answers; all format combinations per model; no confidence ratings.'],
  CAL: ['Answer French two-choice questions and rate confidence.', 'Each item in both answer orders.'],
  WAG: ['Classify grammar strings, then choose a low or high wager.', 'Visible strings; both grammars and orders; no real stakes.'],
};

const el = (tag, attrs = {}, text) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
};
const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? '' : x.toFixed(d);

function taskTable(tasks) {
  const t = document.getElementById('task-table');
  const head = el('tr');
  ['Task', 'What the model does', 'Key measure', 'Human source', 'Changes for models'].forEach(h => head.append(el('th', {}, h)));
  t.append(head);
  let group = '';
  for (const k of tasks) {
    if (k.group !== group) {
      group = k.group;
      const r = el('tr', { class: 'group' }); const c = el('td', { colspan: 5 }, group); r.append(c); t.append(r);
    }
    const r = el('tr');
    r.append(el('td', {}, `${k.name} (${k.code})`), el('td', {}, TASK_GUIDE[k.code][0]), el('td', {}, k.measure),
      el('td', {}, k.source), el('td', {}, TASK_GUIDE[k.code][1]));
    t.append(r);
  }
}

function summaryTable(tasks) {
  const t = document.getElementById('summary-table');
  const head = el('tr');
  ['Task', 'Key measure', 'Human', 'Model mean', 'Model minus human', '95% interval'].forEach((h, i) =>
    head.append(el('th', i > 1 ? { class: 'num' } : {}, h)));
  t.append(head);
  for (const k of tasks) {
    const r = el('tr');
    r.append(el('td', {}, k.code), el('td', {}, k.measure));
    if (k.kind === 'constant') {
      const c = el('td', { colspan: 4 }, `${k.constant_models} of ${k.models} models always place the same wager. No human proportion exists.`);
      r.append(c);
    } else {
      r.append(el('td', { class: 'num' }, fmt(k.human)), el('td', { class: 'num' }, fmt(k.model_mean)),
        el('td', { class: 'num' }, fmt(k.gap)), el('td', { class: 'num' }, `[${fmt(k.lower)}, ${fmt(k.upper)}]`));
    }
    t.append(r);
  }
}

function modelTable(data) {
  const t = document.getElementById('model-table');
  const famSel = document.getElementById('family');
  const gapBox = document.getElementById('gapmode');
  const tasks = data.tasks;
  [...new Set(data.models.map(m => m.family))].sort().forEach(f => famSel.append(el('option', { value: f }, f)));
  // Spread of each measure across models, for shading.
  const sd = {};
  for (const k of tasks) {
    if (k.kind !== 'value') continue;
    const v = data.models.map(m => m.values[k.code]).filter(x => typeof x === 'number');
    const mu = v.reduce((a, b) => a + b, 0) / v.length;
    sd[k.code] = Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / v.length) || 1;
  }
  const cols = [{ key: 'model', label: 'Model' }, { key: 'family', label: 'Family' },
    { key: 'params_b', label: 'Params (B)', num: true }, ...tasks.map(k => ({ key: k.code, label: k.code, num: true, task: k }))];
  let sortKey = 'params_b', asc = true;

  function value(m, c) {
    if (!c.task) return m[c.key];
    const v = m.values[c.key];
    if (c.task.kind === 'constant') return v ? 1 : 0;
    return gapBox.checked && typeof v === 'number' ? v - c.task.human : v;
  }
  function render() {
    t.replaceChildren();
    const head = el('tr');
    for (const c of cols) {
      const th = el('th', c.num ? { class: 'num' } : {}, c.label);
      if (c.task) th.title = `${c.task.name}: ${c.task.measure}`;
      if (c.key === sortKey) th.classList.add('sorted', ...(asc ? ['asc'] : []));
      th.addEventListener('click', () => { asc = sortKey === c.key ? !asc : true; sortKey = c.key; render(); });
      head.append(th);
    }
    t.append(head);
    const col = cols.find(c => c.key === sortKey);
    const rows = data.models.filter(m => !famSel.value || m.family === famSel.value)
      .sort((a, b) => {
        const x = value(a, col), y = value(b, col);
        const r = typeof x === 'string' ? x.localeCompare(y) : (x ?? -Infinity) - (y ?? -Infinity);
        return asc ? r : -r;
      });
    for (const m of rows) {
      const r = el('tr');
      r.append(el('td', { class: 'm' }, m.model + (m.native_thinking ? ' (thinking)' : '')), el('td', {}, m.family),
        el('td', { class: 'num' }, fmt(m.params_b, 1)));
      for (const k of tasks) {
        const v = m.values[k.code];
        if (k.kind === 'constant') { r.append(el('td', { class: 'num' }, v ? 'constant' : 'varied')); continue; }
        const shown = gapBox.checked ? v - k.human : v;
        const td = el('td', { class: 'num' }, fmt(shown));
        const z = Math.max(-1, Math.min(1, (v - k.human) / (2 * sd[k.code])));
        td.style.background = `rgba(var(${z < 0 ? '--lo' : '--hi'}), ${(Math.abs(z) * 0.5).toFixed(2)})`;
        r.append(td);
      }
      t.append(r);
    }
  }
  famSel.addEventListener('change', render);
  gapBox.addEventListener('change', render);
  render();
}

fetch('data/leaderboard.json').then(r => r.json()).then(data => {
  taskTable(data.tasks);
  summaryTable(data.tasks);
  modelTable(data);
}).catch(err => {
  document.getElementById('results').append(el('p', { class: 'small' }, `Could not load data: ${err}`));
});
