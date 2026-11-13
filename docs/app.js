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

// The runner backend of a submitted row. All paper rows were served by Ollama 0.18.0.
const SERVING = { ollama: 'Ollama', openai: 'OpenAI-compatible' };

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

// Model tables. The paradigm buttons and the value switch apply to the leaderboard and the community table alike.
// Each table includes the human reference as a row that is ranked with the models.
const view = { group: '', gap: false };
const renders = [];

function modelTable(t, tasks, models, showServing) {
  const human = { model: 'Human reference', human: true, params_b: null,
    values: Object.fromEntries(tasks.map(k => [k.code, k.kind === 'constant' ? null : k.human])) };
  const rows = [human, ...models];
  let sortKey = 'params_b', asc = true;

  function value(m, key) {
    if (key === 'model') return m.model;
    // Undisclosed sizes sort after every disclosed one; the human row has no size and sorts first.
    if (key === 'params_b') return m.human || m.params_b !== null ? m.params_b : Infinity;
    const k = tasks.find(x => x.code === key), v = m.values[key];
    if (v === null || v === undefined) return null;
    if (k.kind === 'constant') return v ? 1 : 0;
    return view.gap ? v - k.human : v;
  }
  function render() {
    t.replaceChildren();
    const shown = tasks.filter(k => !view.group || k.group === view.group);
    t.classList.toggle('few', shown.length < tasks.length);
    const head = el('tr');
    for (const [key, label, task] of [['model', 'Model'], ['params_b', 'Params (B)'], ...shown.map(k => [k.code, k.code, k])]) {
      const th = el('th', key === 'model' ? { class: 'm' } : {}, label);
      if (task) th.title = `${task.name}: ${task.measure}`;
      if (key === sortKey) th.classList.add('sorted', ...(asc ? ['asc'] : []));
      th.addEventListener('click', () => { asc = sortKey === key ? !asc : !task; sortKey = key; render(); });
      head.append(th);
    }
    t.append(head);
    const sorted = [...rows].sort((a, b) => {
      const x = value(a, sortKey), y = value(b, sortKey);
      const r = typeof x === 'string' ? x.localeCompare(y) : (x ?? -Infinity) - (y ?? -Infinity);
      return asc ? r : -r;
    });
    for (const m of sorted) {
      const r = el('tr', m.human ? { class: 'human' } : {});
      const name = el('td', { class: 'm' }, m.model + (m.native_thinking ? ' (thinking)' : ''));
      if (m.family) name.append(el('small', {}, m.family));
      if (showServing && m.serving) name.append(el('small', {}, SERVING[m.serving] || m.serving));
      const size = m.human ? el('td', {}, '–')
        : m.params_b === null ? el('td', { title: 'Size not disclosed' }, 'n/d') : el('td', {}, fmt(m.params_b, 1));
      r.append(name, size);
      for (const k of shown) {
        const v = m.values[k.code];
        if (k.kind === 'constant') {
          r.append(m.human ? el('td', { title: 'No human proportion exists' }, '–') : el('td', {}, v ? 'constant' : 'varied'));
          continue;
        }
        const td = el('td', {}, fmt(view.gap ? v - k.human : v));
        if (m.human) td.title = `${k.source}, n = ${k.human_n}, SD = ${fmt(k.human_sd)}`;
        else {
          // Distance in across-person human SDs, clipped at 3, as in paper Figure 3.
          const z = Math.max(-1, Math.min(1, (v - k.human) / (3 * k.human_sd)));
          td.style.background = `rgba(var(${z < 0 ? '--lo' : '--hi'}), ${(Math.abs(z) * 0.4).toFixed(2)})`;
        }
        r.append(td);
      }
      t.append(r);
    }
  }
  renders.push(render);
  render();
}

function viewButtons(tasks) {
  const box = document.getElementById('group-buttons');
  const groups = [['', 'All paradigms'], ...[...new Set(tasks.map(k => k.group))].map(g => [g, g])];
  const set = (buttons, pick) => buttons.forEach(b => { const on = b === pick; b.classList.toggle('on', on); b.setAttribute('aria-pressed', on); });
  for (const [g, label] of groups) {
    const b = el('button', { type: 'button', class: g ? '' : 'on', 'aria-pressed': !g }, label);
    b.addEventListener('click', () => { view.group = g; set([...box.children], b); renders.forEach(f => f()); });
    box.append(b);
  }
  const modes = [...document.querySelectorAll('#value-buttons button')];
  modes.forEach(b => b.addEventListener('click', () => { view.gap = b.dataset.gap === 'true'; set(modes, b); renders.forEach(f => f()); }));
}

function communityTable(tasks, community) {
  if (!community.models.length) {
    document.getElementById('community-empty').hidden = false;
    document.getElementById('community-table').parentElement.hidden = true;
    return;
  }
  modelTable(document.getElementById('community-table'), tasks, community.models, true);
}

document.getElementById('copy-bibtex').addEventListener('click', e => {
  navigator.clipboard.writeText(document.getElementById('bibtex').textContent).then(() => { e.target.textContent = 'Copied'; });
});

Promise.all(['data/leaderboard.json', 'data/community.json'].map(u => fetch(u).then(r => r.json()))).then(([data, community]) => {
  taskTable(data.tasks);
  summaryTable(data.tasks);
  viewButtons(data.tasks);
  modelTable(document.getElementById('model-table'), data.tasks, data.models, false);
  communityTable(data.tasks, community);
}).catch(err => {
  document.getElementById('results').append(el('p', { class: 'small' }, `Could not load data: ${err}`));
});
