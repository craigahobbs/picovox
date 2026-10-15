// The training dashboard - reads the neural training runs' logs (build/nn/runs/<run>/log.jsonl and train.log), the
// data pipelines' logs (build/nn/**/*.log), and samples (build/nn/**/*.wav) through python3 -m http.server's directory
// listings, and refreshes every second. Read-only: it never touches the training.


// Where the repository and the build live relative to this page, and timing (ms)
const rootUrl = new URL('../../', import.meta.url);
const buildUrl = new URL('build/nn/', rootUrl);
const refreshMs = 1000;
const stalledMs = 3 * 60 * 1000;
const stoppedMs = 30 * 60 * 1000;


// Elements
const runsElement = document.getElementById('runs');
const chartsElement = document.getElementById('charts');
const chartsTitle = document.getElementById('chartsTitle');
const logsElement = document.getElementById('logs');
const samplesElement = document.getElementById('samples');
const pipelineSection = document.getElementById('pipelineSection');
const pipelineTitle = document.getElementById('pipelineTitle');
const pipelineFacts = document.getElementById('pipelineFacts');
const pipelineBar = document.getElementById('pipelineBar');
const phasesElement = document.getElementById('phases');
const updatedElement = document.getElementById('updated');
const checkpointSection = document.getElementById('checkpointSection');
const checkpointsElement = document.getElementById('checkpoints');


// State - runs by name ({name, entries, info, modified}), the selected run, and fetched files by URL
const runs = new Map();
let selectedRun = null;
const fileCache = new Map();


// Each active log's progress readings by path - [{time, fraction}], for the time-left estimates
const progressHistory = new Map();


// Create an element with a class and text
function make(tag, className = '', text = '') {
    const created = document.createElement(tag);
    if (className !== '') {
        created.className = className;
    }
    if (text !== '') {
        created.textContent = text;
    }
    return created;
}


// List a directory (python's http.server listing) - [{name, isDirectory}]
async function listDirectory(url) {
    try {
        const response = await fetch(url, {'cache': 'no-store'});
        if (!response.ok) {
            return [];
        }
        const html = await response.text();
        const document = new DOMParser().parseFromString(html, 'text/html');
        return [...document.querySelectorAll('li a')].map((link) => {
            const href = link.getAttribute('href');
            return {'name': decodeURIComponent(href.replace(/\/$/, '')), 'isDirectory': href.endsWith('/')};
        }).filter((entry) => !entry.name.startsWith('.'));
    } catch {
        return [];
    }
}


// Fetch a text file, reusing the cached text when its Last-Modified is unchanged - {text, modified} or null. Last-Modified
// has one-second resolution, so a file fetched within two seconds of its last change is fetched in full next time (a
// conditional request could miss a write in the same second). Files quiet for 5 minutes are checked every 10 seconds.
async function fetchText(url) {
    try {
        const cached = fileCache.get(String(url));
        if (cached && cached.settled && Date.now() - cached.modified > 5 * 60 * 1000 && Date.now() - cached.checked < 10000) {
            return cached;
        }
        const headers = cached && cached.settled ? {'If-Modified-Since': cached.lastModified} : {};
        const response = await fetch(url, {'cache': 'no-store', headers});
        if (response.status === 304 && cached) {
            cached.checked = Date.now();
            return cached;
        }
        if (!response.ok) {
            // Not there (yet)
            fileCache.delete(String(url));
            return null;
        }
        const lastModified = response.headers.get('Last-Modified') || '';
        const modified = lastModified ? Date.parse(lastModified) : Date.now();
        const served = Date.parse(response.headers.get('Date') || '') || Date.now();
        const settled = lastModified !== '' && served - modified >= 2000;
        const result = {'text': await response.text(), lastModified, modified, settled, 'checked': Date.now()};
        fileCache.set(String(url), result);
        return result;
    } catch {
        // A failed request - the last known text, if any
        return fileCache.get(String(url)) ?? null;
    }
}


// Parse a run's structured log - {train: [...], val: [...]}
function parseJsonLog(text) {
    const entries = {'train': [], 'val': []};
    for (const line of text.split('\n')) {
        if (line.trim() === '') {
            continue;
        }
        try {
            const record = JSON.parse(line);
            for (const [kind, value] of Object.entries(record)) {
                if (value && typeof value === 'object' && typeof value.step === 'number') {
                    (entries[kind] ||= []).push(value);
                }
            }
        } catch {
            // A partly written last line
        }
    }
    return entries;
}


// Parse a run's text log for its header facts - the target steps, the parameter count, and the data size
function parseTextLog(text) {
    const info = {};
    // The target from the latest launch's header (a resumed run may have changed it)
    const target = [...text.matchAll(/(\d+) steps from (\d+)/g)].pop();
    if (target) {
        info.targetSteps = Number(target[1]);
        info.startStep = Number(target[2]);
    }
    const parameters = text.match(/model: (\d+) parameters/);
    if (parameters) {
        info.parameters = Number(parameters[1]);
    }
    const data = text.match(/data: ([^\n]+)/);
    if (data) {
        info.data = data[1].replace(/\s*\(.*\)$/, '');
    }
    const continued = text.match(/stopped at step (\d+) \(continued as ([^)\s]+)\)/);
    if (continued) {
        const [, stoppedStep, continuedAs] = continued;
        info.stoppedStep = Number(stoppedStep);
        info.continuedAs = continuedAs;
    }
    // An error counts only in the latest launch (after the last header) - a resumed run's log keeps earlier failures
    const launches = text.split(/^.*steps from \d+.*$/m);
    const latest = launches[launches.length - 1];
    if (/Traceback|Error:/.test(latest)) {
        info.error = latest.split('\n').filter((line) => /Error/.test(line)).pop();
    }
    return info;
}


// Format a duration (s)
function formatDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) {
        return '-';
    }
    if (seconds < 90) {
        return `${Math.round(seconds)} s`;
    }
    if (seconds < 5400) {
        return `${Math.round(seconds / 60)} min`;
    }
    return `${(seconds / 3600).toFixed(1)} h`;
}


// The median of numbers
function median(values) {
    const sorted = [...values].sort((first, second) => first - second);
    return sorted.length === 0 ? NaN : sorted[Math.floor(sorted.length / 2)];
}


// A run's status - live (recently updated), stalled (quiet for minutes), stopped (quiet for long), finished, or failed
function runStatus(run) {
    const train = run.entries.train || [];
    const last = train[train.length - 1];
    if (run.info.error) {
        return 'failed';
    }
    if (last && run.info.targetSteps && last.step >= run.info.targetSteps) {
        return 'finished';
    }
    const quiet = Date.now() - run.modified;
    if (quiet < stalledMs) {
        return 'live';
    }
    return quiet < stoppedMs ? 'stalled' : 'stopped';
}


// Refresh the runs - their logs, for runs whose logs changed
async function refreshRuns() {
    const entries = await listDirectory(new URL('runs/', buildUrl));
    for (const entry of entries.filter((item) => item.isDirectory)) {
        const base = new URL(`runs/${encodeURIComponent(entry.name)}/`, buildUrl);
        const [jsonLog, textLog] = await Promise.all([fetchText(new URL('log.jsonl', base)), fetchText(new URL('train.log', base))]);
        if (jsonLog === null && textLog === null) {
            continue;
        }
        runs.set(entry.name, {
            'name': entry.name,
            'entries': jsonLog === null ? {'train': [], 'val': []} : parseJsonLog(jsonLog.text),
            'info': textLog === null ? {} : parseTextLog(textLog.text),
            'modified': Math.max(jsonLog?.modified || 0, textLog?.modified || 0)
        });
    }
    // First-seen order - each run's place is fixed when the page first sees it (remembered per browser)
    let changed = false;
    for (const run of runs.values()) {
        if (!(run.name in runOrder)) {
            runOrder[run.name] = runOrderLoaded ? Date.now() : run.modified;
            changed = true;
        }
    }
    if (changed) {
        storageSet(runOrderKey, JSON.stringify(runOrder));
    }
    runOrderLoaded = true;
    if (selectedRun === null || !runs.has(selectedRun)) {
        // The remembered run, else the newest live run of the first group, else the newest run there, else any
        const remembered = storageGet(selectedKey);
        const [firstGroup] = groupRuns();
        const live = firstGroup.runs.find((run) => runStatus(run) === 'live');
        selectedRun = runs.has(remembered) ? remembered : (live || firstGroup.runs[0] || [...runs.values()][0])?.name ?? null;
    }
}


// The run groups, each run in the first group whose prefixes match its name - the running runs (live or stalled), then
// the done ones (finished, stopped, or failed), each newest first by first-seen order (so a run that finishes moves to
// the done runs, and the others keep their places)
const runGroupSpecs = [
    {'title': 'Multi-speaker voice', 'prefixes': ['multi-'], 'open': true},
    {'title': 'G2P', 'prefixes': ['g2p-'], 'open': false},
    {'title': 'Voice students (distilled and from scratch)', 'prefixes': ['distill-', 'scratch-'], 'open': true},
    {'title': 'Earlier single-speaker voices', 'prefixes': ['slt-', 'bdl-', 'joint-'], 'open': false},
    {'title': 'Other', 'prefixes': [''], 'open': false}
];
const runOrderKey = 'training-dashboard-run-order-v1';
const selectedKey = 'training-dashboard-run-v1';
const groupsOpenKey = 'training-dashboard-run-groups-v1';
let runOrder = {};
try {
    runOrder = JSON.parse(storageGet(runOrderKey)) || {};
} catch {
    runOrder = {};
}
let runOrderLoaded = false;
function groupRuns() {
    const groups = runGroupSpecs.map((spec) => ({...spec, 'runs': []}));
    for (const run of runs.values()) {
        groups.find((group) => group.prefixes.some((prefix) => run.name.startsWith(prefix))).runs.push(run);
    }
    for (const group of groups) {
        const seen = (run) => runOrder[run.name] ?? 0;
        const done = (run) => (['live', 'stalled'].includes(runStatus(run)) ? 0 : 1);
        group.runs.sort((first, second) => done(first) - done(second) || seen(second) - seen(first) ||
            first.name.localeCompare(second.name));
    }
    return groups;
}


// Select a run - remembered per browser
function selectRun(name) {
    selectedRun = name;
    storageSet(selectedKey, name);
    renderRuns();
    renderCharts(true);
}


// Render the run cards
function renderRuns() {
    // The Training tab's fact - the first group's live run (the voice training), by first-seen order
    const [live] = groupRuns()[0].runs.filter((run) => runStatus(run) === 'live');
    const liveLast = live ? (live.entries.train || []).slice(-1)[0] : null;
    let trainingFact = '';
    if (live) {
        trainingFact = live.info.targetSteps && liveLast ? `${Math.round(100 * liveLast.step / live.info.targetSteps)}%` : 'live';
    }
    setTabFact('tabFactTraining', trainingFact);
    if (runs.size === 0) {
        runsElement.replaceChildren(make('p', 'empty', 'No runs yet - training writes to build/nn/runs/<run>/.'));
        return;
    }
    let groupsOpen = {};
    try {
        groupsOpen = JSON.parse(storageGet(groupsOpenKey)) || {};
    } catch {
        groupsOpen = {};
    }
    const groups = groupRuns().filter((group) => group.runs.length !== 0);
    // The groups' elements persist (so their open state does), and their cards are rebuilt each refresh
    for (const group of groups) {
        if (!runGroupElements.has(group.title)) {
            const element = make('details', 'run-group');
            element.open = groupsOpen[group.title] ?? group.open;
            element.addEventListener('toggle', () => {
                groupsOpen[group.title] = element.open;
                storageSet(groupsOpenKey, JSON.stringify(groupsOpen));
            });
            const summary = make('summary');
            const title = make('h3', '', group.title);
            const facts = make('span', 'facts');
            const bar = make('div', 'bar');
            const fill = make('div');
            bar.append(fill);
            summary.append(title, facts, bar);
            const grid = make('div', 'runs');
            element.append(summary, grid);
            runGroupElements.set(group.title, {element, facts, fill, grid});
        }
        const {facts, fill, grid} = runGroupElements.get(group.title);
        const progress = groupProgress(group.runs);
        const counts = `${group.runs.length} run${group.runs.length === 1 ? '' : 's'}${progress.live ? ` · ${progress.live} live` : ''}`;
        const percent = Number.isFinite(progress.fraction) ? ` · ${Math.round(100 * progress.fraction)}% of steps` : '';
        const left = Number.isFinite(progress.secondsLeft) ? ` · ${formatDuration(progress.secondsLeft)} left` : '';
        facts.textContent = `${counts}${percent}${left}`;
        fill.style.width = `${(100 * (progress.fraction || 0)).toFixed(1)}%`;
        grid.replaceChildren(...group.runs.map(runCard));
    }
    const elements = groups.map((group) => runGroupElements.get(group.title).element);
    if (elements.some((element, index) => runsElement.children[index] !== element) || runsElement.children.length !== elements.length) {
        runsElement.replaceChildren(...elements);
    }
}


// A group's overall progress - {fraction (steps done of the runs' targets), live (count), secondsLeft (the longest live
// run's)}; a run stopped and continued as another counts as complete at its stopping step
function groupProgress(members) {
    let done = 0;
    let total = 0;
    let live = 0;
    let secondsLeft = NaN;
    for (const run of members) {
        const train = run.entries.train || [];
        const step = train.length === 0 ? 0 : train[train.length - 1].step;
        const target = run.info.continuedAs ? run.info.stoppedStep : run.info.targetSteps;
        if (target) {
            done += Math.min(step, target);
            total += target;
        }
        if (runStatus(run) === 'live') {
            live += 1;
            const secondsPerStep = median(train.slice(-20).map((entry) => entry.secondsPerStep).filter(Number.isFinite));
            const remaining = target ? (target - step) * secondsPerStep : NaN;
            if (Number.isFinite(remaining)) {
                secondsLeft = Number.isFinite(secondsLeft) ? Math.max(secondsLeft, remaining) : remaining;
            }
        }
    }
    return {'fraction': total === 0 ? NaN : done / total, live, secondsLeft};
}


// A run's card
const runGroupElements = new Map();
function runCard(run) {
    const train = run.entries.train || [];
    const last = train[train.length - 1];
    const status = runStatus(run);
    const card = make('button', 'run');
    card.type = 'button';
    card.setAttribute('aria-pressed', String(run.name === selectedRun));
    const head = make('div', 'run-head');
    head.append(make('span', 'run-name', run.name), make('span', `pill ${status}`, status));
    const bar = make('div', 'bar');
    const fill = make('div');
    const fraction = last && run.info.targetSteps ? Math.min(1, last.step / run.info.targetSteps) : 0;
    fill.style.width = `${(100 * fraction).toFixed(1)}%`;
    bar.append(fill);
    const recent = train.slice(-20).map((entry) => entry.secondsPerStep).filter(Number.isFinite);
    const secondsPerStep = median(recent);
    const remaining = last && run.info.targetSteps ? (run.info.targetSteps - last.step) * secondsPerStep : NaN;
    const timeLeft = status === 'live' ? `${formatDuration(remaining)} left` : '-';
    const val = run.entries.val || [];
    const lastVal = val[val.length - 1];
    const facts = [
        `step ${last ? last.step.toLocaleString() : 0}${run.info.targetSteps ? ` / ${run.info.targetSteps.toLocaleString()}` : ''}`,
        `${Number.isFinite(secondsPerStep) ? secondsPerStep.toFixed(2) : '-'} s/step · ${timeLeft}`,
        `train total ${last ? last.total.toFixed(3) : '-'}${lastVal ? ` · val ${(lastVal.spectral ?? lastVal.total).toFixed(3)}` : ''}`,
        `updated ${formatDuration((Date.now() - run.modified) / 1000)} ago`
    ];
    if (run.info.continuedAs) {
        facts.push(`continued as ${run.info.continuedAs} at step ${run.info.stoppedStep.toLocaleString()}`);
    }
    const parent = parentRun(run);
    if (parent) {
        facts.push(`continues ${parent.run.name} from step ${parent.offset.toLocaleString()}`);
    }
    if (run.info.parameters) {
        facts.push(`${run.info.parameters.toLocaleString()} parameters`);
    }
    if (run.info.error) {
        facts.push(run.info.error);
    }
    const factsElement = make('div', 'facts');
    for (const fact of facts) {
        factsElement.append(make('div', '', fact));
    }
    card.append(head, bar, factsElement);
    card.addEventListener('click', () => selectRun(run.name));
    return card;
}


// The run a run continues - {run, offset} (the parent's step it started from), or null
function parentRun(run) {
    const parent = [...runs.values()].find((other) => other.info.continuedAs === run.name);
    return parent ? {'run': parent, 'offset': parent.info.stoppedStep} : null;
}


// An exponential moving average of a series' values
function smoothSeries(points, factor) {
    let value = null;
    return points.map(([xValue, yValue]) => {
        value = value === null ? yValue : value * factor + yValue * (1 - factor);
        return [xValue, value];
    });
}


// Draw a line chart - series: [{label, points: [[x, y]], color}], with optional log scale
function drawChart(canvas, series, {logScale = false} = {}) {
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const context = canvas.getContext('2d');
    context.scale(ratio, ratio);
    const styles = getComputedStyle(document.documentElement);
    const muted = styles.getPropertyValue('--muted');
    const rule = styles.getPropertyValue('--rule');
    const all = series.flatMap((item) => item.points).filter(([, yValue]) => Number.isFinite(yValue) && (!logScale || yValue > 0));
    if (all.length < 2) {
        context.fillStyle = muted;
        context.font = '12px system-ui, sans-serif';
        context.fillText('Not enough data yet', 12, height / 2);
        return;
    }
    const transform = (yValue) => (logScale ? Math.log10(yValue) : yValue);
    const xs = all.map(([xValue]) => xValue);
    const ys = all.map(([, yValue]) => transform(yValue));
    const [xMin, xMax] = [Math.min(...xs), Math.max(...xs)];
    let [yMin, yMax] = [Math.min(...ys), Math.max(...ys)];
    if (yMax - yMin < 1e-9) {
        yMin -= 0.5;
        yMax += 0.5;
    }
    const pad = {'left': 48, 'right': 10, 'top': 8, 'bottom': 22};
    const plotX = (xValue) => pad.left + (xValue - xMin) / Math.max(1, xMax - xMin) * (width - pad.left - pad.right);
    const plotY = (yValue) => pad.top + (1 - (transform(yValue) - yMin) / (yMax - yMin)) * (height - pad.top - pad.bottom);

    // Grid and labels
    context.strokeStyle = rule;
    context.fillStyle = muted;
    context.lineWidth = 1;
    context.font = '10px ui-monospace, Menlo, monospace';
    for (let tick = 0; tick <= 4; tick++) {
        const value = yMin + (yMax - yMin) * tick / 4;
        const yPixel = pad.top + (1 - tick / 4) * (height - pad.top - pad.bottom);
        context.beginPath();
        context.moveTo(pad.left, yPixel);
        context.lineTo(width - pad.right, yPixel);
        context.stroke();
        const label = logScale ? 10 ** value : value;
        context.fillText(Math.abs(label) >= 100 ? label.toFixed(0) : label.toPrecision(3), 4, yPixel + 3);
    }
    context.fillText(`step ${Math.round(xMin).toLocaleString()}`, pad.left, height - 6);
    const endLabel = Math.round(xMax).toLocaleString();
    context.fillText(endLabel, width - pad.right - context.measureText(endLabel).width, height - 6);

    // Series
    for (const item of series) {
        const points = item.points.filter(([, yValue]) => Number.isFinite(yValue) && (!logScale || yValue > 0));
        context.strokeStyle = item.color;
        context.lineWidth = 1.6;
        context.beginPath();
        points.forEach(([xValue, yValue], index) => {
            if (index === 0) {
                context.moveTo(plotX(xValue), plotY(yValue));
            } else {
                context.lineTo(plotX(xValue), plotY(yValue));
            }
        });
        context.stroke();
        if (points.length !== 0) {
            const [xLast, yLast] = points[points.length - 1];
            context.fillStyle = item.color;
            context.beginPath();
            context.arc(plotX(xLast), plotY(yLast), 2.5, 0, 2 * Math.PI);
            context.fill();
        }
    }
}


// Render the selected run's charts - only when its data changed, unless forced
let chartsKey = null;
function renderCharts(force = false) {
    if (currentTab !== 'training') {
        return;
    }
    const run = runs.get(selectedRun);
    const dataKey = run ? `${run.name} ${run.modified} ${(run.entries.train || []).length} ${(run.entries.val || []).length}` : '';
    if (!force && dataKey === chartsKey) {
        return;
    }
    chartsKey = dataKey;
    if (!run) {
        chartsTitle.textContent = 'Curves';
        chartsElement.replaceChildren(make('p', 'empty', 'Select a run.'));
        return;
    }
    const continues = parentRun(run);
    chartsTitle.textContent = continues ? `Curves - ${run.name} (continuing ${continues.run.name}, faded)` : `Curves - ${run.name}`;
    const styles = getComputedStyle(document.documentElement);
    const colors = ['--accent', '--series-2', '--series-3', '--series-4'].map((name) => styles.getPropertyValue(name));
    const train = run.entries.train || [];
    const val = run.entries.val || [];
    const parent = parentRun(run);
    const offset = parent ? parent.offset : 0;
    const series = (entries, key) => entries.filter((entry) => Number.isFinite(entry[key])).
        map((entry) => [entry.step + offset, entry[key]]);
    const parentSeries = (kind, key) => {
        if (!parent) {
            return [];
        }
        const entries = (parent.run.entries[kind] || []).filter((entry) => Number.isFinite(entry[key]) && entry.step <= offset);
        return entries.map((entry) => [entry.step, entry[key]]);
    };
    const faded = (color) => `${color.trim()}55`;
    // Each series: [label, kind (train or val), key, smoothed, color]
    const charts = [
        {'title': 'Loss (total)', 'logScale': true, 'series': [
            ['train (smoothed)', 'train', 'total', true, colors[0]], ['val', 'val', 'total', false, colors[1]]
        ]},
        {'title': 'Spectral loss', 'series': [
            ['train (smoothed)', 'train', 'spectral', true, colors[0]], ['val', 'val', 'spectral', false, colors[1]]
        ]},
        {'title': 'Durations', 'logScale': true, 'series': [
            ['phones (train)', 'train', 'durPhone', true, colors[0]], ['phones (val)', 'val', 'durPhone', false, colors[1]],
            ['pauses (val)', 'val', 'durPaused', false, colors[2]]
        ]},
        {'title': 'Pitch error (val, cents)', 'series': [['F0 error', 'val', 'f0Cents', false, colors[3]]]},
        {'title': 'Seconds per step (stalls show as spikes)', 'logScale': true, 'series': [
            ['s/step', 'train', 'secondsPerStep', false, colors[2]]
        ]},
        {'title': 'Learning rate', 'series': [['lr', 'train', 'lr', false, colors[0]]]}
    ];

    // Any other numeric training or validation keys become extra charts (e.g. adversarial losses)
    const known = new Set(['step', 'lr', 'secondsPerStep', 'total', 'spectral', 'durPhone', 'durPause', 'durPaused', 'f0Cents', 'guide']);
    const extraKeys = [...new Set([...train, ...val].flatMap((entry) => Object.keys(entry)))].filter((key) => !known.has(key));
    for (const key of extraKeys) {
        charts.push({'title': key, 'logScale': false, 'series': [
            ['train (smoothed)', 'train', key, true, colors[0]], ['val', 'val', key, false, colors[1]]
        ]});
    }

    // The series' points - a continued run's parent drawn behind it, faded, up to the step the run started from
    const entriesOf = {train, val};
    for (const chart of charts) {
        chart.series = chart.series.flatMap(([label, kind, key, smoothed, color]) => {
            const own = series(entriesOf[kind], key);
            const ownSeries = {label, 'points': smoothed ? smoothSeries(own, 0.8) : own, color};
            const before = parentSeries(kind, key);
            if (before.length === 0) {
                return [ownSeries];
            }
            const parentPoints = smoothed ? smoothSeries(before, 0.8) : before;
            return [{'label': `${parent.run.name} ${label}`, 'points': parentPoints, 'color': faded(color)}, ownSeries];
        });
    }
    chartsElement.replaceChildren(...charts.map((chart) => {
        const wrap = make('div', 'chart');
        const canvas = make('canvas');
        const legend = make('div', 'legend');
        for (const item of chart.series) {
            const swatch = make('span', '', item.label);
            swatch.style.setProperty('--swatch', item.color);
            legend.append(swatch);
        }
        wrap.append(make('h3', '', chart.title), canvas, legend);
        requestAnimationFrame(() => drawChart(canvas, chart.series, {'logScale': chart.logScale}));
        return wrap;
    }));
}


// Find the files under build/nn/ (not in runs/), two levels deep - [{path, url}]. The subdirectories are listed every
// 10 seconds (their files are fetched every second).
const listingCache = new Map();
async function findFiles() {
    const found = [];
    const top = await listDirectory(buildUrl);
    for (const entry of top) {
        if (entry.isDirectory && entry.name !== 'runs') {
            let listing = listingCache.get(entry.name);
            if (!listing || Date.now() - listing.time > 10000) {
                listing = {'time': Date.now(), 'entries': await listDirectory(new URL(`${encodeURIComponent(entry.name)}/`, buildUrl))};
                listingCache.set(entry.name, listing);
            }
            const nested = listing.entries;
            for (const item of nested.filter((other) => !other.isDirectory)) {
                const url = new URL(`${encodeURIComponent(entry.name)}/${encodeURIComponent(item.name)}`, buildUrl);
                found.push({'path': `${entry.name}/${item.name}`, url});
            }
        } else if (!entry.isDirectory) {
            found.push({'path': entry.name, 'url': new URL(encodeURIComponent(entry.name), buildUrl)});
        }
    }
    return found;
}


// A log's latest progress - the last of its final lines reporting a percentage or an n/m count - {fraction, line} or null
function parseProgress(lines) {
    for (const line of lines.slice(-5).reverse()) {
        const percents = [...line.matchAll(/(\d+(?:\.\d+)?)\s*%/g)];
        if (percents.length !== 0) {
            return {'fraction': Number(percents[percents.length - 1][1]) / 100, 'line': line.trim()};
        }
        const counts = [...line.matchAll(/(\d[\d,]*)\s*\/\s*(\d[\d,]*)/g)].map((match) => [
            Number(match[1].replace(/,/g, '')), Number(match[2].replace(/,/g, ''))
        ]).filter(([done, total]) => total > 0 && done <= total);
        if (counts.length !== 0) {
            const [done, total] = counts[counts.length - 1];
            return {'fraction': done / total, 'line': line.trim()};
        }
    }
    return null;
}


// The time left (s) for a log's progress, from its readings over the last few minutes
function progressTimeLeft(path, fraction) {
    const now = Date.now();
    const recent = (reading) => now - reading.time < 5 * 60 * 1000 && reading.fraction <= fraction;
    const readings = (progressHistory.get(path) || []).filter(recent);
    readings.push({'time': now, fraction});
    progressHistory.set(path, readings);
    const [first] = readings;
    const rate = (fraction - first.fraction) / ((now - first.time) / 1000);
    return rate > 0 ? (1 - fraction) / rate : NaN;
}


// The pipeline (pipeline.json) - its phases' steps, each tracked by a log, the newest training run, an output file (or
// a file containing a text), and its process
let pipeline = null;
let pipelineLoaded = 0;
async function loadPipeline() {
    if (Date.now() - pipelineLoaded < 30000) {
        return;
    }
    pipelineLoaded = Date.now();
    const result = await fetchText(new URL('pipeline.json', import.meta.url));
    try {
        pipeline = result === null ? null : JSON.parse(result.text);
    } catch {
        pipeline = null;
    }
}


// Whether a file exists - checked every 10 seconds
const existsCache = new Map();
async function fileExists(path) {
    const cached = existsCache.get(path);
    if (cached && Date.now() - cached.time < 10000) {
        return cached.exists;
    }
    let exists = false;
    try {
        exists = (await fetch(new URL(path, rootUrl), {'method': 'HEAD', 'cache': 'no-store'})).ok;
    } catch {
        // Unreachable - not there
    }
    existsCache.set(path, {'time': Date.now(), exists});
    return exists;
}


// The phase each unplanned log in an unknown folder was assigned to - the phase running when it first appeared,
// remembered per browser
const assignedKey = 'training-dashboard-assigned-v1';
let assigned = {};
try {
    assigned = JSON.parse(localStorage.getItem(assignedKey)) || {};
} catch {
    assigned = {};
}
function saveAssigned() {
    try {
        localStorage.setItem(assignedKey, JSON.stringify(assigned));
    } catch {
        // Unavailable storage - assignments last this page load
    }
}


// A log step's state from its fetched log - {status: waiting|running|done|failed, fraction, line, timeLeft}. A step with
// doneWhen (and failWhen) text is running until a log line contains it, however long the log is quiet (a long stage)
function logState(path, log, {doneWhen = null, failWhen = null} = {}) {
    if (!log) {
        return {'status': 'waiting', 'fraction': 0, 'line': 'not started'};
    }
    const {result, lines} = log;
    const lastLine = (lines[lines.length - 1] || '').trim();
    if (lines.some((line) => line.startsWith('Traceback') || (failWhen !== null && line.includes(failWhen)))) {
        return {'status': 'failed', 'fraction': 0, 'line': lastLine};
    }
    if (doneWhen !== null) {
        const done = lines.some((line) => line.includes(doneWhen));
        return {'status': done ? 'done' : 'running', 'fraction': done ? 1 : null, 'line': lastLine};
    }
    // Done: quiet for a few minutes, or briefly quiet after a complete count or a summary line following the progress lines
    const quiet = Date.now() - result.modified;
    const lastProgress = parseProgress(lines.slice(-1));
    const finished = (lastProgress !== null && lastProgress.fraction >= 1) || (lastProgress === null && parseProgress(lines) !== null);
    if (quiet < stalledMs && !(finished && quiet >= 20000)) {
        const progress = parseProgress(lines);
        if (progress === null) {
            return {'status': 'running', 'fraction': null, 'line': lastLine};
        }
        const fraction = Math.min(progress.fraction, 0.99);
        return {'status': 'running', fraction, 'line': progress.line, 'timeLeft': progressTimeLeft(path, progress.fraction)};
    }
    return {'status': 'done', 'fraction': 1, 'line': lastLine};
}


// A training run's state
function runState(run) {
    const train = run.entries.train || [];
    const last = train[train.length - 1];
    const fraction = run.info.targetSteps ? Math.min(1, (last?.step ?? 0) / run.info.targetSteps) : null;
    const status = runStatus(run);
    const target = run.info.targetSteps ? ` of ${run.info.targetSteps.toLocaleString()}` : '';
    const line = `step ${last ? last.step.toLocaleString() : 0}${target}`;
    if (status === 'failed') {
        return {status, fraction, line};
    }
    if (status === 'finished' || status === 'stopped') {
        return {'status': 'done', 'fraction': 1, line};
    }
    const recent = train.slice(-20).map((entry) => entry.secondsPerStep).filter(Number.isFinite);
    const timeLeft = last && run.info.targetSteps ? (run.info.targetSteps - last.step) * median(recent) : NaN;
    return {'status': 'running', fraction, 'line': status === 'stalled' ? `${line} - no update for a few minutes` : line, timeLeft};
}


// The monitor's activity (tools/dashboard/monitor.js writes build/nn/activity.json) - {updated, steps, agents} or null
async function loadActivity() {
    const result = await fetchText(new URL('activity.json', buildUrl));
    try {
        return result === null ? null : JSON.parse(result.text);
    } catch {
        return null;
    }
}


// A process's state - running (with its workers, CPU, and time), or ended
function processState(entry) {
    const workers = entry.workers ? `${entry.workers} workers · ` : '';
    if (entry.running) {
        const line = `${workers}${entry.cpu}% CPU · running ${formatDuration((Date.now() - entry.started) / 1000)}`;
        return {'status': 'running', 'fraction': null, line};
    }
    const ran = formatDuration((entry.lastSeen - entry.started) / 1000);
    const line = `ran ${ran}, ended ${formatDuration((Date.now() - entry.lastSeen) / 1000)} ago`;
    return {'status': 'done', 'fraction': 1, line};
}


// An agent's state - its latest action and when
function agentState(agent) {
    const ago = formatDuration((Date.now() - (agent.actionTime || agent.updated)) / 1000);
    const line = agent.action ? `${agent.action} · ${ago} ago` : 'starting';
    return agent.status === 'done' ? {'status': 'done', 'fraction': 1, line} : {'status': 'running', 'fraction': null, line};
}


// The pipeline's rows - each phase's agents, its planned steps, then the unplanned work that appeared during it (logs,
// runs, and processes since the pipeline started), each with its state. Rows are never removed.
async function pipelineRows(logs, activity) {
    const since = Date.parse(pipeline.since);
    const plannedSteps = pipeline.phases.flatMap((phase) => phase.steps);
    const logsByPath = new Map(logs.map((log) => [`build/nn/${log.file.path}`, log]));
    const newRuns = [...runs.values()].filter((run) => run.modified >= since).sort((first, second) => first.modified - second.modified);
    // The run step tracks the newest run whose name starts with its prefix (true = any run)
    const runStep = plannedSteps.find((step) => step.run);
    const runPrefix = typeof runStep?.run === 'string' ? runStep.run : '';
    const newestRun = newRuns.filter((run) => run.name.startsWith(runPrefix)).pop();
    const planned = new Set(plannedSteps.filter((step) => step.log).map((step) => step.log));
    const processes = (activity?.steps || []).filter((entry) => entry.started >= since);
    const latestProcess = (prefix) => processes.filter((entry) => entry.label.startsWith(prefix)).pop();
    const notStarted = {'status': 'waiting', 'fraction': 0, 'line': 'not started'};
    const phases = await Promise.all(pipeline.phases.map(async (phase) => ({
        phase, 'rows': await Promise.all(phase.steps.map(async (step) => {
            let state;
            if (step.log) {
                state = logState(step.log, logsByPath.get(step.log), step);
            } else if (step.run) {
                state = newestRun ? runState(newestRun) : notStarted;
                if (newestRun) {
                    state.line = `${newestRun.name} - ${state.line}`;
                }
            } else if (step.contains) {
                // Done when the file contains the text
                const result = await fetchText(new URL(step.file, rootUrl));
                const contains = result !== null && result.text.includes(step.contains);
                state = contains ? {'status': 'done', 'fraction': 1, 'line': step.file} : notStarted;
            } else {
                state = await fileExists(step.file) ? {'status': 'done', 'fraction': 1, 'line': step.file} : notStarted;
            }

            // The step's process - running shows (if the log or run doesn't), and ended without output counts as done
            const entry = step.process ? latestProcess(step.process) : undefined;
            if (entry && entry.running && state.status !== 'running') {
                state = processState(entry);
            } else if (entry && !entry.running && state.status === 'waiting') {
                state = processState(entry);
            }
            return {'id': step.title, 'title': step.title, 'kind': 'step', 'weight': step.weight, state};
        }))
    })));

    // The phase running now - the first not done (for assigning unplanned work)
    const currentPhase = Math.max(0, phases.findIndex(({rows}) => !rows.every((row) => row.state.status === 'done')));
    const assign = (id, phaseIndex) => {
        if (phaseIndex !== -1) {
            return phaseIndex;
        }
        if (!(id in assigned)) {
            assigned[id] = currentPhase;
            saveAssigned();
        }
        return assigned[id];
    };
    const addRow = (phaseIndex, row) => {
        phases[Math.min(phaseIndex, phases.length - 1)].rows.push({...row, 'weight': 0, 'unplanned': true});
    };

    // Agents lead their phase
    for (const agent of activity?.agents || []) {
        const phaseIndex = assign(`agent:${agent.id}`, pipeline.phases.findIndex((phase) => phase.title === agent.phase));
        phases[Math.min(phaseIndex, phases.length - 1)].rows.unshift({
            'id': `agent:${agent.id}`, 'title': `agent: ${agent.label}`, 'kind': 'agent', 'weight': 0, 'unplanned': true,
            'state': agentState(agent)
        });
    }

    // Unplanned logs join the phase whose planned logs share their folder
    const folder = (path) => path.slice(0, path.lastIndexOf('/') + 1);
    const phaseOfFolder = (path) => pipeline.phases.findIndex((phase) => phase.steps.some((step) => (
        step.log && folder(step.log) === folder(path)
    )));
    for (const log of logs.filter((item) => item.result.modified >= since && !planned.has(`build/nn/${item.file.path}`)).
        sort((first, second) => first.result.modified - second.result.modified)) {
        const id = `log:${log.file.path}`;
        const row = {id, 'title': log.file.path, 'kind': 'log', 'state': logState(log.file.path, log)};
        addRow(assign(id, phaseOfFolder(`build/nn/${log.file.path}`)), row);
    }

    // Other runs join the phase whose run prefixes match (else the training phase)
    const trainPhase = pipeline.phases.findIndex((phase) => phase.steps.some((step) => step.run));
    for (const run of newRuns.filter((item) => item !== newestRun)) {
        const phaseIndex = pipeline.phases.findIndex((phase) => (phase.runs || []).some((prefix) => run.name.startsWith(prefix)));
        const row = {'id': `run:${run.name}`, 'title': `run ${run.name}`, 'kind': 'run', 'state': runState(run)};
        addRow(phaseIndex === -1 ? trainPhase : phaseIndex, row);
    }

    // Unplanned processes join the phase whose process prefixes match
    const plannedProcess = (entry) => plannedSteps.some((step) => step.process && entry.label.startsWith(step.process));
    for (const entry of processes.filter((item) => !plannedProcess(item))) {
        const id = `process:${entry.label} ${Math.round(entry.started / 10000)}`;
        const phaseIndex = pipeline.phases.findIndex((phase) => (phase.processes || []).some((prefix) => entry.label.startsWith(prefix)));
        addRow(assign(id, phaseIndex), {id, 'title': entry.label, 'kind': 'process', 'state': processState(entry)});
    }
    return phases;
}


// The phases open in the pipeline (by title), remembered per browser
const openKey = 'training-dashboard-open-v1';
let openPhases = new Set();
try {
    openPhases = new Set(JSON.parse(localStorage.getItem(openKey)) || []);
} catch {
    openPhases = new Set();
}
function saveOpen() {
    try {
        localStorage.setItem(openKey, JSON.stringify([...openPhases]));
    } catch {
        // Unavailable storage - remembered this page load
    }
}


// A row's status label - its percent and time left when running
function statusLabel({status, fraction, timeLeft}) {
    if (status !== 'running') {
        return status;
    }
    let label = fraction === null ? 'running' : `${Math.round(100 * fraction)}%`;
    if (Number.isFinite(timeLeft)) {
        label += ` · ${formatDuration(timeLeft)} left`;
    }
    return label;
}


// Render the pipeline - each phase a group (its progress, what's happening now, and how many steps are done) that opens
// to its rows - agents, steps, and other work - updated in place; and the overall progress (by the phases' and steps'
// weights, roughly their expected time)
const stepIcons = {'waiting': '○', 'running': '●', 'done': '✓', 'failed': '✕'};
let pipelineKey = null;
let rowElements = new Map();
let phaseElements = [];
async function renderPipeline(logs) {
    await loadPipeline();
    pipelineSection.hidden = pipeline === null;
    if (pipeline === null) {
        return;
    }
    const activity = await loadActivity();
    const phases = await pipelineRows(logs, activity);

    // Build the groups when the set of rows changes (a step appeared), otherwise update them in place
    const key = phases.map(({rows}) => rows.map((row) => row.id).join('|')).join('||');
    if (key !== pipelineKey) {
        pipelineKey = key;
        rowElements = new Map();
        phaseElements = phases.map(({phase, rows}) => {
            const element = make('details', 'phase');
            element.open = openPhases.has(phase.title);
            element.addEventListener('toggle', () => {
                if (element.open) {
                    openPhases.add(phase.title);
                } else {
                    openPhases.delete(phase.title);
                }
                saveOpen();
            });
            const summary = make('summary');
            const head = make('div', 'pipeline-head');
            const facts = make('span', 'facts');
            head.append(make('h3', '', phase.title), facts);
            const bar = make('div', 'bar');
            const fill = make('div');
            bar.append(fill);
            const now = make('div', 'phase-now');
            const count = make('div', 'phase-count');
            summary.append(head, bar, now, count);
            const list = make('ul', 'steps');
            for (const row of rows) {
                const item = make('li', 'step');
                const icon = make('span', 'icon');
                const percent = make('span', 'facts');
                const rowBar = make('div', 'bar');
                const rowFill = make('div');
                rowBar.append(rowFill);
                const line = make('div', 'progress-line');
                item.append(icon, make('span', row.unplanned ? `unplanned kind-${row.kind}` : '', row.title), percent, rowBar, line);
                list.append(item);
                rowElements.set(row.id, {item, icon, percent, 'fill': rowFill, line});
            }
            element.append(summary, make('div', 'phase-detail', phase.detail), list);
            return {element, facts, fill, now, count};
        });
        phasesElement.replaceChildren(...phaseElements.map(({element}) => element));
    }

    // Update the rows and the groups
    let overall = 0;
    const totalWeight = phases.reduce((sum, {phase}) => sum + phase.weight, 0);
    phases.forEach(({phase, rows}, index) => {
        for (const row of rows) {
            const {item, icon, percent, fill, line} = rowElements.get(row.id);
            const {status, fraction} = row.state;
            item.className = `step ${status}`;
            icon.textContent = stepIcons[status];
            percent.textContent = statusLabel(row.state);
            fill.style.width = `${(100 * (fraction ?? 0)).toFixed(1)}%`;
            fill.classList.toggle('indeterminate', status === 'running' && fraction === null);
            line.textContent = row.state.line || '';
            line.title = row.state.line || '';
        }
        const plannedRows = rows.filter((row) => !row.unplanned);
        const stepWeight = plannedRows.reduce((sum, row) => sum + row.weight, 0);
        const done = plannedRows.reduce((sum, row) => sum + row.weight * (row.state.fraction ?? 0.5), 0) / stepWeight;
        overall += phase.weight * done / totalWeight;
        const work = rows.filter((row) => row.kind !== 'agent');
        let status = 'waiting';
        if (rows.every((row) => row.state.status === 'done')) {
            status = 'done';
        } else if (rows.some((row) => row.state.status !== 'waiting')) {
            status = 'running';
        }

        // What's happening now - the running work, else the running agent's latest action
        const running = work.filter((row) => row.state.status === 'running' || row.state.status === 'failed');
        const agent = rows.find((row) => row.kind === 'agent' && row.state.status === 'running');
        let now = '';
        if (running.length !== 0) {
            now = running.map((row) => `${stepIcons[row.state.status]} ${row.title} · ${statusLabel(row.state)}`).join('   ');
        } else if (agent) {
            now = `${agent.title} - ${agent.state.line}`;
        } else if (status === 'done') {
            now = 'complete';
        }
        const group = phaseElements[index];
        group.element.className = `phase ${status}`;
        group.facts.textContent = status === 'waiting' ? 'waiting' : `${Math.round(100 * done)}%`;
        group.fill.style.width = `${(100 * done).toFixed(1)}%`;
        group.now.textContent = now;
        group.now.title = now;
        const doneCount = work.filter((row) => row.state.status === 'done').length;
        group.count.textContent = `${doneCount} of ${work.length} steps done`;
    });
    pipelineTitle.textContent = pipeline.title;
    pipelineBar.style.width = `${(100 * overall).toFixed(1)}%`;
    const elapsed = formatDuration((Date.now() - Date.parse(pipeline.since)) / 1000);
    const monitor = activity && Date.now() - activity.updated < 10000 ? '' :
        ' · the monitor isn\'t running (node tools/dashboard/monitor.js) - steps without logs aren\'t shown';
    pipelineFacts.textContent = `about ${Math.round(100 * overall)}% (weighted by expected time) · started ${elapsed} ago${monitor}`;
    setTabFact('tabFactProgress', `${Math.round(100 * overall)}%`);
}


// A log's heading - its path and age
function logHeading(file, result) {
    return `${file.path} · updated ${formatDuration((Date.now() - result.modified) / 1000)} ago`;
}


// Fetch the logs under build/nn/ - [{file, result, lines}], the most recently updated first
async function loadLogs(allFiles) {
    const files = allFiles.filter((file) => file.path.endsWith('.log'));
    const logs = (await Promise.all(files.map(async (file) => ({file, 'result': await fetchText(file.url)})))).
        filter((item) => item.result !== null).sort((first, second) => second.result.modified - first.result.modified);
    for (const log of logs) {
        log.lines = log.result.text.replace(/\r/g, '\n').split('\n').filter((line) => line.trim() !== '');
    }
    return logs;
}


// Render the activity logs - the most recently updated first, each one's tail - rebuilt only when they change
let logsKey = null;
let logHeads = [];
function renderLogs(allLogs) {
    const activeLogs = allLogs.filter(({result}) => Date.now() - result.modified < stalledMs).length;
    setTabFact('tabFactLogs', activeLogs === 0 ? '' : `${activeLogs} active`);
    const logs = allLogs.slice(0, 8);
    const key = logs.map(({file, result}) => `${file.path} ${result.modified} ${result.text.length}`).join('\n');
    if (key === logsKey && logs.length !== 0) {
        logs.forEach(({file, result}, index) => {
            logHeads[index].textContent = logHeading(file, result);
        });
        return;
    }
    logsKey = key;
    logHeads = [];
    if (logs.length === 0) {
        logsElement.replaceChildren(make('p', 'empty', 'No logs yet.'));
        return;
    }
    logsElement.replaceChildren(...logs.map(({file, result, lines}) => {
        const wrap = make('div', 'log');
        const head = make('h3', '', logHeading(file, result));
        logHeads.push(head);
        wrap.append(head);
        const pre = make('pre', '', lines.slice(-14).join('\n'));
        wrap.append(pre);
        requestAnimationFrame(() => {
            pre.scrollTop = pre.scrollHeight;
        });
        return wrap;
    }));
}


// The mean of the scores of a sex's samples, formatted
function meanScore(samples, sex) {
    const scores = samples.filter((sample) => sample.sex === sex && Number.isFinite(sample.mos)).map((sample) => sample.mos);
    return scores.length === 0 ? '-' : (scores.reduce((sum, score) => sum + score, 0) / scores.length).toFixed(2);
}


// Render the checkpoint samples (build/nn/samples-progress/index.json, written by the monitor) - each checkpoint's
// voices, newest first, then the reference app voices - rebuilt when the index changes, reusing each sample's player so
// playback continues
let checkpointsKey = null;
const voiceTiles = new Map();
async function renderCheckpointSamples() {
    const result = await fetchText(new URL('samples-progress/index.json', buildUrl));
    if (result === null || result.text === checkpointsKey) {
        checkpointSection.hidden = result === null && checkpointsKey === null;
        return;
    }
    checkpointsKey = result.text;
    let index;
    try {
        index = JSON.parse(result.text);
    } catch {
        return;
    }
    const tile = (sample) => {
        if (!voiceTiles.has(sample.file)) {
            const element = make('div', 'voice');
            const head = make('div', 'voice-head');
            head.append(make('span', '', sample.label), make('span', 'mos', Number.isFinite(sample.mos) ? sample.mos.toFixed(2) : '-'));
            const audio = make('audio');
            audio.controls = true;
            audio.preload = 'none';
            audio.src = new URL(`samples-progress/${encodeURIComponent(sample.file)}`, buildUrl);
            element.append(head, audio);
            voiceTiles.set(sample.file, element);
        }
        return voiceTiles.get(sample.file);
    };
    const block = (title, facts, samples) => {
        const element = make('div', 'checkpoint');
        const head = make('div', 'checkpoint-head');
        head.append(make('h3', '', title), make('span', 'facts', facts));
        const voices = make('div', 'voices');
        voices.append(...samples.map(tile));
        element.append(head, voices);
        return element;
    };
    const checkpoints = [...index.checkpoints].sort((first, second) => second.rendered - first.rendered);
    checkpointSteps = new Map();
    for (const checkpoint of checkpoints) {
        checkpointSteps.set(checkpoint.run, [...checkpointSteps.get(checkpoint.run) || [], checkpoint.step]);
    }
    if (checkpoints.length !== 0) {
        const [newest] = checkpoints;
        newestCheckpoint = `${newest.run} ${newest.step}`;
        if (currentTab === 'listen') {
            markHeard();
        } else {
            document.getElementById('tabDotListen').hidden = storageGet(heardKey) === newestCheckpoint;
        }
    }
    const blocks = checkpoints.map((checkpoint) => block(
        `${checkpoint.run} · step ${checkpoint.step.toLocaleString()}`,
        `female ${meanScore(checkpoint.samples, 'female')} · male ${meanScore(checkpoint.samples, 'male')} · ` +
            `rendered ${new Date(checkpoint.rendered).toLocaleTimeString()}`,
        checkpoint.samples
    ));
    if (index.reference.length !== 0) {
        const facts = `female ${meanScore(index.reference, 'female')} · male ${meanScore(index.reference, 'male')}`;
        blocks.push(block('Today\'s app voices (reference)', facts, index.reference));
    }
    if (checkpoints.length === 0) {
        blocks.unshift(make('p', 'empty', 'No checkpoints yet - samples appear as each one is saved.'));
    }
    checkpointsElement.replaceChildren(...blocks);
    checkpointSection.hidden = false;
}


// Render the time until the next checkpoint's samples - the live run's next checkpoint (every pipeline.json
// samples.checkpointEvery steps, or the interval between its rendered checkpoints), estimated from its recent seconds
// per step, plus about 1.5 minutes for the monitor to notice, render, and score it
let checkpointSteps = new Map();
const renderSeconds = 90;
function renderNextCheckpoint() {
    const section = document.getElementById('nextSection');
    // Only the runs whose checkpoints get samples (pipeline.json samples.runs), newest first by first-seen order
    const prefix = pipeline?.samples?.runs ?? '';
    const [live] = [...runs.values()].filter((run) => run.name.startsWith(prefix) && runStatus(run) === 'live').
        sort((first, second) => (runOrder[second.name] ?? 0) - (runOrder[first.name] ?? 0));
    const train = live ? live.entries.train || [] : [];
    const last = train[train.length - 1];
    if (!live || !last || !pipeline?.samples) {
        section.hidden = true;
        setTabFact('tabFactListen', newestCheckpoint === null ? '' : `step ${Number(newestCheckpoint.split(' ')[1]).toLocaleString()}`);
        return;
    }
    const rendered = (checkpointSteps.get(live.name) || []).sort((first, second) => first - second);
    const gaps = rendered.slice(1).map((step, index) => step - rendered[index]).filter((gap) => gap > 0);
    const every = gaps.length !== 0 ? Math.min(...gaps) : pipeline.samples.checkpointEvery || 5000;
    let next = (Math.floor(last.step / every) + 1) * every;
    if (live.info.targetSteps) {
        next = Math.min(next, live.info.targetSteps);
    }
    const previous = next - every;
    const secondsPerStep = median(train.slice(-20).map((entry) => entry.secondsPerStep).filter(Number.isFinite));
    const seconds = (next - last.step) * secondsPerStep + renderSeconds;
    const when = new Date(Date.now() + seconds * 1000).toLocaleTimeString([], {'hour': 'numeric', 'minute': '2-digit'});
    section.hidden = false;
    const whenText = Number.isFinite(seconds) ? `in about ${formatDuration(seconds)} · around ${when}` : '';
    document.getElementById('nextWhen').textContent = whenText;
    const fraction = Math.max(0, Math.min(1, (last.step - previous) / every));
    document.getElementById('nextBar').style.width = `${(100 * fraction).toFixed(1)}%`;
    document.getElementById('nextLine').textContent = `${live.name} step ${next.toLocaleString()} - now at step ` +
        `${last.step.toLocaleString()}, ${secondsPerStep.toFixed(2)} s/step; the samples appear a minute or two after it's saved`;
    setTabFact('tabFactListen', Number.isFinite(seconds) ? `next ${formatDuration(seconds)}` : '');
}


// Render the samples - WAV files under build/nn/ (but the checkpoint samples, shown above)
function renderSamples(allFiles) {
    const files = allFiles.filter((file) => file.path.endsWith('.wav') && !file.path.startsWith('samples-progress/'));
    if (files.length === 0) {
        samplesElement.replaceChildren(make('p', 'empty', 'No samples yet.'));
        return;
    }
    const existing = new Set([...samplesElement.querySelectorAll('audio')].map((audio) => audio.dataset.path));
    if (files.length === existing.size && files.every((file) => existing.has(file.path))) {
        return;
    }
    samplesElement.replaceChildren(...files.slice(-60).map((file) => {
        const wrap = make('div', 'sample');
        const audio = make('audio');
        audio.controls = true;
        audio.preload = 'none';
        audio.src = file.url;
        audio.dataset.path = file.path;
        wrap.append(make('span', '', file.path), audio);
        return wrap;
    }));
}


// Tabs - the selected tab is in the URL's hash (and remembered per browser); the Listen tab marks new checkpoints until
// it's viewed
const tabNames = ['progress', 'listen', 'training', 'logs'];
const tabKey = 'training-dashboard-tab-v1';
const heardKey = 'training-dashboard-heard-v1';
let currentTab = 'progress';
function storageGet(key) {
    try {
        return localStorage.getItem(key);
    } catch {
        return null;
    }
}
function storageSet(key, value) {
    try {
        localStorage.setItem(key, value);
    } catch {
        // Unavailable storage - remembered this page load
    }
}
function selectTab(name, {updateHash = true} = {}) {
    currentTab = tabNames.includes(name) ? name : 'progress';
    for (const tab of tabNames) {
        const button = document.getElementById(`tab-${tab}`);
        button.setAttribute('aria-selected', String(tab === currentTab));
        button.tabIndex = tab === currentTab ? 0 : -1;
        document.getElementById(`panel-${tab}`).hidden = tab !== currentTab;
    }
    storageSet(tabKey, currentTab);
    if (updateHash && window.location.hash !== `#${currentTab}`) {
        window.history.replaceState(null, '', `#${currentTab}`);
    }
    if (currentTab === 'training') {
        // The charts draw at their visible size
        renderCharts(true);
    }
    if (currentTab === 'listen') {
        markHeard();
    }
}
for (const tab of tabNames) {
    const button = document.getElementById(`tab-${tab}`);
    button.addEventListener('click', () => selectTab(tab));
    button.addEventListener('keydown', (event) => {
        // Arrow keys move between the tabs
        const step = {'ArrowRight': 1, 'ArrowLeft': -1}[event.key];
        if (step) {
            const next = tabNames[(tabNames.indexOf(tab) + step + tabNames.length) % tabNames.length];
            selectTab(next);
            document.getElementById(`tab-${next}`).focus();
        }
    });
}
window.addEventListener('hashchange', () => selectTab(window.location.hash.slice(1), {'updateHash': false}));


// The newest checkpoint's key, and marking it heard (the Listen tab's dot)
let newestCheckpoint = null;
function markHeard() {
    if (newestCheckpoint !== null) {
        storageSet(heardKey, newestCheckpoint);
    }
    document.getElementById('tabDotListen').hidden = true;
}


// Update the tabs' facts - the overall progress, the newest checkpoint, the live run, and the active logs
function setTabFact(id, text) {
    const element = document.getElementById(id);
    if (element.textContent !== text) {
        element.textContent = text;
    }
}


// Reload the page when the dashboard's own files change (checked every 5 seconds), so it never runs stale code
const codeFiles = ['index.html', 'dashboard.js'].map((name) => new URL(`tools/dashboard/${name}`, rootUrl));
let codeVersions = null;
async function checkCode() {
    try {
        const versions = await Promise.all(codeFiles.map(async (url) => (
            (await fetch(url, {'method': 'HEAD', 'cache': 'no-store'})).headers.get('Last-Modified')
        )));
        if (codeVersions === null) {
            codeVersions = versions;
        } else if (versions.some((version, index) => version !== codeVersions[index])) {
            window.location.reload();
        }
    } catch {
        // Unreachable server - check again later
    }
}
checkCode();
setInterval(checkCode, 5000);
selectTab(window.location.hash.slice(1) || storageGet(tabKey) || 'progress');


// Refresh - then again a second later, so refreshes never overlap
async function refresh() {
    try {
        await refreshRuns();
        renderRuns();
        renderCharts();
        const files = await findFiles();
        const logs = await loadLogs(files);
        await renderPipeline(logs);
        renderLogs(logs);
        renderSamples(files);
        await renderCheckpointSamples();
        renderNextCheckpoint();
        updatedElement.textContent = `Updated ${new Date().toLocaleTimeString()} · every second`;
    } finally {
        setTimeout(refresh, refreshMs);
    }
}
refresh();
window.addEventListener('resize', () => renderCharts(true));
