// The training dashboard - reads the neural training runs' logs (build/nn/runs/<run>/log.jsonl and train.log), the
// data pipelines' logs (build/nn/**/*.log), and samples (build/nn/**/*.wav) through python3 -m http.server's directory
// listings, and refreshes every 10 seconds. Read-only: it never touches the training.


// Where the build lives relative to this page, and timing (ms)
const buildUrl = new URL('../../build/nn/', import.meta.url);
const refreshMs = 10000;
const stalledMs = 3 * 60 * 1000;
const stoppedMs = 30 * 60 * 1000;


// Elements
const runsElement = document.getElementById('runs');
const chartsElement = document.getElementById('charts');
const chartsTitle = document.getElementById('chartsTitle');
const logsElement = document.getElementById('logs');
const samplesElement = document.getElementById('samples');
const progressSection = document.getElementById('progressSection');
const progressElement = document.getElementById('progress');
const updatedElement = document.getElementById('updated');


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


// Fetch a text file, reusing the cached text when its Last-Modified is unchanged - {text, modified} or null
async function fetchText(url) {
    try {
        const cached = fileCache.get(url);
        const headers = cached ? {'If-Modified-Since': cached.lastModified} : {};
        const response = await fetch(url, {'cache': 'no-store', headers});
        if (response.status === 304 && cached) {
            return cached;
        }
        if (!response.ok) {
            return null;
        }
        const lastModified = response.headers.get('Last-Modified') || '';
        const result = {'text': await response.text(), lastModified, 'modified': lastModified ? Date.parse(lastModified) : Date.now()};
        fileCache.set(url, result);
        return result;
    } catch {
        return null;
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
    const target = text.match(/(\d+) steps from (\d+)/);
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
    if (/Traceback|Error:/.test(text)) {
        info.error = text.split('\n').filter((line) => /Error/.test(line)).pop();
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
    if (selectedRun === null || !runs.has(selectedRun)) {
        // Select the most recently updated run
        selectedRun = [...runs.values()].sort((first, second) => second.modified - first.modified)[0]?.name ?? null;
    }
}


// Render the run cards
function renderRuns() {
    if (runs.size === 0) {
        runsElement.replaceChildren(make('p', 'empty', 'No runs yet - training writes to build/nn/runs/<run>/.'));
        return;
    }
    const sorted = [...runs.values()].sort((first, second) => second.modified - first.modified);
    runsElement.replaceChildren(...sorted.map((run) => {
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
        card.addEventListener('click', () => {
            selectedRun = run.name;
            render();
        });
        return card;
    }));
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


// Render the selected run's charts
function renderCharts() {
    const run = runs.get(selectedRun);
    if (!run) {
        chartsTitle.textContent = 'Curves';
        chartsElement.replaceChildren(make('p', 'empty', 'Select a run.'));
        return;
    }
    chartsTitle.textContent = `Curves - ${run.name}`;
    const styles = getComputedStyle(document.documentElement);
    const colors = ['--accent', '--series-2', '--series-3', '--series-4'].map((name) => styles.getPropertyValue(name));
    const train = run.entries.train || [];
    const val = run.entries.val || [];
    const series = (entries, key) => entries.filter((entry) => Number.isFinite(entry[key])).map((entry) => [entry.step, entry[key]]);
    const charts = [
        {'title': 'Loss (total)', 'logScale': true, 'series': [
            {'label': 'train (smoothed)', 'points': smoothSeries(series(train, 'total'), 0.8), 'color': colors[0]},
            {'label': 'val', 'points': series(val, 'total'), 'color': colors[1]}
        ]},
        {'title': 'Spectral loss', 'series': [
            {'label': 'train (smoothed)', 'points': smoothSeries(series(train, 'spectral'), 0.8), 'color': colors[0]},
            {'label': 'val', 'points': series(val, 'spectral'), 'color': colors[1]}
        ]},
        {'title': 'Durations', 'logScale': true, 'series': [
            {'label': 'phones (train)', 'points': smoothSeries(series(train, 'durPhone'), 0.8), 'color': colors[0]},
            {'label': 'phones (val)', 'points': series(val, 'durPhone'), 'color': colors[1]},
            {'label': 'pauses (val)', 'points': series(val, 'durPaused'), 'color': colors[2]}
        ]},
        {'title': 'Pitch error (val, cents)', 'series': [{'label': 'F0 error', 'points': series(val, 'f0Cents'), 'color': colors[3]}]},
        {'title': 'Seconds per step (stalls show as spikes)', 'logScale': true, 'series': [
            {'label': 's/step', 'points': series(train, 'secondsPerStep'), 'color': colors[2]}
        ]},
        {'title': 'Learning rate', 'series': [{'label': 'lr', 'points': series(train, 'lr'), 'color': colors[0]}]}
    ];

    // Any other numeric training or validation keys become extra charts (e.g. adversarial losses)
    const known = new Set(['step', 'lr', 'secondsPerStep', 'total', 'spectral', 'durPhone', 'durPause', 'durPaused', 'f0Cents', 'guide']);
    const extraKeys = [...new Set([...train, ...val].flatMap((entry) => Object.keys(entry)))].filter((key) => !known.has(key));
    for (const key of extraKeys) {
        charts.push({'title': key, 'logScale': false, 'series': [
            {'label': 'train (smoothed)', 'points': smoothSeries(series(train, key), 0.8), 'color': colors[0]},
            {'label': 'val', 'points': series(val, key), 'color': colors[1]}
        ]});
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


// Find files under build/nn/ (not in runs/) by extension, two levels deep - [{path, url}]
async function findFiles(extension) {
    const found = [];
    const top = await listDirectory(buildUrl);
    for (const entry of top) {
        if (entry.isDirectory && entry.name !== 'runs') {
            const nested = await listDirectory(new URL(`${encodeURIComponent(entry.name)}/`, buildUrl));
            for (const item of nested.filter((other) => !other.isDirectory && other.name.endsWith(extension))) {
                const url = new URL(`${encodeURIComponent(entry.name)}/${encodeURIComponent(item.name)}`, buildUrl);
                found.push({'path': `${entry.name}/${item.name}`, url});
            }
        } else if (!entry.isDirectory && entry.name.endsWith(extension)) {
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


// Render the progress of the active logs (updated in the last few minutes, not yet complete)
function renderProgress(logs) {
    const active = logs.filter(({result}) => Date.now() - result.modified < stalledMs).map(({file, result, lines}) => {
        const progress = parseProgress(lines);
        return progress === null || progress.fraction >= 1 ? null : {file, result, progress};
    }).filter((item) => item !== null);
    progressSection.hidden = active.length === 0;
    progressElement.replaceChildren(...active.map(({file, progress}) => {
        const item = make('div', 'progress-item');
        const head = make('div', 'progress-head');
        const timeLeft = progressTimeLeft(file.path, progress.fraction);
        const estimate = Number.isFinite(timeLeft) ? ` · about ${formatDuration(timeLeft)} left` : '';
        head.append(make('h3', '', file.path), make('span', 'facts', `${(100 * progress.fraction).toFixed(0)}%${estimate}`));
        const bar = make('div', 'bar');
        const fill = make('div');
        fill.style.width = `${(100 * progress.fraction).toFixed(1)}%`;
        bar.append(fill);
        item.append(head, bar, make('div', 'progress-line', progress.line));
        return item;
    }));
}


// Render the activity logs - the most recently updated first, each one's tail
async function renderLogs() {
    const files = await findFiles('.log');
    const allLogs = (await Promise.all(files.map(async (file) => ({file, 'result': await fetchText(file.url)})))).
        filter((item) => item.result !== null).sort((first, second) => second.result.modified - first.result.modified);
    for (const log of allLogs) {
        log.lines = log.result.text.replace(/\r/g, '\n').split('\n').filter((line) => line.trim() !== '');
    }
    renderProgress(allLogs);
    const logs = allLogs.slice(0, 8);
    if (logs.length === 0) {
        logsElement.replaceChildren(make('p', 'empty', 'No logs yet.'));
        return;
    }
    logsElement.replaceChildren(...logs.map(({file, result, lines}) => {
        const wrap = make('div', 'log');
        const age = formatDuration((Date.now() - result.modified) / 1000);
        wrap.append(make('h3', '', `${file.path} · updated ${age} ago`));
        const pre = make('pre', '', lines.slice(-14).join('\n'));
        wrap.append(pre);
        requestAnimationFrame(() => {
            pre.scrollTop = pre.scrollHeight;
        });
        return wrap;
    }));
}


// Render the samples - WAV files under build/nn/
async function renderSamples() {
    const files = await findFiles('.wav');
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


// Render everything
function render() {
    renderRuns();
    renderCharts();
}


// Refresh periodically
async function refresh() {
    await refreshRuns();
    render();
    await Promise.all([renderLogs(), renderSamples()]);
    updatedElement.textContent = `Updated ${new Date().toLocaleTimeString()} · every ${refreshMs / 1000} s`;
}
refresh();
setInterval(refresh, refreshMs);
window.addEventListener('resize', () => renderCharts());
