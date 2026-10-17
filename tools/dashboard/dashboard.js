// The training dashboard - reads make train's bookkeeping (the TRAIN_DIR of pipeline.json's train: stages.jsonl, the
// stamps, util.log, logs/, runs/, samples/; tools/dashboard/train.js models it), the neural training runs' logs
// (build/nn/runs/<run>/log.jsonl and train.log), the data pipelines' logs (build/nn/**/*.log), and samples
// (build/nn/**/*.wav) through python3 -m http.server's directory listings, and refreshes every second. Read-only: it
// never touches the training.


// Where the repository and the build live relative to this page, and timing (ms)
const rootUrl = new URL('../../', import.meta.url);
const buildUrl = new URL('build/nn/', rootUrl);
const refreshMs = 1000;
const stalledMs = 3 * 60 * 1000;
const stoppedMs = 30 * 60 * 1000;


// The make train model (loaded fresh, like this script)
const trainModel = await import(`./train.js?v=${Date.now()}`);


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


// The latest logs (for re-rendering between refreshes)
let lastLogs = [];


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


// A run's card - or, if its log can't be shown, a card saying so (one odd run never blanks the dashboard)
function safeRunCard(run, onSelect = selectRun, selected = selectedRun) {
    try {
        return runCard(run, onSelect, selected);
    } catch (error) {
        const card = make('div', 'run');
        card.append(make('span', 'run-name', run.name), make('div', 'run-facts', `can't show this run: ${error.message}`));
        return card;
    }
}


// A number to three decimals, or '-' if it isn't one (a run's log may lack a field)
function fixed3(value) {
    return Number.isFinite(value) ? value.toFixed(3) : '-';
}


// A run's status - live (recently updated), stalled (quiet for minutes), stopped (quiet for long), finished, failed, or
// waiting (not started - a make train run's later stage)
function runStatus(run) {
    if (run.waiting) {
        return 'waiting';
    }
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
        // A run stopped early by plan (pipeline.json stopAt - for example, a stage launched with a longer schedule, for its
        // learning rate, and stopped part way) counts its progress to the planned stop
        const info = textLog === null ? {} : parseTextLog(textLog.text);
        const stopAt = pipeline?.stopAt?.[entry.name];
        if (stopAt) {
            info.targetSteps = stopAt;
        }
        runs.set(entry.name, {
            'name': entry.name,
            'entries': jsonLog === null ? {'train': [], 'val': []} : parseJsonLog(jsonLog.text),
            info,
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
        grid.replaceChildren(...group.runs.map((run) => safeRunCard(run)));
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
function runCard(run, onSelect, selected) {
    const train = run.entries.train || [];
    const last = train[train.length - 1];
    const status = runStatus(run);
    const card = make('button', 'run');
    card.type = 'button';
    card.setAttribute('aria-pressed', String(run.name === selected));
    const head = make('div', 'run-head');
    head.append(make('span', 'run-name', run.name), make('span', `pill ${status}`, status));
    const bar = make('div', 'bar');
    const fill = make('div');
    const fraction = last && run.info.targetSteps ? Math.min(1, last.step / run.info.targetSteps) : 0;
    fill.style.width = `${(100 * fraction).toFixed(1)}%`;
    bar.append(fill);
    card.addEventListener('click', () => onSelect(run.name));
    if (run.waiting) {
        card.append(head, bar, make('div', 'facts', run.note || 'not started'));
        return card;
    }
    const recent = train.slice(-20).map((entry) => entry.secondsPerStep).filter(Number.isFinite);
    const secondsPerStep = median(recent);
    const remaining = last && run.info.targetSteps ? (run.info.targetSteps - last.step) * secondsPerStep : NaN;
    const timeLeft = status === 'live' ? `${formatDuration(remaining)} left` : '-';
    const val = run.entries.val || [];
    const lastVal = val[val.length - 1];
    const facts = [
        `step ${last ? last.step.toLocaleString() : 0}${run.info.targetSteps ? ` / ${run.info.targetSteps.toLocaleString()}` : ''}`,
        `${Number.isFinite(secondsPerStep) ? secondsPerStep.toFixed(2) : '-'} s/step · ${timeLeft}`,
        `train total ${fixed3(last?.total)}${lastVal ? ` · val ${fixed3(lastVal.spectral ?? lastVal.total)}` : ''}`,
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
    return card;
}


// The run a run continues - {run, offset} (the parent's step it started from), or null
function parentRun(run) {
    const parent = [...runs.values()].find((other) => other.info.continuedAs === run.name);
    return parent ? {'run': parent, 'offset': parent.info.stoppedStep} : null;
}


// A run's reference run, drawn faded on its charts for comparison - pipeline.json's compare, {run name prefix: the
// reference run's name} (for example, a quality run against its same-seed control), or null
function referenceRun(run) {
    const entry = Object.entries(pipeline?.compare || {}).find(([prefix]) => run.name.startsWith(prefix));
    return entry && entry[1] !== run.name ? runs.get(entry[1]) ?? null : null;
}


// What the training measurements mean - each chart's (?) explanation. Losses are lower-is-better. "val" is measured on
// held-out recordings every 1,000 steps; "train (smoothed)" is the training batches' running value. A "cremad." prefix
// is the same measurement on the emotional speech (the CREMA-D actors).
const chartHelp = {
    'total': 'The weighted sum of every training loss below - what the optimizer minimizes. It mixes units, so compare it ' +
        'only within one run; the spectral loss is the cleaner yardstick.',
    'spectral': 'How far the synthesized voice\'s spectrum is from the real recording\'s - multi-resolution STFT and log-mel ' +
        'differences. The main training loss and the best progress gauge, but blind to some of what sounds synthetic: the ' +
        'critic stage raises naturalness (UTMOS) far more than it lowers this.',
    'durations': 'How well the model predicts timing - phone lengths (durPhone), whether a pause happens at a word or phrase ' +
        'boundary (durPause), and how long the pauses are (durPaused). Errors are in log-duration units.',
    'f0Cents': 'Pitch error on held-out recordings - the average difference between the voice\'s pitch (F0) and the ' +
        'recording\'s, in cents (100 cents = one semitone).',
    'secondsPerStep': 'Wall-clock seconds per training step (one batch). Spikes are stalls - the GPU shared with other jobs, ' +
        'or new batch shapes compiling.',
    'lr': 'The learning rate - how big each update is. It warms up, then decays on a cosine schedule.',
    'durPhone': 'Phone-duration error (log units) - how far the predicted length of each speech sound is from the ' +
        'recording\'s.',
    'durPause': 'Pause-placement error - whether the model puts a pause at each word or phrase boundary where the reader ' +
        'did (a classification loss).',
    'durPaused': 'Pause-length error (log units), at the boundaries where the reader paused.',
    'f0': 'Pitch (F0) error in log units on voiced frames - the training loss behind the cents chart.',
    'formant': 'The formant guide - how far the first three formants (F1-F3, the vocal-tract resonances that make vowels) ' +
        'are from Praat\'s measurements of the recording. Weighted heavily early, then faded (guide), so the spectral loss ' +
        'takes over.',
    'guide': 'The formant guide\'s weight - starts high to steer the vocal tract into place, then fades toward its floor.',
    'order': 'A penalty when formants come out of order or closer than 150 Hz apart (F1 below F2 below F3...), which a real ' +
        'vocal tract can\'t do.',
    'smooth': 'A penalty on frame-to-frame jumps in the synthesizer parameters - keeps the voice from buzzing or warbling.',
    'band': 'A penalty when the frication resonances (the hiss of s, sh, f) move above the recording\'s bandwidth limit.',
    'sonorantAf': 'A penalty on hiss (frication) inside vowels and other sonorants, where there shouldn\'t be any - in dB/10 ' +
        'above the allowed level.',
    'eqSmooth': 'A penalty on jagged channel equalizers - each speaker\'s smooth recording-channel coloring.',
    'score': 'The validation score used to pick the best checkpoint: spectral + phone durations + pitch.',
    'b1': 'The median first-formant bandwidth (Hz) on the voice\'s strong frames - narrow (about 80-150 Hz) sounds clear; ' +
        'wide sounds muffled. The critic exists largely to keep these narrow.',
    'b2': 'The median second-formant bandwidth (Hz) on strong frames - lower is sharper.',
    'b3': 'The median third-formant bandwidth (Hz) on strong frames - lower is sharper.',
    'ahRel': 'Aspiration (breathy noise) level relative to voicing, in dB, on strong frames - too high sounds breathy.',
    'afRel': 'Frication (hiss) level relative to voicing, in dB, on strong frames - too high sounds hissy.',
    'critic': 'The critic\'s own loss (stage B) - how well it tells real recordings from synthesized ones. If it falls toward ' +
        'zero, it is winning too easily.',
    'adversarial': 'The voice\'s adversarial loss (stage B) - how well it fools the critic. Pushes toward sharper, more ' +
        'natural-sounding detail than the spectral loss can.',
    'matching': 'Feature matching (stage B) - the voice matching the critic\'s internal features of real recordings; ' +
        'steadies the adversarial training.',
    'corrL2': 'The spectral correction head\'s size penalty (an optional architecture feature).',
    'corrSmooth': 'The spectral correction head\'s smoothness penalty, over time and frequency.',
    'frames': 'Distillation: how far the student\'s synthesizer parameters are from the teacher\'s, per frame.',
    'features': 'Distillation: how far the student\'s internal features are from the teacher\'s (feature matching).',
    'durationErrorFrames': 'Distillation: how far the student\'s phone durations are from the teacher\'s, in 10 ms frames.',
    'book': 'G2P: the share of word tokens in the evaluation books whose predicted pronunciation is wrong (weighted by how ' +
        'often each word occurs - what a listener hears).',
    'bookAny': 'G2P: book-token error, counting any of the dictionary\'s pronunciations as right.',
    'bookSeen': 'G2P: book-token error on words the model trained on.',
    'bookSeenAny': 'G2P: book-token error on trained words, any listed pronunciation counting as right.',
    'bookStripped': 'G2P: book-token error after stripping suffixes.',
    'bookWrongDistinct': 'G2P: the number of distinct book words pronounced wrong (each word counted once).',
    'top10kWrong': 'G2P: how many of the 10,000 most frequent book words are pronounced wrong.',
    'top10kWrongSeen': 'G2P: how many of the 10,000 most frequent words, among those trained on, are wrong.',
    'wer': 'G2P: word error rate - the share of words with any phone wrong.',
    'per': 'G2P: phone error rate - the share of phones wrong.',
    'testWer': 'G2P: word error rate on held-out dictionary words the model never saw (generalization to new spellings).',
    'testWerFirst': 'G2P: held-out word error rate against only each word\'s first listed pronunciation.',
    'testPer': 'G2P: phone error rate on held-out words.',
    'trainWer': 'G2P: word error rate on the training words.',
    'monitorWer': 'G2P: word error rate on a fixed monitoring sample, tracked during training.'
};


// The explanation of a measurement - its own, or an emotional-speech variant of one
function measureHelp(key) {
    if (key in chartHelp) {
        return chartHelp[key];
    }
    const base = key.startsWith('cremad.') ? key.slice('cremad.'.length) : null;
    return base !== null && base in chartHelp ? `On the emotional speech (CREMA-D): ${chartHelp[base]}` : null;
}


// A (?) button that shows an explanation - on hover (its title) and, tapped or clicked, beside it (remembered by key
// across refreshes)
const openHelp = new Set();
function helpTip(key, text) {
    const wrap = make('span', 'help');
    const button = make('button', 'help-button', '?');
    button.type = 'button';
    button.title = text;
    button.setAttribute('aria-label', 'What is this?');
    const note = make('span', 'help-text', text);
    const show = (open) => {
        note.hidden = !open;
        button.setAttribute('aria-expanded', String(open));
    };
    show(openHelp.has(key));
    button.addEventListener('click', (event) => {
        event.stopPropagation();
        if (openHelp.has(key)) {
            openHelp.delete(key);
        } else {
            openHelp.add(key);
        }
        show(openHelp.has(key));
    });
    wrap.append(button, note);
    return wrap;
}


// The Runs heading's explanation - the run cards' facts
document.getElementById('runsTitle').append(' ', helpTip('runs', 'Each card is a training run: its status (live - updating; ' +
    'stalled - quiet for minutes; stopped - quiet for long; finished - reached its target), steps done of its target, ' +
    'seconds per step and the time left at that pace, the latest training total loss and validation loss (lower is ' +
    'better - the spectral loss for training on recordings; for distillation runs (distill-sNN), the loss against the ' +
    'teacher, on its own small scale), and when its log last changed. Select a card for its curves; each chart\'s (?) ' +
    'explains its measurement.'));


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


// Render the selected run's charts (the Runs tab) - only when its data changed, unless forced
function renderCharts(force = false) {
    if (currentTab !== 'runs') {
        return;
    }
    const run = runs.get(selectedRun);
    drawRunCharts(chartsElement, chartsTitle, run ?? null, run ? referenceRun(run) : null, run ? parentRun(run) : null, force);
}


// Draw a run's charts into an element (its title into titleElement) - with its reference run and the run it continues
// (parent, {run, offset}) faded behind it - only when its data changed, unless forced; with extras (a list), only those
// of the other measurements' charts
const chartKeys = new Map();
function drawRunCharts(element, titleElement, run, reference, parent, force = false, extras = null) {
    const dataKey = run ? [run.name, run.modified, (run.entries.train || []).length, (run.entries.val || []).length,
        reference?.name, reference?.modified, element.clientWidth].join(' ') : '';
    if (!force && dataKey === chartKeys.get(element)) {
        return;
    }
    chartKeys.set(element, dataKey);
    if (!run) {
        titleElement.textContent = 'Curves';
        element.replaceChildren(make('p', 'empty', 'Select a run.'));
        return;
    }
    titleElement.textContent = parent ? `Curves - ${run.name} (continuing ${parent.run.name}, faded)` : `Curves - ${run.name}`;
    if (reference) {
        titleElement.textContent += ` - compared with ${reference.name} (faded)`;
    }
    const styles = getComputedStyle(document.documentElement);
    const colors = ['--accent', '--series-2', '--series-3', '--series-4'].map((name) => styles.getPropertyValue(name));
    const train = run.entries.train || [];
    const val = run.entries.val || [];
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
        {'title': 'Loss (total)', 'help': 'total', 'logScale': true, 'series': [
            ['train (smoothed)', 'train', 'total', true, colors[0]], ['val', 'val', 'total', false, colors[1]]
        ]},
        {'title': 'Spectral loss', 'help': 'spectral', 'series': [
            ['train (smoothed)', 'train', 'spectral', true, colors[0]], ['val', 'val', 'spectral', false, colors[1]]
        ]},
        {'title': 'Durations', 'help': 'durations', 'logScale': true, 'series': [
            ['phones (train)', 'train', 'durPhone', true, colors[0]], ['phones (val)', 'val', 'durPhone', false, colors[1]],
            ['pauses (val)', 'val', 'durPaused', false, colors[2]]
        ]},
        {'title': 'Pitch error (val, cents)', 'help': 'f0Cents', 'series': [['F0 error', 'val', 'f0Cents', false, colors[3]]]},
        {'title': 'Seconds per step (stalls show as spikes)', 'help': 'secondsPerStep', 'logScale': true, 'series': [
            ['s/step', 'train', 'secondsPerStep', false, colors[2]]
        ]},
        {'title': 'Learning rate', 'help': 'lr', 'series': [['lr', 'train', 'lr', false, colors[0]]]}
    ];

    // Any other numeric training or validation keys become extra charts (e.g. adversarial losses)
    const known = new Set(['step', 'lr', 'secondsPerStep', 'total', 'spectral', 'durPhone', 'durPause', 'durPaused', 'f0Cents', 'guide']);
    const extraKeys = [...new Set([...train, ...val].flatMap((entry) => Object.keys(entry)))].
        filter((key) => !known.has(key) && (extras === null || extras.includes(key)));
    for (const key of extraKeys) {
        charts.push({'title': key, 'help': key, 'logScale': false, 'series': [
            ['train (smoothed)', 'train', key, true, colors[0]], ['val', 'val', key, false, colors[1]]
        ]});
    }

    // The series' points - a continued run's parent drawn behind it, faded, up to the step the run started from
    const entriesOf = {train, val};
    for (const chart of charts) {
        chart.series = chart.series.flatMap(([label, kind, key, smoothed, color]) => {
            const own = series(entriesOf[kind], key);
            const ownSeries = {label, 'points': smoothed ? smoothSeries(own, 0.8) : own, color};
            const drawn = [ownSeries];
            const before = parentSeries(kind, key);
            if (before.length !== 0) {
                const parentPoints = smoothed ? smoothSeries(before, 0.8) : before;
                drawn.unshift({'label': `${parent.run.name} ${label}`, 'points': parentPoints, 'color': faded(color)});
            }
            if (reference) {
                const theirs = series(reference.entries[kind] || [], key).map(([step, value]) => [step - offset, value]);
                if (theirs.length !== 0) {
                    drawn.unshift({'label': `${reference.name} ${label}`, 'points': smoothed ? smoothSeries(theirs, 0.8) : theirs,
                        'color': faded(color)});
                }
            }
            return drawn;
        });
    }
    element.replaceChildren(...charts.map((chart) => {
        const wrap = make('div', 'chart');
        const canvas = make('canvas');
        const legend = make('div', 'legend');
        for (const item of chart.series) {
            const swatch = make('span', '', item.label);
            swatch.style.setProperty('--swatch', item.color);
            legend.append(swatch);
        }
        const title = make('h3', '', chart.title);
        const help = measureHelp(chart.help);
        if (help !== null) {
            title.append(' ', helpTip(`chart:${chart.help}`, help));
        }
        wrap.append(title, canvas, legend);
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


// The dashboard's settings (pipeline.json) - the make train run to show (train), the runs' planned stops (stopAt) and
// comparisons (compare), and the checkpoint samples (samples, also the monitor's)
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


// The archived pipelines (tools/dashboard/pipelines/*.json - earlier experiment pipelines: phases of steps, each tracked
// by a log, a training run, an output file (or a file containing a text), or a process) and the selected one, newest
// first - the archive selector's choice, remembered per browser
const archiveKey = 'training-dashboard-archive-v1';
const archiveSelect = document.getElementById('archiveSelect');
let archived = null;
let archiveNames = [];
let archiveListed = 0;
archiveSelect.addEventListener('change', () => {
    storageSet(archiveKey, archiveSelect.value);
    archived = null;
    renderPipeline(lastLogs);
});
async function loadArchive() {
    if (Date.now() - archiveListed > 30000) {
        archiveListed = Date.now();
        const entries = await listDirectory(new URL('pipelines/', import.meta.url));
        const names = entries.filter((entry) => entry.name.endsWith('.json')).map((entry) => entry.name.slice(0, -5)).sort().reverse();
        if (names.join() !== archiveNames.join()) {
            archiveNames = names;
            archiveSelect.replaceChildren(...names.map((name) => {
                const option = make('option', '', name);
                option.value = name;
                return option;
            }));
            const remembered = storageGet(archiveKey);
            archiveSelect.value = names.includes(remembered) ? remembered : names[0] ?? '';
        }
    }
    if (archiveSelect.value === '') {
        archived = null;
        return;
    }
    const result = await fetchText(new URL(`pipelines/${encodeURIComponent(archiveSelect.value)}.json`, import.meta.url));
    try {
        archived = result === null ? null : JSON.parse(result.text);
    } catch {
        archived = null;
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
const assignedKey = 'training-dashboard-assigned-v2';
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
    const since = Date.parse(archived.since);
    const plannedSteps = archived.phases.flatMap((phase) => phase.steps);
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
    const phases = await Promise.all(archived.phases.map(async (phase) => ({
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
    // A phase's rows by index - clamped to the pipeline's phases (an assignment can outlive a pipeline change)
    const phaseRows = (phaseIndex) => phases[Math.max(0, Math.min(phaseIndex, phases.length - 1))].rows;
    const addRow = (phaseIndex, row) => {
        phaseRows(phaseIndex).push({...row, 'weight': 0, 'unplanned': true});
    };

    // Agents lead their phase
    for (const agent of activity?.agents || []) {
        const phaseIndex = assign(`agent:${agent.id}`, archived.phases.findIndex((phase) => phase.title === agent.phase));
        phaseRows(phaseIndex).unshift({
            'id': `agent:${agent.id}`, 'title': `agent: ${agent.label}`, 'kind': 'agent', 'weight': 0, 'unplanned': true,
            'state': agentState(agent)
        });
    }

    // Unplanned logs join the phase whose planned logs share their folder
    const folder = (path) => path.slice(0, path.lastIndexOf('/') + 1);
    const phaseOfFolder = (path) => archived.phases.findIndex((phase) => phase.steps.some((step) => (
        step.log && folder(step.log) === folder(path)
    )));
    const ownLogs = logs.filter((item) => !item.file.path.startsWith('make train: '));
    for (const log of ownLogs.filter((item) => item.result.modified >= since && !planned.has(`build/nn/${item.file.path}`)).
        sort((first, second) => first.result.modified - second.result.modified)) {
        const id = `log:${log.file.path}`;
        const row = {id, 'title': log.file.path, 'kind': 'log', 'state': logState(log.file.path, log)};
        addRow(assign(id, phaseOfFolder(`build/nn/${log.file.path}`)), row);
    }

    // Other runs join the phase whose run prefixes match (else the training phase, else the phase running when they appeared)
    const trainPhase = archived.phases.findIndex((phase) => phase.steps.some((step) => step.run));
    for (const run of newRuns.filter((item) => item !== newestRun)) {
        const phaseIndex = archived.phases.findIndex((phase) => (phase.runs || []).some((prefix) => run.name.startsWith(prefix)));
        const id = `run:${run.name}`;
        const row = {id, 'title': `run ${run.name}`, 'kind': 'run', 'state': runState(run)};
        addRow(phaseIndex === -1 ? assign(id, trainPhase) : phaseIndex, row);
    }

    // Unplanned processes join the phase whose process prefixes match
    const plannedProcess = (entry) => plannedSteps.some((step) => step.process && entry.label.startsWith(step.process));
    for (const entry of processes.filter((item) => !plannedProcess(item))) {
        const id = `process:${entry.label} ${Math.round(entry.started / 10000)}`;
        const phaseIndex = archived.phases.findIndex((phase) => (phase.processes || []).some((prefix) => entry.label.startsWith(prefix)));
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
    if (currentTab !== 'archive') {
        return;
    }
    await loadArchive();
    pipelineSection.hidden = archived === null;
    if (archived === null) {
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
    pipelineTitle.textContent = archived.title;
    pipelineBar.style.width = `${(100 * overall).toFixed(1)}%`;
    const elapsed = formatDuration((Date.now() - Date.parse(archived.since)) / 1000);
    const monitor = activity && Date.now() - activity.updated < 10000 ? '' :
        ' · the monitor isn\'t running (node tools/dashboard/monitor.js) - steps without logs aren\'t shown';
    pipelineFacts.textContent = `about ${Math.round(100 * overall)}% (weighted by expected time) · started ${elapsed} ago${monitor}`;
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


// Render the activity logs - make train's logs, then the others updated in the last hour (all of them with "Show
// older logs"), the most recently updated first, each one's tail - rebuilt only when they change
let logsKey = null;
let logHeads = [];
const recentMs = 60 * 60 * 1000;
const olderKey = 'training-dashboard-older-logs-v1';
const olderToggle = document.getElementById('olderLogs');
olderToggle.checked = storageGet(olderKey) === 'true';
olderToggle.addEventListener('change', () => {
    storageSet(olderKey, String(olderToggle.checked));
    renderLogs(lastLogs);
});
function renderLogs(allLogs) {
    const activeLogs = allLogs.filter(({result}) => Date.now() - result.modified < stalledMs).length;
    setTabFact('tabFactLogs', activeLogs === 0 ? '' : `${activeLogs} active`);
    const isTrain = (log) => log.file.path.startsWith('make train: ');
    const recent = allLogs.filter((log) => isTrain(log) || Date.now() - log.result.modified < recentMs);
    const shown = olderToggle.checked ? allLogs : recent;
    document.getElementById('olderCount').textContent = `(${allLogs.length - recent.length} more)`;
    const logs = [...shown.filter(isTrain), ...shown.filter((log) => !isTrain(log))].slice(0, olderToggle.checked ? 40 : 12);
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
// voices, newest first, then the package's voices (the reference) - rebuilt when the index changes, reusing each sample's player so
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
        updateListenDot();
    }
    const blocks = checkpoints.map((checkpoint) => block(
        `${checkpoint.run} · step ${checkpoint.step.toLocaleString()}`,
        `female ${meanScore(checkpoint.samples, 'female')} · male ${meanScore(checkpoint.samples, 'male')} · ` +
            `rendered ${new Date(checkpoint.rendered).toLocaleTimeString()}`,
        checkpoint.samples
    ));
    if (index.reference.length !== 0) {
        const facts = `female ${meanScore(index.reference, 'female')} · male ${meanScore(index.reference, 'male')}`;
        blocks.push(block('The package\'s voices (reference)', facts, index.reference));
    }
    if (checkpoints.length === 0) {
        blocks.unshift(make('p', 'empty', 'No checkpoints yet - samples appear as each one is saved.'));
    }
    checkpointsElement.replaceChildren(...blocks);
    checkpointSection.hidden = false;
}


// make train's listening samples (TRAIN_DIR/samples/index.json, tools/trainsamples.js) - each model's samples, newest
// model first, every speaker's sample beside the shipped model's (the reference); a model not rendered yet is listed as
// waiting, so nothing moves when it arrives. Rebuilt when the index or the models' states change, reusing each sample's
// player so playback continues.
const listenTitles = {'teacher-a': 'Teacher A', 'teacher-b': 'Teacher B', 'student': 'The student', 'polish': 'The polished student',
    'reference': 'The shipped model'};
const listenNotes = {'teacher-a': 'its final checkpoint, which teacher B starts from', 'teacher-b': 'the distillation\'s teacher',
    'student': 'distilled from teacher B, before its polish',
    'polish': 'the model make train packages, if it passes the gate',
    'reference': 'lib/models/voiceModel.js, before make train replaces it'};
const trainSamplesHelp = 'make train renders listening samples of each training stage\'s final model as the stage finishes ' +
    '(its own CPU stage, beside the next GPU stage): teacher A (its last checkpoint, which teacher B starts from), teacher B ' +
    '(its best - the distillation\'s teacher), the student (its best, before the polish), and the polished student (its ' +
    'best - the model the package stage packages, if it passes the gate). Each speaks the recipe\'s texts ' +
    '(data/nn/train.json samples) in six of the training speakers\' voices, three female and three male, exactly as the ' +
    'package speaks (exported and quantized as the package stage exports it, its G2P network, 24 kHz, its gain). Beside each ' +
    'is the shipped model, ' +
    'lib/models/voiceModel.js, rendered the same way before make train replaces it - so the new model can be judged by ' +
    'ear. The numbers are UTMOS (1-5, a neural predictor of naturalness; real recordings score about 4.1-4.3), scored on ' +
    'the CPU - green when above the shipped model\'s, red when below by 0.1 or more. A model not rendered yet is listed as ' +
    'waiting. The files are TRAIN_DIR/samples/ (index.json lists them).';
document.getElementById('trainSamplesTitle').append(' ', helpTip('trainSamples', trainSamplesHelp));
let trainSamplesKey = null;
const trainSampleTiles = new Map();
let newestTrainSamples = null;
async function renderTrainSamples() {
    const section = document.getElementById('trainSamplesSection');
    if (!trainData) {
        section.hidden = true;
        return;
    }
    const samplesUrl = new URL(`${trainData.config.dir.replace(/\/$/, '')}/samples/`, rootUrl);
    const [result, recipeResult] = await Promise.all([fetchText(new URL('index.json', samplesUrl)),
        fetchText(new URL('data/nn/train.json', rootUrl))]);
    let index;
    let texts;
    try {
        index = result === null ? null : JSON.parse(result.text);
        texts = JSON.parse(recipeResult?.text ?? '{}').samples?.texts ?? [];
    } catch {
        // Being written - the last rendering stays
        return;
    }
    const models = trainModel.listenModels(trainData.manifest, trainData.states, index);
    const key = JSON.stringify([samplesUrl.href, result?.text ?? null, recipeResult?.text ?? null,
        models.map((model) => [model.key, model.status])]);
    if (key === trainSamplesKey) {
        return;
    }
    trainSamplesKey = key;

    // A sample's player in a model's block (reused across rebuilds - the shipped model's samples have one in each block)
    const player = (sample, block) => {
        const id = `${block} ${samplesUrl.href}${sample.file}`;
        if (!trainSampleTiles.has(id)) {
            const audio = make('audio');
            audio.controls = true;
            audio.preload = 'none';
            audio.src = new URL(encodeURIComponent(sample.file), samplesUrl);
            trainSampleTiles.set(id, audio);
        }
        return trainSampleTiles.get(id);
    };
    const mosText = (sample) => (Number.isFinite(sample?.mos) ? sample.mos.toFixed(2) : '-');

    // A model's samples by text, each speaker's tile with the shipped model's sample of the same text below it
    const reference = models.find((model) => model.key === 'reference');
    const referenceOf = (sample) => reference?.samples.find((item) => item.speaker === sample.speaker && item.text === sample.text);
    const textKeys = (samples) => [...new Set([...texts.map((text) => text.key), ...samples.map((sample) => sample.text)])].
        filter((textKey) => samples.some((sample) => sample.text === textKey));
    const modelBlock = (model) => {
        const element = make('div', 'checkpoint');
        const head = make('div', 'checkpoint-head');
        const title = make('h3', '', listenTitles[model.key] ?? model.key);
        head.append(title);
        element.append(head);
        if (model.samples.length === 0) {
            const stageName = model.stage ?? model.key;
            const status = {
                'waiting': `waiting - rendered when the ${model.key === 'reference' ? 'readers\' data' : model.key} stage finishes`,
                'queued': 'rendering...', 'running': 'rendering...', 'stopped': 'stopped - make train isn\'t running',
                'failed': `failed - see logs/${stageName}.log`, 'skipped': 'not in this run\'s make target'
            }[model.status] ?? model.status;
            head.append(make('span', 'facts', `${listenNotes[model.key] ?? ''} · ${status}`));
            element.classList.add('waiting');
            return element;
        }
        const [first] = model.samples;
        const facts = [listenNotes[model.key], first.run, first.step === null ? null : `step ${first.step.toLocaleString()}`,
            `UTMOS female ${meanScore(model.samples, 'female')} · male ${meanScore(model.samples, 'male')}`];
        if (model.key !== 'reference' && reference && reference.samples.length !== 0) {
            facts.push(`shipped ${meanScore(reference.samples, 'female')} · ${meanScore(reference.samples, 'male')}`);
        }
        head.append(make('span', 'facts', facts.filter((fact) => fact).join(' · ')));
        for (const textKey of textKeys(model.samples)) {
            const text = texts.find((item) => item.key === textKey);
            const line = make('div', 'sample-text');
            line.append(make('strong', '', text?.label ?? textKey), make('span', 'muted', text?.text ?? ''));
            // (three across - the recipe's three female speakers, then its three male)
            const voices = make('div', 'voices speakers');
            for (const sample of model.samples.filter((item) => item.text === textKey)) {
                const tile = make('div', 'voice');
                const voiceHead = make('div', 'voice-head');
                voiceHead.append(make('span', '', sample.label), make('span', 'muted', sample.sex));
                tile.append(voiceHead);
                const shipped = model.key === 'reference' ? null : referenceOf(sample);
                const take = make('div', 'voice-take');
                const mos = make('span', 'mos', mosText(sample));
                if (shipped && Number.isFinite(sample.mos) && Number.isFinite(shipped.mos)) {
                    mos.classList.toggle('better', sample.mos > shipped.mos);
                    mos.classList.toggle('worse', sample.mos <= shipped.mos - 0.1);
                }
                take.append(make('span', '', model.key === 'reference' ? 'shipped' : 'new'), mos);
                tile.append(take, player(sample, model.key));
                if (shipped) {
                    const shippedTake = make('div', 'voice-take');
                    shippedTake.append(make('span', '', 'shipped'), make('span', 'mos', mosText(shipped)));
                    tile.append(shippedTake, player(shipped, model.key));
                }
                voices.append(tile);
            }
            element.append(line, voices);
        }
        return element;
    };

    // The models, newest first, then the shipped model's own samples
    const trained = models.filter((model) => model.key !== 'reference').reverse();
    const blocks = [...trained, ...reference ? [reference] : []].map(modelBlock);
    if (blocks.length === 0) {
        blocks.push(make('p', 'empty', 'This make train run renders no listening samples.'));
    }
    document.getElementById('trainSamples').replaceChildren(...blocks);
    const rendered = models.filter((model) => model.samples.length !== 0);
    document.getElementById('trainSamplesFacts').textContent =
        `${rendered.length} of ${models.length} rendered · ${trainData.config.dir.replace(/\/$/, '')}/samples/`;
    section.hidden = false;

    // The newest model rendered (the Listen tab's dot and fact)
    const newest = trained.find((model) => model.samples.length !== 0) ?? null;
    newestTrainSamples = newest === null ? null :
        {'key': `${samplesUrl.href} ${newest.key}`, 'title': listenTitles[newest.key] ?? newest.key};
    updateListenDot();
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
        const checkpointFact = newestCheckpoint === null ? '' : `step ${Number(newestCheckpoint.split(' ')[1]).toLocaleString()}`;
        setTabFact('tabFactListen', newestTrainSamples?.title.toLowerCase() ?? checkpointFact);
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


// make train (the Train tab) - pipeline.json's train: {dir (TRAIN_DIR, default build/train), log (make's output - for a
// run without stage events, its stages are inferred from it and the stamps), logs (other make logs, searched for
// failures and checks), util (default TRAIN_DIR/util.log), target (the make target, else the log's), manifest (default
// data/nn/train-stages.json), compare ({run: the reference run in build/nn/runs/ drawn faded behind it})}
let trainManifestText = null;
let trainManifest = null;
let trainStamps = new Map();
let trainStampsChecked = 0;
const trainRuns = new Map();
const trainRunKey = 'training-dashboard-train-run-v1';
let trainSelectedRun = storageGet(trainRunKey);
let trainData;


// A run's planned stop - its data/nn/train.json section's (a dotted path, e.g. teacher.a) stop, else its steps
function recipeStop(recipe, path) {
    const section = path.split('.').reduce((value, key) => value?.[key], recipe);
    return section?.stop ?? section?.steps ?? null;
}


// The last line of texts matching a pattern (the later texts first) - or null
function lastMatch(texts, pattern) {
    for (const text of texts) {
        const lines = (text || '').split('\n').filter((line) => pattern.test(line));
        if (lines.length !== 0) {
            return lines[lines.length - 1].trim();
        }
    }
    return null;
}


// The last-modified time of each of a directory's stamps (*.done) - Map stage -> ms, listed every 5 seconds
async function loadStamps(dirUrl) {
    if (Date.now() - trainStampsChecked < 5000) {
        return trainStamps;
    }
    trainStampsChecked = Date.now();
    const entries = (await listDirectory(dirUrl)).filter((entry) => !entry.isDirectory && entry.name.endsWith('.done'));
    const stamps = new Map();
    await Promise.all(entries.map(async (entry) => {
        const name = entry.name.slice(0, -'.done'.length);
        let modified = trainStamps.get(name);
        if (!Number.isFinite(modified)) {
            try {
                const response = await fetch(new URL(encodeURIComponent(entry.name), dirUrl), {'method': 'HEAD', 'cache': 'no-store'});
                modified = Date.parse(response.headers.get('Last-Modified') || '');
            } catch {
                modified = NaN;
            }
        }
        stamps.set(name, modified);
    }));
    trainStamps = stamps;
    return stamps;
}


// Load make train's state - {config, manifest, states, forecast, samples, alive, start, end, status, gates, logFiles} or
// null (no stage manifest)
async function loadTrain() {
    const config = {'dir': 'build/train', ...pipeline?.train};
    const dirUrl = new URL(`${config.dir.replace(/\/$/, '')}/`, rootUrl);
    const manifestResult = await fetchText(new URL(config.manifest ?? 'data/nn/train-stages.json', rootUrl));
    if (manifestResult === null) {
        return null;
    }
    if (manifestResult.text !== trainManifestText) {
        try {
            trainManifest = JSON.parse(manifestResult.text);
            trainManifestText = manifestResult.text;
        } catch {
            return trainData;
        }
    }
    const manifest = trainManifest;
    let recipe;
    try {
        recipe = JSON.parse((await fetchText(new URL('data/nn/train.json', rootUrl)))?.text);
    } catch {
        recipe = null;
    }

    // The bookkeeping - the stage events, the stamps, make's logs, and the utilization samples
    const [eventsResult, makeResult, utilResult, stamps, ...extra] = await Promise.all([
        fetchText(new URL('stages.jsonl', dirUrl)),
        config.log ? fetchText(new URL(config.log, rootUrl)) : null,
        fetchText(config.util ? new URL(config.util, rootUrl) : new URL('util.log', dirUrl)),
        loadStamps(dirUrl),
        ...(config.logs || []).map((path) => fetchText(new URL(path, rootUrl)))
    ]);
    const events = eventsResult === null ? new Map() : trainModel.parseStageEvents(eventsResult.text);
    const segments = makeResult === null ? [] : trainModel.parseMakeLog(makeResult.text);
    const stageLogs = new Map();
    await Promise.all(manifest.stages.filter((stage) => events.has(stage.stage)).map(async (stage) => {
        const result = await fetchText(new URL(`logs/${encodeURIComponent(stage.stage)}.log`, dirUrl));
        if (result !== null) {
            stageLogs.set(stage.stage, result);
        }
    }));
    // (a GPU stage waiting for the GPU lock - its last gpulock line is "waiting")
    const queued = new Set([...stageLogs].filter(([, result]) => (
        /^gpulock: waiting/.test(lastMatch([result.text], /^gpulock: /) || '')
    )).map(([name]) => name));
    const utilFresh = utilResult !== null && Date.now() - utilResult.modified < 90000;
    const alive = utilFresh || (makeResult !== null && Date.now() - makeResult.modified < 10 * 60000);
    const extraTexts = extra.filter((result) => result !== null).map((result) => result.text);
    const states = trainModel.stageStates(manifest, {events, stamps, segments, 'errorTexts': extraTexts, alive, queued,
        'target': config.target});

    // The training runs (TRAIN_DIR/runs/<run>/), each stopped at its recipe's stop
    for (const stage of manifest.stages.filter((item) => item.run)) {
        const base = new URL(`runs/${encodeURIComponent(stage.run)}/`, dirUrl);
        const [jsonLog, textLog] = await Promise.all([fetchText(new URL('log.jsonl', base)), fetchText(new URL('train.log', base))]);
        const info = textLog === null ? {} : parseTextLog(textLog.text);
        const stop = stage.recipe ? recipeStop(recipe, stage.recipe) : null;
        if (stop !== null) {
            info.targetSteps = Math.min(stop, info.targetSteps ?? Infinity);
        }
        const waiting = jsonLog === null && textLog === null;
        trainRuns.set(stage.run, {
            'name': stage.run, 'stage': stage.stage, info, waiting,
            'entries': jsonLog === null ? {'train': [], 'val': []} : parseJsonLog(jsonLog.text),
            'modified': Math.max(jsonLog?.modified || 0, textLog?.modified || 0),
            'note': `not started - the ${stage.stage} stage is ${states.get(stage.stage).status}`
        });
    }

    // The forecast - a running training run's remaining steps at its recent pace, another running stage's expected time
    // less its time so far (at least a minute), and a waiting stage's expected time
    const now = Date.now();
    const remaining = (stage, state) => {
        const run = stage.run ? trainRuns.get(stage.run) : null;
        const started = ['running', 'stopped', 'queued'].includes(state.status);
        if (started && run && !run.waiting) {
            const entries = run.entries.train || [];
            const last = entries[entries.length - 1];
            const secondsPerStep = median(entries.slice(-20).map((entry) => entry.secondsPerStep).filter(Number.isFinite));
            if (last && run.info.targetSteps && Number.isFinite(secondsPerStep)) {
                return Math.max(0, run.info.targetSteps - last.step) * secondsPerStep / 60 + 0.5;
            }
        }
        if (started && state.start !== null) {
            return Math.max(stage.minutes - (now - state.start) / 60000, 1);
        }
        return stage.minutes;
    };
    const plan = trainModel.forecast(manifest, states, remaining, now);
    const starts = [...states.values()].map((state) => state.start).filter((time) => time !== null);
    const firstEvent = Math.min(...[...events.values()].map((event) => event.start ?? Infinity));
    const start = Math.min(segments[0]?.start ?? Infinity, firstEvent, ...starts);
    const counted = [...states.values()].filter((state) => state.status !== 'skipped');
    let status = 'waiting';
    if (counted.length !== 0 && counted.every((state) => state.status === 'done')) {
        status = 'finished';
    } else if (counted.some((state) => state.status === 'failed')) {
        status = 'failed';
    } else if (alive && Number.isFinite(start)) {
        status = 'running';
    } else if (Number.isFinite(start)) {
        status = 'stopped';
    }
    const end = status === 'finished' ? Math.max(...counted.map((state) => state.end)) : null;
    const samples = utilResult === null ? [] : trainModel.parseUtilLog(utilResult.text, Number.isFinite(start) ? start : now);

    // The gates' results - the evaluations (picovox_train.distill eval and pick) and the checks in the logs
    const [teacherResult, studentResult, summaryResult] = await Promise.all(['distill/eval/teacher/result.json',
        'distill/eval/student/result.json', 'distill/summary.json'].map((path) => fetchText(new URL(path, dirUrl))));
    const parse = (result) => {
        try {
            return result === null ? null : JSON.parse(result.text);
        } catch {
            return null;
        }
    };
    const texts = (stage) => [stageLogs.get(stage)?.text, ...extraTexts, makeResult?.text];
    const gates = {
        'heteronyms': lastMatch(texts('heteronyms'), /^check: (?:the \d+ relabeled|the relabeled)/),
        'g2pCheck': lastMatch(texts('g2p'), /^check: .*g2pModel/),
        'g2pFolds': lastMatch(texts('context-network'), /^\d+ folds: /),
        'teacher': parse(teacherResult),
        'student': parse(studentResult),
        'summary': parse(summaryResult)
    };

    // make train's logs (for the Logs tab)
    const logFiles = [];
    const addLog = (label, url) => logFiles.push({'path': `make train: ${label}`, url});
    if (config.log) {
        addLog(config.log, new URL(config.log, rootUrl));
    }
    for (const path of config.logs || []) {
        addLog(path, new URL(path, rootUrl));
    }
    for (const name of stageLogs.keys()) {
        addLog(`logs/${name}.log`, new URL(`logs/${encodeURIComponent(name)}.log`, dirUrl));
    }
    addLog(config.util ?? 'util.log', config.util ? new URL(config.util, rootUrl) : new URL('util.log', dirUrl));
    return {config, manifest, states, plan, samples, alive, start, end, status, gates, logFiles, now};
}


// The explanations of the Train tab's sections
const trainHelp = {
    'summary': 'make train rebuilds the generated models (lib/models/) from the recipe, data/nn/train.json, one Make target ' +
        'per stage (data/nn/train-stages.json lists them, their dependencies, and their expected minutes). The ETA ' +
        'schedules the stages not done yet: each starts when the stages it needs finish (a GPU stage also when the GPU is ' +
        'free - the GPU stages run one at a time) and takes its expected time - a running training run, its remaining steps ' +
        'at its recent seconds per step. The critical path is the chain of stages that sets the ETA: its stages are ' +
        'marked ✓ done, ● running, ○ waiting; the bar is the run\'s time so far of its time to the ETA.',
    'util': 'How busy the machine is over the run: the CPU busy % (all cores, from top) and the GPU utilization % ' +
        '(ioreg\'s "Device Utilization %"), sampled every 20 seconds into TRAIN_DIR/util.log by make train. Shaded: ' +
        'the GPU idle (under 10%) - time the GPU-bound critical path isn\'t moving, unless a CPU stage on it is running. ' +
        'Below, each stage as a band over its time (blue: GPU stage, green: CPU stage, gray: network); the faded bands ' +
        'after now are the forecast. The goal is both near 100% for the shortest time.',
    'stages': 'Every stage of make train, grouped, in the order they can run: waiting (its inputs aren\'t done), queued ' +
        '(waiting for the GPU lock), running, done, failed, stopped (started, but make isn\'t running - resumable), or ' +
        'skipped (not in this run\'s make target). Each shows its start and end, its time against the expected time, and ' +
        'the mean CPU and GPU use while it ran. Stage events come from TRAIN_DIR/stages.jsonl; without them (a run before ' +
        'the bookkeeping) the states are inferred from the stamps (*.done) and make\'s log.',
    'runs': 'The make train stages that train a network, each a run in TRAIN_DIR/runs/ - the teacher\'s stages A and B, ' +
        'the distilled student, and its polish. Select one for its curves; the teacher\'s are drawn over the earlier ' +
        'teacher runs (multi-a, multi-b - faded) to show whether this run is on the same track.',
    'gates': 'The checks make train must pass: the heteronym relabeling reproduces data/nn/g2p-relabel.json; the ' +
        'retrained G2P is identical to the committed lib/models/g2pModel.js (else within the recipe\'s tolerances) with ' +
        'its held-out context accuracy; the teacher and the student through the listener panel (UTMOS - naturalness, ' +
        '1-5; Whisper\'s word error rate; the speaker similarity to the teacher); and the gate, picovox_train.distill pick - the ' +
        'student within the tolerance of the teacher.'
};
for (const [id, key] of [['trainTitle', 'summary'], ['utilTitle', 'util'], ['stagesTitle', 'stages'], ['trainRunsTitle', 'runs'],
    ['gatesTitle', 'gates']]) {
    document.getElementById(id).append(' ', helpTip(`train:${key}`, trainHelp[key]));
}


// A time of day (HH:MM)
function clockTime(time) {
    return new Date(time).toLocaleTimeString([], {'hour': 'numeric', 'minute': '2-digit'});
}


// A stage's status label and fraction done - {label, fraction}
const stageIcons = {'waiting': '○', 'queued': '◌', 'running': '●', 'done': '✓', 'failed': '✕', 'stopped': '■', 'skipped': '–'};
function stageProgress(stage, state, data) {
    const run = stage.run ? trainRuns.get(stage.run) : null;
    if (state.status === 'running' || state.status === 'stopped') {
        let fraction = null;
        const entries = run && !run.waiting ? run.entries.train || [] : [];
        if (entries.length !== 0 && run.info.targetSteps) {
            fraction = Math.min(1, entries[entries.length - 1].step / run.info.targetSteps);
        } else if (state.start !== null) {
            fraction = Math.min(0.99, (data.now - state.start) / 60000 / stage.minutes);
        }
        const left = data.plan.finishes.get(stage.stage) - data.now;
        const percent = fraction === null ? '' : `${Math.round(100 * fraction)}%`;
        const label = state.status === 'stopped' ? `stopped ${percent}` : `${percent} · ${formatDuration(left / 1000)} left`;
        return {label, fraction};
    }
    const labels = {'waiting': 'waiting', 'queued': 'queued for the GPU', 'done': 'done', 'failed': 'failed',
        'skipped': 'not in this run'};
    return {'label': labels[state.status], 'fraction': {'done': 1, 'failed': 1}[state.status] ?? 0};
}


// A stage's facts line - what it runs on, when it ran, its time against the expected time, and its CPU and GPU use
function stageLine(stage, state, data) {
    const parts = [stage.on.toUpperCase()];
    if (state.start !== null) {
        parts.push(`${clockTime(state.start)}-${state.end === null ? 'now' : clockTime(state.end)}`);
        const seconds = ((state.end ?? data.now) - state.start) / 1000;
        parts.push(`${formatDuration(seconds)} (expected ~${formatDuration(stage.minutes * 60)})`);
        const use = trainModel.meanUtilization(data.samples, state.start, state.end ?? data.now);
        if (use.count !== 0) {
            parts.push(`CPU ${Math.round(use.cpu)}% · GPU ${Math.round(use.gpu)}%`);
        }
    } else if (state.end !== null) {
        parts.push(`done ${clockTime(state.end)}`);
    } else if (state.status === 'skipped') {
        parts.push('not in this run\'s make target');
    } else {
        const finish = data.plan.finishes.get(stage.stage);
        const forecastText = finish ? ` · forecast ${clockTime(finish - stage.minutes * 60000)}-${clockTime(finish)}` : '';
        parts.push(`expected ~${formatDuration(stage.minutes * 60)}${forecastText}`);
    }
    if (state.failure) {
        parts.push(state.failure);
    }
    return `${parts.join(' · ')} - ${stage.title}`;
}


// Build the Train tab's fixed elements from the stage manifest - every stage listed from the start (in collapsible
// groups, remembered per browser), the training runs, and the gates - updated in place after
const trainGroupsKey = 'training-dashboard-train-groups-v1';
let trainBuilt = null;
let trainElements = null;
function buildTrain(manifest) {
    let groupsOpen = {};
    try {
        groupsOpen = JSON.parse(storageGet(trainGroupsKey)) || {};
    } catch {
        groupsOpen = {};
    }
    const groupNames = [...new Set(manifest.stages.map((stage) => stage.group))];
    const rows = new Map();
    const groups = groupNames.map((name) => {
        const element = make('details', 'phase');
        element.open = groupsOpen[name] ?? true;
        element.addEventListener('toggle', () => {
            groupsOpen[name] = element.open;
            storageSet(trainGroupsKey, JSON.stringify(groupsOpen));
        });
        const summary = make('summary');
        const head = make('div', 'pipeline-head');
        const facts = make('span', 'facts');
        head.append(make('h3', '', name), facts);
        const bar = make('div', 'bar');
        const fill = make('div');
        bar.append(fill);
        const now = make('div', 'phase-now');
        const count = make('div', 'phase-count');
        summary.append(head, bar, now, count);
        const list = make('ul', 'steps');
        for (const stage of manifest.stages.filter((item) => item.group === name)) {
            const item = make('li', 'step');
            const icon = make('span', 'icon');
            const title = make('span', 'stage-name', stage.stage);
            const percent = make('span', 'facts');
            const rowBar = make('div', 'bar');
            const rowFill = make('div');
            rowBar.append(rowFill);
            const line = make('div', 'progress-line');
            item.append(icon, title, percent, rowBar, line);
            list.append(item);
            rows.set(stage.stage, {item, icon, percent, 'fill': rowFill, line});
        }
        element.append(summary, list);
        return {name, element, facts, fill, now, count};
    });
    document.getElementById('trainStages').replaceChildren(...groups.map((group) => group.element));

    // The gates - each the stage that produces it
    const gateSpecs = [
        {'key': 'heteronyms', 'title': 'Heteronym relabeling reproduces data/nn/g2p-relabel.json', 'stage': 'heteronyms'},
        {'key': 'g2p', 'title': 'G2P check - the retrained context G2P against lib/models/g2pModel.js', 'stage': 'g2p'},
        {'key': 'teacher', 'title': 'The teacher on the listener panel', 'stage': 'teacher-eval'},
        {'key': 'student', 'title': 'The student on the listener panel', 'stage': 'student-eval'},
        {'key': 'pick', 'title': 'The gate - picovox_train.distill pick: the student within the teacher\'s tolerance',
            'stage': 'student-eval'}
    ];
    const gates = new Map();
    document.getElementById('trainGates').replaceChildren(...gateSpecs.map((spec) => {
        const item = make('li', 'step');
        const icon = make('span', 'icon');
        const percent = make('span', 'facts');
        const line = make('div', 'progress-line');
        item.append(icon, make('span', '', spec.title), percent, line);
        gates.set(spec.key, {...spec, item, icon, percent, line});
        return item;
    }));
    trainElements = {groups, rows, gates, 'cards': null};
}


// A gate's state - {status (waiting, running, done - passed, failed), label, line}
function gateState(gate, data) {
    const stageState = data.states.get(gate.stage);
    const waitingFor = {'status': stageState.status === 'running' ? 'running' : 'waiting', 'label': stageState.status === 'skipped' ?
        'not in this run' : 'waiting', 'line': `when the ${gate.stage} stage is done (${stageState.status})`};
    const panel = (result) => {
        const measures = result.panel || {};
        return `UTMOS ${fixed3(measures.mos)} · WER ${Number.isFinite(measures.wer) ? measures.wer.toFixed(2) : '-'}% · ` +
            `similarity ${fixed3(measures.similarity ?? 1)} · ${result.parameters?.toLocaleString()} parameters · module ` +
            `${Math.round((result.size?.moduleBytes ?? 0) / 1024)} KB (${Math.round((result.size?.moduleGzipBytes ?? 0) / 1024)} KB gzip)`;
    };
    const {gates} = data;
    if (gate.key === 'heteronyms' || gate.key === 'g2p') {
        const line = gate.key === 'g2p' ? gates.g2pCheck : gates.heteronyms;
        if (line === null) {
            return stageState.status === 'failed' ? {'status': 'failed', 'label': 'failed', 'line': 'the stage failed'} : waitingFor;
        }
        const failed = /FAILED|differ/.test(line) && !/within the tolerances/.test(line);
        const folds = gate.key === 'g2p' && gates.g2pFolds ? ` · held-out context: ${gates.g2pFolds}` : '';
        return {'status': failed ? 'failed' : 'done', 'label': failed ? 'failed' : 'passed', 'line': `${line}${folds}`};
    }
    if (gate.key === 'teacher' || gate.key === 'student') {
        const result = gates[gate.key];
        return result === null ? waitingFor : {'status': 'done', 'label': 'scored', 'line': panel(result)};
    }
    const {summary} = gates;
    if (summary === null) {
        return waitingFor;
    }
    const row = (summary.rows || []).find((item) => item.name === 'student');
    const within = row?.within ? Object.entries(row.within).map(([key, value]) => `${key} ${value ? '✓' : '✕'}`).join(' · ') : '';
    const tolerance = summary.tolerance ? `tolerance: UTMOS -${summary.tolerance.mos}, WER +${summary.tolerance.wer}, similarity ` +
        `≥ ${summary.tolerance.similarity}` : '';
    const passed = summary.chosen !== null && summary.chosen !== undefined;
    return {'status': passed ? 'done' : 'failed', 'label': passed ? 'passed' : 'failed',
        'line': `chosen: ${summary.chosen ?? 'none'} · ${within} · ${tolerance}`};
}


// The Train tab's curves beyond the main six (the Runs tab has them all) - the critic's, the formant bandwidths, the
// emotional speech's spectral loss, the validation score, and distillation's
const trainChartExtras = ['score', 'critic', 'adversarial', 'b1', 'cremad.spectral', 'frames', 'features', 'durationErrorFrames'];


// Select a make train run's curves - remembered per browser
function selectTrainRun(name) {
    trainSelectedRun = name;
    storageSet(trainRunKey, name);
    renderTrain(true);
}


// Render the Train tab (updated in place; the charts drawn while it's shown)
function renderTrain(force = false) {
    const data = trainData;
    if (data === undefined) {
        return;
    }
    if (data === null) {
        document.getElementById('trainFacts').textContent = 'no stage manifest (data/nn/train-stages.json)';
        return;
    }
    if (trainBuilt !== trainManifestText) {
        trainBuilt = trainManifestText;
        buildTrain(data.manifest);
    }
    const {manifest, states, plan} = data;

    // The summary - the run's state, its start, the ETA, and the critical path
    const elapsed = Number.isFinite(data.start) ? (data.end ?? data.now) - data.start : NaN;
    const left = data.status === 'finished' ? 0 : plan.eta - data.now;
    const facts = [data.status];
    if (Number.isFinite(data.start)) {
        facts.push(`started ${clockTime(data.start)} (${formatDuration(elapsed / 1000)}${data.end ? '' : ' ago'})`);
    }
    if (data.status === 'finished') {
        facts.push(`finished ${clockTime(data.end)}`);
    } else {
        facts.push(`ETA ${clockTime(plan.eta)} (${formatDuration(left / 1000)} left)`);
    }
    facts.push(data.config.dir);
    document.getElementById('trainFacts').textContent = facts.join(' · ');
    const fraction = Number.isFinite(elapsed) && elapsed + left > 0 ? elapsed / (elapsed + left) : 0;
    document.getElementById('trainBar').style.width = `${(100 * fraction).toFixed(1)}%`;
    const pathElement = document.getElementById('trainPath');
    const pathStages = plan.path.map((name) => `${stageIcons[states.get(name).status]} ${name}`).join(' → ');
    const pathText = `critical path ${Math.round(100 * fraction)}%: ${pathStages}`;
    if (pathElement.textContent !== pathText) {
        pathElement.textContent = pathText;
    }
    const running = manifest.stages.filter((stage) => ['running', 'queued'].includes(states.get(stage.stage).status));
    setTabFact('tabFactTrain', data.status === 'running' ?
        `${running.map((stage) => stage.stage).join(', ')} · ETA ${clockTime(plan.eta)}` : data.status);

    // The stages and their groups
    for (const stage of manifest.stages) {
        const state = states.get(stage.stage);
        const {item, icon, percent, fill, line} = trainElements.rows.get(stage.stage);
        const progress = stageProgress(stage, state, data);
        item.className = `step ${state.status}`;
        icon.textContent = stageIcons[state.status];
        percent.textContent = progress.label;
        fill.style.width = `${(100 * (progress.fraction ?? 0)).toFixed(1)}%`;
        fill.classList.toggle('indeterminate', state.status === 'running' && progress.fraction === null);
        const text = stageLine(stage, state, data);
        if (line.textContent !== text) {
            line.textContent = text;
            line.title = text;
        }
    }
    for (const group of trainElements.groups) {
        const members = manifest.stages.filter((stage) => stage.group === group.name && states.get(stage.stage).status !== 'skipped');
        const total = members.reduce((sum, stage) => sum + stage.minutes, 0);
        const stageFraction = (stage) => stageProgress(stage, states.get(stage.stage), data).fraction ?? 0;
        const done = members.reduce((sum, stage) => sum + stage.minutes * stageFraction(stage), 0);
        const doneCount = members.filter((stage) => states.get(stage.stage).status === 'done').length;
        const active = members.filter((stage) => ['running', 'queued', 'failed', 'stopped'].includes(states.get(stage.stage).status));
        let groupStatus = 'waiting';
        if (members.length === 0 || doneCount === members.length) {
            groupStatus = 'done';
        } else if (active.length !== 0 || doneCount !== 0) {
            groupStatus = 'running';
        }
        group.element.className = `phase ${groupStatus}`;
        const groupFraction = total ? done / total : 1;
        const groupFacts = `${Math.round(100 * groupFraction)}% of ~${formatDuration(total * 60)}`;
        group.facts.textContent = members.length === 0 ? 'not in this run' : groupFacts;
        group.fill.style.width = `${(100 * groupFraction).toFixed(1)}%`;
        const nowText = active.map((stage) => `${stageIcons[states.get(stage.stage).status]} ${stage.stage} · ` +
            `${stageProgress(stage, states.get(stage.stage), data).label}`).join('   ') || (groupStatus === 'done' ? 'complete' : '');
        group.now.textContent = nowText;
        group.now.title = nowText;
        const all = manifest.stages.filter((stage) => stage.group === group.name).length;
        const skipped = all !== members.length ? ` (${all - members.length} not in this run)` : '';
        group.count.textContent = `${doneCount} of ${members.length} stages done${skipped}`;
    }

    // The gates
    for (const gate of trainElements.gates.values()) {
        const state = gateState(gate, data);
        gate.item.className = `step ${state.status}`;
        gate.icon.textContent = stageIcons[state.status];
        gate.percent.textContent = state.label;
        if (gate.line.textContent !== state.line) {
            gate.line.textContent = state.line;
            gate.line.title = state.line;
        }
    }

    // The training runs' cards (all listed from the start), the selected one's curves - compared with its reference
    const trainRunList = manifest.stages.filter((stage) => stage.run).map((stage) => trainRuns.get(stage.run));
    if (!trainRuns.has(trainSelectedRun)) {
        const [live] = trainRunList.filter((run) => runStatus(run) === 'live');
        trainSelectedRun = (live ?? trainRunList.filter((run) => !run.waiting).pop() ?? trainRunList[0])?.name ?? null;
    }
    document.getElementById('trainRuns').replaceChildren(...trainRunList.map((run) => safeRunCard(run, selectTrainRun, trainSelectedRun)));
    if (currentTab === 'train') {
        const run = trainRuns.get(trainSelectedRun) ?? null;
        const referenceName = data.config.compare?.[run?.name];
        const reference = referenceName ? runs.get(referenceName) ?? null : null;
        drawRunCharts(document.getElementById('trainCharts'), document.getElementById('trainChartsTitle'),
            run && !run.waiting ? run : null, reference, null, force, trainChartExtras);
        renderUtilization(data, force);
    }
}


// Render the utilization - the current and the run's mean numbers, and the chart (redrawn every 10 seconds, or forced)
let utilDrawn = 0;
function renderUtilization(data, force) {
    const {samples} = data;
    const last = samples[samples.length - 1];
    const facts = [];
    if (last) {
        const memory = Number.isFinite(last.mem) ? ` · memory ${Math.round(last.mem)}%` : '';
        const ago = formatDuration((data.now - last.time) / 1000);
        facts.push(`now CPU ${Math.round(last.cpu)}% · GPU ${Math.round(last.gpu)}%${memory} (${ago} ago)`);
        const mean = trainModel.meanUtilization(samples, -Infinity, Infinity);
        // GPU idle time - the intervals between samples starting under 10% (gaps over 3 minutes aren't counted)
        let idle = 0;
        let sampled = 0;
        samples.slice(1).forEach((sample, index) => {
            const interval = sample.time - samples[index].time;
            if (interval <= 180000) {
                sampled += interval;
                idle += samples[index].gpu < 10 ? interval : 0;
            }
        });
        facts.push(`run mean CPU ${Math.round(mean.cpu)}% · GPU ${Math.round(mean.gpu)}% · GPU idle ${formatDuration(idle / 1000)} of ` +
            `${formatDuration(sampled / 1000)} sampled`);
    } else {
        facts.push('no samples yet - make train writes TRAIN_DIR/util.log');
    }
    document.getElementById('utilFacts').textContent = facts.join(' · ');
    if (force || Date.now() - utilDrawn > 10000) {
        utilDrawn = Date.now();
        drawUtilization(document.getElementById('utilChart'), data);
    }
}


// Draw the utilization chart - the CPU and GPU % over the run (the GPU idle shaded), and below, each stage's band over
// its time (the forecast faded)
function drawUtilization(canvas, data) {
    const {manifest, states, plan, samples} = data;
    const laneHeight = 13;
    const linesHeight = 120;
    const pad = {'left': 104, 'right': 10, 'top': 8, 'bottom': 22};
    const height = pad.top + linesHeight + 10 + manifest.stages.length * laneHeight + pad.bottom;
    canvas.style.height = `${height}px`;
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const context = canvas.getContext('2d');
    context.scale(ratio, ratio);
    const styles = getComputedStyle(document.documentElement);
    const color = (name) => styles.getPropertyValue(name).trim();
    const classColor = {'gpu': color('--accent'), 'cpu': color('--series-3'), 'network': color('--muted')};
    const start = Number.isFinite(data.start) ? data.start : data.now;
    const end = Math.max(data.end ?? data.now, data.status === 'finished' ? 0 : plan.eta, start + 60000);
    const plotX = (time) => pad.left + (time - start) / (end - start) * (width - pad.left - pad.right);
    const plotY = (percent) => pad.top + (1 - percent / 100) * linesHeight;

    // The grid, the GPU idle shading, and the lines
    context.font = '10px ui-monospace, Menlo, monospace';
    context.strokeStyle = color('--rule');
    context.fillStyle = color('--muted');
    context.lineWidth = 1;
    for (const percent of [0, 50, 100]) {
        context.beginPath();
        context.moveTo(pad.left, plotY(percent));
        context.lineTo(width - pad.right, plotY(percent));
        context.stroke();
        context.fillText(`${percent}%`, pad.left - 34, plotY(percent) + 3);
    }
    context.fillStyle = `${color('--warn')}33`;
    samples.slice(1).forEach((sample, index) => {
        const previous = samples[index];
        if (previous.gpu < 10 && sample.time - previous.time <= 180000) {
            context.fillRect(plotX(previous.time), pad.top, plotX(sample.time) - plotX(previous.time), linesHeight);
        }
    });
    for (const [key, name] of [['cpu', '--series-3'], ['gpu', '--accent']]) {
        context.strokeStyle = color(name);
        context.lineWidth = 1.6;
        context.beginPath();
        samples.forEach((sample, index) => {
            const gap = index === 0 || sample.time - samples[index - 1].time > 180000;
            if (gap) {
                context.moveTo(plotX(sample.time), plotY(sample[key]));
            } else {
                context.lineTo(plotX(sample.time), plotY(sample[key]));
            }
        });
        context.stroke();
    }

    // The stages' bands - done and running solid (failed red), the rest of a running stage and the waiting ones faded
    const lanesTop = pad.top + linesHeight + 10;
    manifest.stages.forEach((stage, index) => {
        const state = states.get(stage.stage);
        const top = lanesTop + index * laneHeight;
        context.fillStyle = state.status === 'skipped' ? color('--rule') : color('--muted');
        context.fillText(stage.stage, 4, top + 9);
        const band = (from, to, fill) => {
            if (Number.isFinite(from) && Number.isFinite(to) && to > from) {
                context.fillStyle = fill;
                context.fillRect(plotX(from), top + 2, Math.max(1.5, plotX(to) - plotX(from)), laneHeight - 4);
            }
        };
        const solid = state.status === 'failed' ? color('--bad') : classColor[stage.on];
        if (state.start !== null) {
            band(state.start, state.end ?? data.now, solid);
        }
        const finish = plan.finishes.get(stage.stage);
        if (state.status !== 'done' && state.status !== 'failed' && finish !== undefined) {
            const from = state.start !== null ? data.now : finish - stage.minutes * 60000;
            band(Math.max(from, data.now), finish, `${classColor[stage.on]}44`);
        }
    });

    // The time axis and now
    context.strokeStyle = color('--muted');
    context.setLineDash([3, 3]);
    context.beginPath();
    context.moveTo(plotX(data.now), pad.top);
    context.lineTo(plotX(data.now), height - pad.bottom);
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = color('--muted');
    const span = end - start;
    const stepMs = [15, 30, 60, 120, 240].map((minutes) => minutes * 60000).find((step) => span / step <= 8) ?? 480 * 60000;
    for (let tick = Math.ceil(start / stepMs) * stepMs; tick <= end; tick += stepMs) {
        const label = clockTime(tick);
        const xPixel = plotX(tick);
        if (xPixel - pad.left > 60 && width - pad.right - xPixel > 60) {
            context.fillText(label, xPixel - context.measureText(label).width / 2, height - 6);
        }
    }
    context.fillText(clockTime(start), pad.left, height - 6);
    const endLabel = clockTime(end);
    context.fillText(endLabel, width - pad.right - context.measureText(endLabel).width, height - 6);
}


// The utilization chart's legend
document.getElementById('utilLegend').append(...[
    ['CPU busy %', '--series-3'], ['GPU utilization %', '--accent'], ['GPU idle (shaded)', '--warn'], ['GPU stage', '--accent'],
    ['CPU stage', '--series-3'], ['network stage', '--muted']
].map(([label, name]) => {
    const swatch = make('span', '', label);
    swatch.style.setProperty('--swatch', `var(${name})`);
    return swatch;
}));


// Tabs - the selected tab is in the URL's hash (and remembered per browser); the Listen tab marks new checkpoints until
// it's viewed
const tabNames = ['train', 'listen', 'runs', 'logs', 'archive'];
const tabAliases = {'progress': 'archive', 'training': 'runs'};
const tabKey = 'training-dashboard-tab-v1';
const heardKey = 'training-dashboard-heard-v1';
let currentTab = 'train';
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
    const wanted = tabAliases[name] ?? name;
    currentTab = tabNames.includes(wanted) ? wanted : 'train';
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
    // The charts draw at their visible size
    if (currentTab === 'runs') {
        renderCharts(true);
    }
    if (currentTab === 'train') {
        renderTrain(true);
    }
    if (currentTab === 'archive') {
        renderPipeline(lastLogs);
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


// The newest checkpoint's key, and marking it and the newest make train samples heard (the Listen tab's dot)
let newestCheckpoint = null;
function listenKey() {
    if (newestTrainSamples === null && newestCheckpoint === null) {
        return null;
    }
    return `${newestTrainSamples?.key ?? ''} | ${newestCheckpoint ?? ''}`;
}
function markHeard() {
    if (listenKey() !== null) {
        storageSet(heardKey, listenKey());
    }
    document.getElementById('tabDotListen').hidden = true;
}
function updateListenDot() {
    if (currentTab === 'listen') {
        markHeard();
    } else {
        document.getElementById('tabDotListen').hidden = listenKey() === null || storageGet(heardKey) === listenKey();
    }
}


// Update the tabs' facts - the overall progress, the newest checkpoint, the live run, and the active logs
function setTabFact(id, text) {
    const element = document.getElementById(id);
    if (element.textContent !== text) {
        element.textContent = text;
    }
}


// Reload the page when the dashboard's own files change (checked every 5 seconds), so it never runs stale code
const codeFiles = ['index.html', 'dashboard.js', 'train.js'].map((name) => new URL(`tools/dashboard/${name}`, rootUrl));
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
// The first tab - the URL's, else the Train tab while a make train runs (decided on the first refresh), else the
// remembered one
const hashTab = window.location.hash.slice(1);
selectTab(hashTab || storageGet(tabKey) || 'train');
let firstRefresh = true;


// Refresh - then again a second later, so refreshes never overlap
async function refresh() {
    try {
        await loadPipeline();
        await refreshRuns();
        renderRuns();
        renderCharts();
        trainData = await loadTrain();
        if (firstRefresh && hashTab === '' && trainData?.status === 'running') {
            selectTab('train');
        }
        firstRefresh = false;
        renderTrain();
        const files = await findFiles();
        const logs = await loadLogs([...trainData?.logFiles ?? [], ...files]);
        lastLogs = logs;
        await renderPipeline(logs);
        renderLogs(logs);
        renderSamples(files);
        await renderCheckpointSamples();
        await renderTrainSamples();
        renderNextCheckpoint();
        updatedElement.textContent = `Updated ${new Date().toLocaleTimeString()} · every second`;
    } finally {
        setTimeout(refresh, refreshMs);
    }
}
refresh();
window.addEventListener('resize', () => {
    renderCharts(true);
    renderTrain(true);
});
