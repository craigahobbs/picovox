// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// The voice lab - PicoVox synthesized in the page (the package's lib/, staged beside this script as picovox/lib/ by make
// voicelab) with adjustable settings: edits of the network's parameter frames (each sentence's), the rendering's sample
// rate, and a stereo room at playback. Two views: Claude's blind questions (an artifact's db store - "questions" Claude
// writes, "answers" the page writes), and free exploration with saved settings ("presets"). The models are the package's
// (shipped), or make train's voice with the package's G2P (trained - the voice staged as trained/ by make train-check,
// when it's there). Served by make train-check's blind A/B (tools/trainab.js), the questions are its trials and the
// answers are posted to it (its api/), not Claude's.
//
// A question: {order, type ('ab' - choose a version, or 'Same'; 'yesno'; 'choice' - one of options), prompt, text, voice,
// model, versions: [{name, voice, model, settings}] (played in a shuffled order - a version's voice and model, else the
// question's, else shipped), allowSame, options, hint, retired, repeatOf}. An answer:
// {question, choice, versionIndex, version (the chosen version's name), order (the versions as played), plays, note,
// answeredAt}.

import {
    createSentenceRenderer, defaultSampleRate, paragraphPause, sentenceFrames, sentencePause, voiceTable
} from './picovox/lib/synthesis.js';
import {speechText, splitSentences} from './picovox/lib/sentences.js';
import {NAMES} from './picovox/lib/params.js';
import {loadG2p} from './picovox/lib/g2p.js';
import {loadPackagedModel} from './picovox/lib/package.js';
import {setPronouncer} from './picovox/lib/lexicon.js';


// The settings - frame edits (scale, add, or smooth - a centered moving average, ms - named parameters), the
// rendering's sample rate, and the playback's stereo room - each from its neutral value
const SETTINGS = [
    {'key': 'jitter', 'label': 'Jitter', 'kind': 'scale', 'params': ['jitter'], 'min': 0, 'max': 5, 'step': 0.25, 'unit': '×'},
    {'key': 'shimmer', 'label': 'Shimmer', 'kind': 'scale', 'params': ['shimmer'], 'min': 0, 'max': 5, 'step': 0.25, 'unit': '×'},
    {'key': 'ah', 'label': 'Aspiration (breath)', 'kind': 'add', 'params': ['ah'], 'min': -12, 'max': 12, 'step': 1, 'unit': ' dB'},
    {'key': 'af', 'label': 'Frication', 'kind': 'add', 'params': ['af'], 'min': -12, 'max': 12, 'step': 1, 'unit': ' dB'},
    {'key': 'rd', 'label': 'Pulse softness (Rd)', 'kind': 'scale', 'params': ['rd'], 'min': 0.7, 'max': 1.5, 'step': 0.05, 'unit': '×'},
    {'key': 'shelf', 'label': 'Source shelf', 'kind': 'add', 'params': ['shelf'], 'min': -10, 'max': 10, 'step': 1, 'unit': ' dB'},
    {'key': 'bandwidth', 'label': 'Formant bandwidths', 'kind': 'scale', 'params': ['b1', 'b2', 'b3', 'b4', 'b5'], 'min': 0.5, 'max': 2,
        'step': 0.05, 'unit': '×'},
    {'key': 'smoothFormants', 'label': 'Formant smoothing', 'kind': 'smooth', 'params': ['f1', 'f2', 'f3', 'b1', 'b2', 'b3'], 'min': 0,
        'max': 60, 'step': 5, 'unit': ' ms'},
    {'key': 'smoothF0', 'label': 'Pitch smoothing', 'kind': 'smooth', 'params': ['f0'], 'min': 0, 'max': 60, 'step': 5, 'unit': ' ms'},
    {'key': 'sampleRate', 'label': 'Sample rate', 'kind': 'render', 'neutral': 24000, 'min': 24000, 'max': 44100, 'step': 20100,
        'unit': ' Hz'},
    {'key': 'roomMix', 'label': 'Stereo room (mix)', 'kind': 'playback', 'neutral': 0, 'min': 0, 'max': 0.4, 'step': 0.05, 'unit': '',
        'percent': true},
    {'key': 'roomSize', 'label': 'Stereo room (size)', 'kind': 'playback', 'neutral': 0.6, 'min': 0.2, 'max': 1.6, 'step': 0.1,
        'unit': ' s'}
];
const NEUTRAL = Object.fromEntries(SETTINGS.map((setting) => [setting.key, setting.neutral ?? (setting.kind === 'scale' ? 1 : 0)]));


// The exploration's sentences
const TEXTS = [
    'The Mole had been working very hard all the morning, spring-cleaning his little home.',
    'It was the best of times, it was the worst of times, it was the age of wisdom.',
    'Call me Ishmael. Some years ago, never mind how long precisely, I thought I would sail about a little.',
    '"How are you?" he said cordially, gripping my hand with a strength for which I should hardly have given him credit.'
];


// The status line
const statusElement = document.getElementById('status');
function setStatus(text, kind = '') {
    statusElement.textContent = text;
    statusElement.className = `status ${kind}`;
}


// The models - the package's (shipped), and make train's voice with the package's G2P (trained - so a comparison is the
// voice's alone), each loaded once when first used
const MODELS = {
    'shipped': {
        'label': 'Shipped models', 'g2p': () => import('./picovox/lib/models/g2pModel.js'),
        'voice': () => import('./picovox/lib/models/voiceModel.js')
    },
    'trained': {
        'label': 'Trained voice (make train)', 'g2p': () => import('./picovox/lib/models/g2pModel.js'),
        'voice': () => import('./trained/voiceModel.js')
    }
};
const loadedModels = new Map();
function loadModel(name = 'shipped') {
    if (!loadedModels.has(name)) {
        const {g2p, voice} = MODELS[name];
        const loading = Promise.all([g2p().then((module) => loadG2p(module.g2pModel)), voice().then((module) => loadPackagedModel(module))])
            .then(([pronounce, model]) => ({pronounce, model}));
        loading.catch(() => loadedModels.delete(name));
        loadedModels.set(name, loading);
    }
    return loadedModels.get(name);
}
let modelReady = false;


// A sentence's frames edited by settings - scaled, added to, and smoothed (5 ms frames)
function editFrames(frames, settings) {
    const out = frames.map((frame) => Float64Array.from(frame));
    for (const setting of SETTINGS) {
        const value = settings[setting.key] ?? NEUTRAL[setting.key];
        if (value === NEUTRAL[setting.key] || !setting.params) {
            continue;
        }
        for (const name of setting.params) {
            const index = NAMES.indexOf(name);
            if (setting.kind === 'scale') {
                out.forEach((frame) => {
                    frame[index] *= value;
                });
            } else if (setting.kind === 'add') {
                out.forEach((frame) => {
                    frame[index] += value;
                });
            } else {
                const half = Math.round(value / 10);
                const column = out.map((frame) => frame[index]);
                out.forEach((frame, row) => {
                    const window = column.slice(Math.max(0, row - half), row + half + 1);
                    frame[index] = window.reduce((sum, item) => sum + item, 0) / window.length;
                });
            }
        }
    }
    return out;
}


// Render text in a voice with a model and settings - {samples (Float32Array), sampleRate}, cached (the playback settings
// aside). The G2P is the lexicon's pronouncer, so it's the model's for the rendering (synchronous, after the load).
const renders = new Map();
async function render(voice, modelName, text, settings, seed = 1000) {
    const sampleRate = settings.sampleRate ?? defaultSampleRate;
    const rendered = Object.fromEntries(Object.entries(settings).filter(
        ([name]) => SETTINGS.find((setting) => setting.key === name)?.kind !== 'playback'
    ));
    const key = JSON.stringify([voice, modelName, text, rendered, seed]);
    if (!renders.has(key)) {
        const {pronounce, model} = await loadModel(modelName);
        await new Promise((resolve) => {
            setTimeout(resolve, 0);
        });
        setPronouncer(pronounce);
        const blocks = [];
        for (const sentence of splitSentences(text)) {
            const {chunks, eq} = sentenceFrames(model, speechText(sentence.text), {voice});
            const frames = editFrames([...chunks].flat(), settings);
            const pause = sentence.paragraphStart ? paragraphPause : sentencePause;
            const onBlock = (block) => blocks.push(block.slice());
            const renderer = createSentenceRenderer({onBlock, eq, 'seed': seed + sentence.index, pause, sampleRate});
            if (frames.length !== 0) {
                renderer.push(frames);
            }
            renderer.end();
        }
        const samples = new Float32Array(blocks.reduce((sum, block) => sum + block.length, 0));
        blocks.reduce((offset, block) => {
            samples.set(block, offset);
            return offset + block.length;
        }, 0);
        renders.set(key, {samples, sampleRate});
    }
    return renders.get(key);
}


// A stereo room's impulse response at a context's sample rate - each channel its own decaying noise (seeded, so a room
// always sounds the same), 60 dB down at size seconds after an 8 ms pre-delay, darkened (a one-pole low-pass), unit energy
const rooms = new Map();
function roomImpulse(context, size) {
    const key = `${context.sampleRate}-${size}`;
    if (!rooms.has(key)) {
        const rate = context.sampleRate;
        const length = Math.round((size + 0.008) * rate);
        const buffer = context.createBuffer(2, length, rate);
        for (let channel = 0; channel < 2; channel++) {
            let state = 0x9e3779b9 + channel * 0x85ebca6b;
            const random = () => {
                state = (state + 0x6d2b79f5) | 0;
                let mixed = Math.imul(state ^ (state >>> 15), state | 1);
                mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
                return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296 * 2 - 1;
            };
            const data = buffer.getChannelData(channel);
            let smooth = 0;
            let energy = 0;
            for (let index = Math.round(0.008 * rate); index < length; index++) {
                smooth = 0.55 * smooth + 0.45 * random();
                data[index] = smooth * Math.exp(-6.91 * (index / rate - 0.008) / size);
                energy += data[index] * data[index];
            }
            const scale = 1 / Math.sqrt(energy);
            for (let index = 0; index < length; index++) {
                data[index] *= scale;
            }
        }
        rooms.set(key, buffer);
    }
    return rooms.get(key);
}


// Mark a button rendering (disabled, "Rendering…"), or restore it with its label
function markRendering(button, label = null) {
    button.disabled = label === null;
    button.textContent = label ?? 'Rendering…';
}


// Play text in a voice with a model and settings (stopping whatever plays), marking the button while it plays - the dry
// voice centered, and a stereo room at its mix (the dry level lowered so the loudness stays the same)
let audio = null;
let current = null;
async function play(button, voice, modelName, text, settings) {
    audio ??= new AudioContext();
    if (audio.state === 'suspended') {
        await audio.resume();
    }
    current?.source.stop();
    const label = button.textContent;
    markRendering(button);
    let rendered;
    try {
        rendered = await render(voice, modelName, text, settings);
    } catch (error) {
        setStatus(`Couldn't synthesize: ${error.message}`, 'warn');
        return;
    } finally {
        markRendering(button, label);
    }
    const buffer = audio.createBuffer(1, rendered.samples.length, rendered.sampleRate);
    buffer.copyToChannel(rendered.samples, 0);
    const source = audio.createBufferSource();
    source.buffer = buffer;
    const mix = settings.roomMix ?? 0;
    const dry = audio.createGain();
    dry.gain.value = Math.sqrt(1 - mix * mix);
    source.connect(dry).connect(audio.destination);
    if (mix > 0) {
        const room = audio.createConvolver();
        room.normalize = false;
        room.buffer = roomImpulse(audio, settings.roomSize ?? NEUTRAL.roomSize);
        const wet = audio.createGain();
        wet.gain.value = mix;
        source.connect(room).connect(wet).connect(audio.destination);
    }
    current?.button.classList.remove('playing');
    current = {source, button};
    button.classList.add('playing');
    source.onended = () => {
        button.classList.remove('playing');
        if (current?.source === source) {
            current = null;
        }
    };
    source.start();
}


// The tabs
const tabs = {'questions': document.getElementById('tab-questions'), 'explore': document.getElementById('tab-explore')};
function showTab(name) {
    for (const [key, tab] of Object.entries(tabs)) {
        tab.setAttribute('aria-selected', String(key === name));
        document.getElementById(`view-${key}`).hidden = key !== name;
    }
}
tabs.questions.addEventListener('click', () => showTab('questions'));
tabs.explore.addEventListener('click', () => showTab('explore'));


// A setting's value for display
function format(setting, value) {
    if (setting.percent) {
        return `${Math.round(value * 100)}%`;
    }
    return setting.kind === 'add' && value > 0 ? `+${value}${setting.unit}` : `${value}${setting.unit}`;
}


// Settings described - the changed ones
function describe(settings) {
    const changed = SETTINGS.filter((setting) => (settings[setting.key] ?? NEUTRAL[setting.key]) !== NEUTRAL[setting.key]);
    return changed.length ? changed.map((setting) => `${setting.label} ${format(setting, settings[setting.key])}`).join(', ') : 'baseline';
}


// Explore - the voice, the models (the trained ones when they're staged), the sentence, and a slider per setting
const voiceSelect = document.getElementById('explore-voice');
const modelSelect = document.getElementById('explore-model');
modelSelect.append(new Option(MODELS.shipped.label, 'shipped'));
fetch('./trained/voiceModel.js').then((response) => {
    if (response.ok) {
        modelSelect.append(new Option(MODELS.trained.label, 'trained'));
        modelSelect.hidden = false;
    }
}, () => null);
for (const voice of voiceTable) {
    voiceSelect.append(new Option(`${voice.label} (${voice.sex})`, voice.key));
}
voiceSelect.value = 'female';
const textSelect = document.getElementById('explore-text');
TEXTS.forEach((text, index) => textSelect.append(new Option(text.length > 60 ? `${text.slice(0, 57)}…` : text, String(index))));
const explore = {...NEUTRAL};
const inputs = new Map();
const outputs = new Map();
for (const setting of SETTINGS) {
    const row = document.createElement('div');
    row.className = 'slider';
    const label = document.createElement('label');
    label.htmlFor = `slider-${setting.key}`;
    label.textContent = setting.label;
    const input = document.createElement('input');
    input.type = 'range';
    input.id = `slider-${setting.key}`;
    Object.assign(input, {'min': setting.min, 'max': setting.max, 'step': setting.step, 'value': NEUTRAL[setting.key]});
    const output = document.createElement('output');
    output.htmlFor = input.id;
    output.textContent = format(setting, NEUTRAL[setting.key]);
    input.addEventListener('input', () => {
        explore[setting.key] = Number(input.value);
        output.textContent = format(setting, explore[setting.key]);
    });
    inputs.set(setting.key, input);
    outputs.set(setting.key, output);
    row.append(label, input, output);
    document.getElementById('sliders').append(row);
}
function setExplore(settings) {
    for (const setting of SETTINGS) {
        explore[setting.key] = settings[setting.key] ?? NEUTRAL[setting.key];
        inputs.get(setting.key).value = explore[setting.key];
        outputs.get(setting.key).textContent = format(setting, explore[setting.key]);
    }
}
const exploreText = () => TEXTS[Number(textSelect.value)];
document.getElementById('explore-play').addEventListener(
    'click', (event) => play(event.currentTarget, voiceSelect.value, modelSelect.value, exploreText(), {...explore})
);
document.getElementById('explore-baseline').addEventListener(
    'click', (event) => play(event.currentTarget, voiceSelect.value, modelSelect.value, exploreText(), {...NEUTRAL})
);
document.getElementById('explore-reset').addEventListener('click', () => setExplore(NEUTRAL));


// The store (the artifact's db - null without one), Claude's questions, and the answers
let store = null;
let questions = [];
const answers = new Map();
let showing = null;


// An element of a tag and class, with text
function element(tag, className, text = null) {
    const created = document.createElement(tag);
    created.className = className;
    if (text !== null) {
        created.textContent = text;
    }
    return created;
}


// Show the next unanswered question - its versions in a shuffled (blind) order, the choices, a note, and Submit
function renderQuestion() {
    const card = document.getElementById('question');
    const pending = questions.filter((question) => !answers.has(question.id));
    const next = pending[0] ?? null;
    if (next !== null && showing?.id === next.id) {
        return;
    }
    card.replaceChildren();
    showing = next;
    if (next === null) {
        const done = local ? 'The report is in your terminal.' : 'Tell Claude, and new ones will appear here.';
        card.append(element('p', 'empty', questions.length ?
            `All ${questions.length} questions answered - thank you. ${done}` : 'Waiting for Claude\'s questions…'));
        return;
    }
    const order = next.versions.map((version, index) => ({version, index}));
    if (next.shuffle !== false) {
        for (let position = order.length - 1; position > 0; position--) {
            const other = Math.floor(Math.random() * (position + 1));
            [order[position], order[other]] = [order[other], order[position]];
        }
    }
    const letters = 'ABCDEF';
    const plays = new Map();
    const players = element('div', 'row');
    order.forEach((item, position) => {
        const button = element('button', 'btn play', next.versions.length === 1 ? 'Play' : `Play ${letters[position]}`);
        button.type = 'button';
        button.id = `play-${next.id}-${position}`;
        button.addEventListener('click', () => {
            plays.set(position, (plays.get(position) ?? 0) + 1);
            const modelName = item.version.model ?? next.model ?? 'shipped';
            play(button, item.version.voice ?? next.voice, modelName, next.text, item.version.settings ?? {});
        });
        players.append(button);
    });

    // The choices - a version's letter (or Same), yes or no, or the question's options
    let choices = [...order.map((_item, position) => letters[position]), ...(next.allowSame === false ? [] : ['Same'])];
    if (next.type === 'yesno') {
        choices = ['Yes', 'No'];
    } else if (next.type === 'choice' && next.options) {
        choices = next.options;
    }
    let chosen = null;
    const submit = element('button', 'btn primary', 'Submit');
    submit.type = 'button';
    submit.disabled = true;
    const choiceRow = element('div', 'row');
    const buttons = choices.map((choice) => {
        const button = element('button', 'btn', choice);
        button.type = 'button';
        button.id = `choice-${next.id}-${choice}`;
        button.setAttribute('aria-pressed', 'false');
        button.addEventListener('click', () => {
            chosen = choice;
            buttons.forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
            submit.disabled = false;
        });
        choiceRow.append(button);
        return button;
    });
    const note = element('textarea', '');
    note.id = `note-${next.id}`;
    note.placeholder = 'Anything you hear (optional - dictation works: the globe key twice on a Mac, or the mic on an iPhone keyboard)';
    note.setAttribute('aria-label', 'Note');
    submit.addEventListener('click', async () => {
        if (store === null) {
            return;
        }
        submit.disabled = true;
        // (the answer in the versions' own terms - a version's letter maps back to the version)
        const position = letters.indexOf(chosen);
        const isVersion = next.type === 'ab' && chosen.length === 1 && position >= 0;
        const versionIndex = isVersion ? order[position].index : null;
        const body = {
            'question': next.id, 'choice': chosen, versionIndex, 'version': isVersion ? (next.versions[versionIndex].name ?? null) : null,
            'order': order.map((item) => item.index),
            'plays': Object.fromEntries([...plays].map(([played, count]) => [letters[played], count])),
            'note': note.value, 'answeredAt': new Date().toISOString()
        };
        try {
            await store.doc(`answers/${next.id}`).set(body);
            answers.set(next.id, body);
            setStatus('Answer saved.', 'ok');
            renderQuestion();
        } catch (error) {
            submit.disabled = false;
            setStatus(error?.code === 'invalid_argument' ? 'You can\'t answer on this page (read-only).' :
                'Not saved - the store is unavailable. Try again in a moment.', 'warn');
        }
    });
    const submitRow = element('div', 'row');
    submitRow.append(submit);
    card.append(
        element('p', 'progress', `Question ${questions.length - pending.length + 1} of ${questions.length}`),
        element('p', 'prompt', next.prompt), element('p', 'text', `"${next.text}"`), players,
        element('p', 'hint', next.hint ?? (next.type === 'yesno' ? 'Answer yes or no.' : 'Play each as often as you like, then choose.')),
        choiceRow, note, submitRow
    );
}


// Show the saved settings - each loadable into Explore
function renderPresets(snapshot) {
    const list = document.getElementById('presets');
    list.replaceChildren();
    if (snapshot.empty) {
        list.append(element('p', 'empty', 'None yet.'));
        return;
    }
    for (const doc of snapshot.docs) {
        const preset = doc.data();
        const row = element('div', 'preset');
        const loadButton = element('button', 'btn', 'Load');
        loadButton.type = 'button';
        loadButton.addEventListener('click', () => {
            voiceSelect.value = preset.voice;
            modelSelect.value = [...modelSelect.options].some((option) => option.value === preset.model) ? preset.model : 'shipped';
            textSelect.value = String(preset.textIndex ?? 0);
            setExplore(preset.settings ?? {});
        });
        row.append(loadButton, element('span', 'name', preset.name || '(unnamed)'),
            element('span', 'detail', `${preset.voice}${preset.model === 'trained' ? ' (trained)' : ''}: ` +
                `${describe(preset.settings ?? {})}${preset.note ? ` - ${preset.note}` : ''}`));
        list.append(row);
    }
}


// Save Explore's settings as a preset
document.getElementById('preset-save').addEventListener('click', async () => {
    if (store === null) {
        return;
    }
    const nameInput = document.getElementById('preset-name');
    const noteInput = document.getElementById('preset-note');
    try {
        await store.collection('presets').add({
            'name': nameInput.value.trim(), 'voice': voiceSelect.value, 'model': modelSelect.value, 'textIndex': Number(textSelect.value),
            'settings': {...explore},
            'note': noteInput.value, 'savedAt': new Date().toISOString()
        });
        setStatus('Settings saved.', 'ok');
        nameInput.value = '';
        noteInput.value = '';
    } catch {
        setStatus('Not saved - the store is unavailable.', 'warn');
    }
});


// The status of PicoVox's loading and the store's connection (null while connecting, false without one)
let storeConnected = null;
function showReady() {
    if (!modelReady) {
        return;
    }
    if (storeConnected === null) {
        setStatus('PicoVox ready - connecting to the notes store…');
    } else if (storeConnected) {
        setStatus('Ready.', 'ok');
    } else {
        setStatus('Ready to explore - answers can\'t be saved in this view (open the published page signed in to claude.ai).', 'warn');
    }
}


// The local store - make train-check's blind A/B (tools/trainab.js), its api/ beside the page: its questions and the
// answers so far (once), an answer posted to it, and no presets - the db store's calls the page makes, or null without it
let local = false;
async function localStore() {
    const response = await fetch('./api/questions').catch(() => null);
    if (!response?.ok) {
        return null;
    }
    local = true;
    const localQuestions = await response.json();
    const localAnswers = await (await fetch('./api/answers')).json();
    const snapshot = (items) => ({'empty': items.length === 0, 'docs': items.map(([id, value]) => ({id, 'data': () => value}))});
    const collections = {
        'questions': snapshot(localQuestions.map((question) => [question.id, question])),
        'answers': snapshot(Object.entries(localAnswers)),
        'presets': snapshot([])
    };
    const collection = (name) => ({
        'orderBy': () => collection(name),
        'onSnapshot': (callback) => callback(collections[name]),
        'add': () => Promise.reject(new Error('No presets in the A/B'))
    });
    return {
        collection,
        'doc': (path) => ({
            'set': async (body) => {
                const posted = await fetch(`./api/${path.split('/').map(encodeURIComponent).join('/')}`, {
                    'method': 'POST', 'headers': {'Content-Type': 'application/json'}, 'body': JSON.stringify(body)
                });
                if (!posted.ok) {
                    throw new Error(`Not saved (${posted.status})`);
                }
            }
        })
    };
}


// Start - PicoVox, then the store (live questions, answers, and presets - the A/B's, else the artifact's db)
loadModel().then(() => {
    modelReady = true;
    showReady();
}, (error) => setStatus(`PicoVox failed to load: ${error.message}`, 'warn'));
const db = await localStore() ?? await globalThis.claude?.use?.('db');
storeConnected = Boolean(db);
showReady();
if (db) {
    store = db;
    const stopped = () => setStatus('The store stopped responding - reload to reconnect.', 'warn');
    db.collection('answers').onSnapshot((snapshot) => {
        for (const doc of snapshot.docs) {
            answers.set(doc.id, doc.data());
        }
        renderQuestion();
    }, stopped);
    db.collection('questions').orderBy('order').onSnapshot((snapshot) => {
        questions = snapshot.docs.map((doc) => ({'id': doc.id, ...doc.data()})).filter((question) => !question.retired);
        renderQuestion();
    }, stopped);
    db.collection('presets').orderBy('savedAt', 'desc').onSnapshot(renderPresets, stopped);
}
