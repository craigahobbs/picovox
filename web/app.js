// The browser front end - text, voice, rate, and room controls; synthesis in a Web Worker; playback with Web Audio;
// and WAV download

import {encodeWav} from '../lib/wav.js';


// Elements
const textElement = document.getElementById('text');
const rateElement = document.getElementById('rate');
const rateValueElement = document.getElementById('rateValue');
const roomElement = document.getElementById('room');
const speakButton = document.getElementById('speak');
const stopButton = document.getElementById('stop');
const downloadButton = document.getElementById('download');
const statusElement = document.getElementById('status');


// State - the worker, the current request ({id, key, play, download, chunks, done}), and the playing sources
let worker = null;
let workerReady = false;
let request = null;
let nextRequestId = 1;
let audioContext = null;
let sources = [];
let nextStartTime = 0;


// Show a status message
function setStatus(message, isError = false) {
    statusElement.textContent = message;
    statusElement.classList.toggle('error', isError);
}


// The current settings, and a key identifying the audio they produce
function getSettings() {
    const voice = document.querySelector('input[name="voice"]:checked').value;
    const settings = {'text': textElement.value, voice, 'rate': Number(rateElement.value), 'seed': 1, 'room': roomElement.checked};
    return {settings, 'key': JSON.stringify(settings)};
}


// Update the buttons for the current state
function updateButtons() {
    const busy = request !== null && !request.done;
    stopButton.disabled = !busy && sources.length === 0;
}


// Start (or restart) the synthesis worker - restarting abandons any synthesis in progress
function startWorker() {
    if (worker !== null) {
        worker.terminate();
    }
    workerReady = false;
    worker = new Worker(new URL('worker.js', import.meta.url), {'type': 'module'});
    worker.onmessage = (event) => onWorkerMessage(event.data);
    worker.onerror = (event) => {
        event.preventDefault();
        setStatus(`The synthesis worker failed: ${event.message || 'could not load (serve the page over HTTP)'}`, true);
        request = null;
        updateButtons();
    };
}


// Handle a worker message
function onWorkerMessage(message) {
    if (message.type === 'ready') {
        workerReady = true;
        setStatus(request !== null && !request.done ? 'Synthesizing...' : 'Ready.');
        return;
    }
    if (message.type === 'error' && message.id === null) {
        // The worker could not start (the dictionary failed to load)
        setStatus(message.message, true);
        request = null;
    } else if (request === null || message.id !== request.id) {
        return;
    } else if (message.type === 'error') {
        setStatus(message.message, true);
        request = null;
    } else if (message.type === 'chunk') {
        request.chunks.push(message);
        if (request.play) {
            playChunk(message.samples, message.sampleRate);
            setStatus('Speaking...');
        }
    } else if (message.type === 'done') {
        request.done = true;
        if (message.count === 0) {
            setStatus('Nothing to say - type some text.');
        } else if (request.download) {
            saveWav(request);
            request.download = false;
            setStatus('Saved.');
        } else if (request.play && sources.length === 0) {
            setStatus('Done.');
        }
    }
    updateButtons();
}


// Send a synthesis request to the worker
function startRequest(play, download) {
    if (request !== null && !request.done) {
        startWorker();
    }
    const {settings, key} = getSettings();
    request = {'id': nextRequestId++, key, play, download, 'chunks': [], 'done': false};
    worker.postMessage({'id': request.id, ...settings});
    setStatus(workerReady ? 'Synthesizing...' : 'Loading the pronouncing dictionary...');
    updateButtons();
}


// Schedule a chunk of samples to play after those already scheduled
function playChunk(samples, sampleRate) {
    const buffer = audioContext.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(samples, 0);
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    const startTime = Math.max(audioContext.currentTime + 0.05, nextStartTime);
    source.start(startTime);
    nextStartTime = startTime + buffer.duration;
    source.onended = () => {
        sources = sources.filter((other) => other !== source);
        if (sources.length === 0 && request !== null && request.done && request.play) {
            setStatus('Done.');
        }
        updateButtons();
    };
    sources.push(source);
}


// Stop playback
function stopPlayback() {
    for (const source of sources) {
        source.onended = null;
        source.stop();
    }
    sources = [];
    nextStartTime = 0;
}


// Save a finished request's audio as a WAV file
function saveWav(finished) {
    const length = finished.chunks.reduce((total, chunk) => total + chunk.samples.length, 0);
    const samples = new Float32Array(length);
    let offset = 0;
    for (const chunk of finished.chunks) {
        samples.set(chunk.samples, offset);
        offset += chunk.samples.length;
    }
    const blob = new Blob([encodeWav(samples, finished.chunks[0].sampleRate)], {'type': 'audio/wav'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `speech-${JSON.parse(finished.key).voice}.wav`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
}


// Speak - synthesize the text and play each paragraph as it is ready
function onSpeak() {
    stopPlayback();
    if (audioContext === null) {
        audioContext = new AudioContext();
    }
    audioContext.resume();
    startRequest(true, false);
}


// Stop - stop playback and abandon any synthesis in progress
function onStop() {
    stopPlayback();
    if (request !== null && !request.done) {
        startWorker();
        request = null;
    }
    setStatus('Stopped.');
    updateButtons();
}


// Download WAV - reuse the current audio if the settings are unchanged, else synthesize without playing
function onDownload() {
    const {key} = getSettings();
    if (request !== null && request.key === key) {
        if (request.done) {
            if (request.chunks.length !== 0) {
                saveWav(request);
                setStatus('Saved.');
            }
        } else {
            request.download = true;
        }
        return;
    }
    startRequest(false, true);
}


// Remember the settings per viewer (browser storage may be unavailable)
function saveSettings() {
    try {
        localStorage.setItem('voice-synth', getSettings().key);
    } catch {
        // Settings are a convenience only
    }
}

function loadSettings() {
    try {
        const settings = JSON.parse(localStorage.getItem('voice-synth'));
        if (settings !== null) {
            textElement.value = settings.text;
            document.querySelector(`input[name="voice"][value="${settings.voice === 'female' ? 'female' : 'male'}"]`).checked = true;
            rateElement.value = settings.rate;
            roomElement.checked = settings.room;
        }
    } catch {
        // Settings are a convenience only
    }
}


// Show the rate
function showRate() {
    rateValueElement.textContent = `${Number(rateElement.value).toFixed(2)}x`;
}


// Start up
loadSettings();
showRate();
startWorker();
speakButton.addEventListener('click', onSpeak);
stopButton.addEventListener('click', onStop);
downloadButton.addEventListener('click', onDownload);
rateElement.addEventListener('input', showRate);
for (const element of [textElement, rateElement, roomElement, ...document.querySelectorAll('input[name="voice"]')]) {
    element.addEventListener('change', saveSettings);
}
textElement.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        onSpeak();
    }
});
speakButton.disabled = false;
downloadButton.disabled = false;
updateButtons();
