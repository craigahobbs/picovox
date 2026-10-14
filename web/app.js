// The browser front end - text, voice, rate, room, and crowd controls; streaming synthesis in Web Workers (one per
// crowd member, so the members synthesize in parallel); playback with Web Audio (each member panned, the room ambience
// by a convolver); and WAV download

import {encodeWav} from '../lib/wav.js';
import {convolve, roomImpulse} from '../lib/room.js';
import {normalize} from '../lib/klatt.js';


// Elements
const textElement = document.getElementById('text');
const rateElement = document.getElementById('rate');
const rateValueElement = document.getElementById('rateValue');
const roomElement = document.getElementById('room');
const crowdElement = document.getElementById('crowd');
const speakButton = document.getElementById('speak');
const stopButton = document.getElementById('stop');
const downloadButton = document.getElementById('download');
const statusElement = document.getElementById('status');


// The crowd - the lead (the chosen voice, centered) and five other speakers, each with its own voice, pitch, vocal
// tract length, rate, seed, start offset (s), stereo position, and level
const crowdMembers = [
    {'voice': null, 'pitchScale': 1, 'tractScale': 1, 'rate': 1, 'seed': 1, 'offset': 0, 'pan': 0, 'gain': 1},
    {'voice': 'male', 'pitchScale': 0.88, 'tractScale': 1.05, 'rate': 0.95, 'seed': 11, 'offset': 0.11, 'pan': -0.6, 'gain': 0.72},
    {'voice': 'female', 'pitchScale': 1.08, 'tractScale': 0.97, 'rate': 1.04, 'seed': 12, 'offset': 0.05, 'pan': 0.5, 'gain': 0.72},
    {'voice': 'male', 'pitchScale': 1.12, 'tractScale': 0.96, 'rate': 1.06, 'seed': 13, 'offset': 0.2, 'pan': 0.3, 'gain': 0.66},
    {'voice': 'female', 'pitchScale': 0.9, 'tractScale': 1.03, 'rate': 0.97, 'seed': 14, 'offset': 0.16, 'pan': -0.35, 'gain': 0.66},
    {'voice': 'male', 'pitchScale': 0.95, 'tractScale': 1, 'rate': 1.02, 'seed': 15, 'offset': 0.27, 'pan': 0.8, 'gain': 0.58}
];


// State - the workers ([{worker, ready}]), the current request, the playing sources, and the audio graph
let workers = [];
let request = null;
let nextRequestId = 1;
let audioContext = null;
let sources = [];
let output = null;


// Show a status message
function setStatus(message, isError = false) {
    statusElement.textContent = message;
    statusElement.classList.toggle('error', isError);
}


// The current settings, and a key identifying the audio they produce
function getSettings() {
    const voice = document.querySelector('input[name="voice"]:checked').value;
    const settings = {
        'text': textElement.value, voice, 'rate': Number(rateElement.value), 'seed': 1, 'room': roomElement.checked,
        'crowd': crowdElement.checked
    };
    return {settings, 'key': JSON.stringify(settings)};
}


// The speakers for settings - the crowd, or just the chosen voice
function getMembers(settings) {
    const members = settings.crowd ? crowdMembers : crowdMembers.slice(0, 1);
    return members.map((member) => ({...member, 'voice': member.voice === null ? settings.voice : member.voice}));
}


// Is a request still synthesizing?
function isBusy() {
    return request !== null && request.done.some((done) => !done);
}


// Update the buttons for the current state
function updateButtons() {
    stopButton.disabled = !isBusy() && sources.length === 0;
}


// Create a synthesis worker at an index of the pool
function createWorker(index) {
    const worker = new Worker(new URL('worker.js', import.meta.url), {'type': 'module'});
    const entry = {worker, 'ready': false};
    worker.onmessage = (event) => onWorkerMessage(entry, event.data);
    worker.onerror = (event) => {
        event.preventDefault();
        setStatus(`The synthesis worker failed: ${event.message || 'could not load (serve the page over HTTP)'}`, true);
        request = null;
        updateButtons();
    };
    workers[index] = entry;
}


// Make sure the pool has a count of workers
function ensureWorkers(count) {
    for (let ix = workers.length; ix < count; ix++) {
        createWorker(ix);
    }
}


// Restart all workers - abandoning any synthesis in progress
function restartWorkers() {
    const count = workers.length;
    for (const entry of workers) {
        entry.worker.terminate();
    }
    workers = [];
    ensureWorkers(Math.max(1, count));
}


// Handle a worker message
function onWorkerMessage(entry, message) {
    if (message.type === 'ready') {
        entry.ready = true;
        if (!isBusy() && sources.length === 0) {
            setStatus('Ready.');
        }
        return;
    }
    if (message.type === 'error' && message.id === null) {
        // A worker could not start (the dictionary failed to load)
        setStatus(message.message, true);
        request = null;
    } else if (request === null || message.id !== request.id) {
        return;
    } else if (message.type === 'error') {
        setStatus(message.message, true);
        request = null;
    } else if (message.type === 'block') {
        if (request.play) {
            playBlock(message.member, message.samples, message.sampleRate);
            setStatus(request.members.length > 1 ? 'The crowd is speaking...' : 'Speaking...');
        }
    } else if (message.type === 'chunk') {
        request.chunks[message.member].push(message.samples);
        request.sampleRate = message.sampleRate;
    } else if (message.type === 'done') {
        request.done[message.member] = true;
        request.count = message.count;
        if (!isBusy()) {
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
    }
    updateButtons();
}


// Send a synthesis request to the workers, one crowd member per worker - streaming requests post blocks to play as
// they are synthesized
function startRequest(play, download, stream) {
    if (isBusy()) {
        restartWorkers();
    }
    const {settings, key} = getSettings();
    const members = getMembers(settings);
    ensureWorkers(members.length);
    request = {
        'id': nextRequestId++, key, play, download, stream, members,
        'chunks': members.map(() => []), 'done': members.map(() => false), 'next': members.map(() => null),
        'nodes': [], 'startTime': null, 'sampleRate': 44100, 'count': 0
    };
    if (play) {
        request.nodes = members.map((member) => memberNode(member));
    }
    members.forEach((member, ix) => {
        workers[ix].worker.postMessage({
            'id': request.id, 'member': ix, 'text': settings.text, 'voice': member.voice, 'rate': settings.rate * member.rate,
            'seed': member.seed, 'room': settings.room && members.length === 1, stream,
            'overrides': {'pitchScale': member.pitchScale, 'tractScale': member.tractScale}
        });
    });
    setStatus(workers[0].ready ? 'Synthesizing...' : 'Loading the pronouncing dictionary...');
    updateButtons();
}


// A crowd member's playback node - its level and stereo position, into the output
function memberNode(member) {
    const gain = audioContext.createGain();
    gain.gain.value = member.gain;
    const panner = audioContext.createStereoPanner();
    panner.pan.value = member.pan;
    gain.connect(panner).connect(output);
    return gain;
}


// Schedule a member's block to play after the member's last block - the members start together (after a short lead),
// each at its offset
function playBlock(member, samples, sampleRate) {
    const buffer = audioContext.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(samples, 0);
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(request.nodes[member]);
    if (request.startTime === null) {
        request.startTime = audioContext.currentTime + (request.members.length > 1 ? 0.15 : 0.08);
    }
    const scheduled = request.next[member] === null ? request.startTime + request.members[member].offset : request.next[member];
    const startTime = Math.max(audioContext.currentTime + 0.02, scheduled);
    source.start(startTime);
    request.next[member] = startTime + buffer.duration;
    source.onended = () => {
        sources = sources.filter((other) => other !== source);
        if (sources.length === 0 && request !== null && !isBusy() && request.play) {
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
}


// Mix a finished request's members to mono - each at its offset and level, then the room ambience (a crowd's members
// are synthesized dry), normalized
function mixRequest(finished) {
    const sampleRate = finished.sampleRate;
    const tracks = finished.chunks.map((chunks) => {
        const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
        const track = new Float32Array(length);
        let offset = 0;
        for (const chunk of chunks) {
            track.set(chunk, offset);
            offset += chunk.length;
        }
        return track;
    });
    if (tracks.length === 1) {
        return tracks[0];
    }
    const starts = finished.members.map((member) => Math.round(member.offset * sampleRate));
    const length = Math.max(...tracks.map((track, ix) => starts[ix] + track.length));
    const mix = new Float32Array(length);
    tracks.forEach((track, ix) => {
        const gain = finished.members[ix].gain;
        for (let jx = 0; jx < track.length; jx++) {
            mix[starts[ix] + jx] += gain * track[jx];
        }
    });
    const settings = JSON.parse(finished.key);
    return normalize(settings.room ? convolve(mix, roomImpulse(sampleRate)) : mix, 0.89);
}


// Save a finished request's audio as a WAV file
function saveWav(finished) {
    const samples = mixRequest(finished);
    const blob = new Blob([encodeWav(samples, finished.sampleRate)], {'type': 'audio/wav'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const settings = JSON.parse(finished.key);
    link.href = url;
    link.download = `speech-${settings.crowd ? 'crowd' : settings.voice}.wav`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
}


// Connect the output for the room setting - streamed audio is dry, so the room ambience is a convolver
function connectOutput(room, crowd) {
    if (output !== null) {
        output.disconnect();
    }
    // A crowd is louder - its members sum
    const master = audioContext.createGain();
    master.gain.value = crowd ? 0.42 : 1;
    master.connect(audioContext.destination);
    if (room) {
        // The convolver's impulse response must be at the context's sample rate
        const impulse = roomImpulse(audioContext.sampleRate);
        const buffer = audioContext.createBuffer(1, impulse.length, audioContext.sampleRate);
        buffer.copyToChannel(impulse, 0);
        const convolver = audioContext.createConvolver();
        convolver.normalize = false;
        convolver.buffer = buffer;
        const gain = audioContext.createGain();
        gain.gain.value = 0.85;
        convolver.connect(gain).connect(master);
        output = convolver;
    } else {
        output = master;
    }
}


// Speak - stream the text, playing each block as soon as it is synthesized
function onSpeak() {
    stopPlayback();
    if (audioContext === null) {
        audioContext = new AudioContext();
    }
    audioContext.resume();
    connectOutput(roomElement.checked, crowdElement.checked);
    startRequest(true, false, true);
}


// Stop - stop playback and abandon any synthesis in progress
function onStop() {
    stopPlayback();
    if (isBusy()) {
        restartWorkers();
        request = null;
    }
    setStatus('Stopped.');
    updateButtons();
}


// Download WAV - reuse the current (non-streamed) audio if the settings are unchanged, else synthesize without playing
function onDownload() {
    const {key} = getSettings();
    if (request !== null && request.key === key && !request.stream) {
        if (!isBusy()) {
            if (request.count !== 0) {
                saveWav(request);
                setStatus('Saved.');
            }
        } else {
            request.download = true;
        }
        return;
    }
    startRequest(false, true, false);
}


// Start the crowd's workers when the crowd is chosen, so they're ready (dictionary loaded) by the next Speak
function onCrowd() {
    if (crowdElement.checked) {
        ensureWorkers(crowdMembers.length);
    }
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
            crowdElement.checked = settings.crowd === true;
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
ensureWorkers(1);
onCrowd();
speakButton.addEventListener('click', onSpeak);
stopButton.addEventListener('click', onStop);
downloadButton.addEventListener('click', onDownload);
rateElement.addEventListener('input', showRate);
crowdElement.addEventListener('change', onCrowd);
for (const element of [textElement, rateElement, roomElement, crowdElement, ...document.querySelectorAll('input[name="voice"]')]) {
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
