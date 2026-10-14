// The browser front end - text, voice, rate, room, and crowd controls; streaming synthesis in Web Workers (one per
// crowd member, so the members synthesize in parallel); a player (play/pause, and skipping by 1 or 5 minutes) over
// the streamed audio with Web Audio (each member panned, the room ambience by a convolver); and WAV download

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
const skipButtons = [...document.querySelectorAll('button[data-skip]')];
const playPauseButton = document.getElementById('playPause');
const timeElement = document.getElementById('time');
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


// State - the workers ([{worker, ready}]), the current request, and the audio graph
let workers = [];
let request = null;
let nextRequestId = 1;
let audioContext = null;
let output = null;


// The player - the streamed request's audio is kept as tracks of blocks on a timeline (s), so it can be played from any
// position. While playing, the timeline position anchorPosition is at audio context time anchorTime, and each member's
// blocks are scheduled a few seconds ahead (cursors are each member's next block to schedule).
const player = {
    'request': null, 'playing': false, 'anchorPosition': 0, 'anchorTime': 0, 'sources': new Set(), 'startTimer': null, 'cursors': []
};


// How far ahead to schedule blocks (s)
const scheduleAheadSeconds = 4;


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
function isBusy(target = request) {
    return target !== null && target.done.some((done) => !done);
}


// The player's timeline duration so far (s)
function duration() {
    const target = player.request;
    if (target === null) {
        return 0;
    }
    return Math.max(0, ...target.tracks.map((track, ix) => (track.length === 0 ? 0 : target.members[ix].offset + track.length)));
}


// The player's current timeline position (s)
function position() {
    if (!player.playing) {
        return player.anchorPosition;
    }
    return Math.max(0, player.anchorPosition + (audioContext.currentTime - player.anchorTime));
}


// Format seconds as m:ss.s
function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    return `${minutes}:${(seconds - minutes * 60).toFixed(1).padStart(4, '0')}`;
}


// Update the buttons and the time for the current state
function updateControls() {
    const hasAudio = duration() > 0;
    stopButton.disabled = !isBusy() && !player.playing;
    playPauseButton.disabled = !hasAudio && player.request === null;
    for (const button of skipButtons) {
        button.disabled = !hasAudio;
    }
    playPauseButton.innerHTML = player.playing ? '&#x23F8; Pause' : '&#x25B6; Play';
    playPauseButton.setAttribute('aria-label', player.playing ? 'Pause' : 'Play');
    timeElement.textContent = `${formatTime(Math.min(position(), duration()))} / ${formatTime(duration())}`;
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
        updateControls();
    };
    workers[index] = entry;
}


// Make sure the pool has a count of workers
function ensureWorkers(count) {
    for (let ix = workers.length; ix < count; ix++) {
        createWorker(ix);
    }
}


// Restart all workers - abandoning any synthesis in progress (the request keeps the audio it has, as if finished)
function restartWorkers() {
    const count = workers.length;
    for (const entry of workers) {
        entry.worker.terminate();
    }
    workers = [];
    ensureWorkers(Math.max(1, count));
    if (request !== null) {
        request.done = request.done.map(() => true);
    }
}


// Handle a worker message
function onWorkerMessage(entry, message) {
    if (message.type === 'ready') {
        entry.ready = true;
        if (!isBusy() && !player.playing && player.request === null) {
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
        request.done = request.done.map(() => true);
    } else if (message.type === 'block') {
        addBlock(request, message.member, message.samples, message.sampleRate);
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
            }
        }
    }
    updateControls();
}


// Send a synthesis request to the workers, one crowd member per worker - streaming requests post blocks for the player
function startRequest(stream, download) {
    if (isBusy()) {
        restartWorkers();
    }
    const {settings, key} = getSettings();
    const members = getMembers(settings);
    ensureWorkers(members.length);
    request = {
        'id': nextRequestId++, key, stream, download, members, 'count': 0, 'sampleRate': 44100, 'autoplay': false,
        'chunks': members.map(() => []), 'done': members.map(() => false), 'tracks': members.map(() => ({'blocks': [], 'length': 0})),
        'nodes': []
    };
    members.forEach((member, ix) => {
        workers[ix].worker.postMessage({
            'id': request.id, 'member': ix, 'text': settings.text, 'voice': member.voice, 'rate': settings.rate * member.rate,
            'seed': member.seed, 'room': settings.room && members.length === 1, stream,
            'overrides': {'pitchScale': member.pitchScale, 'tractScale': member.tractScale}
        });
    });
    setStatus(workers[0].ready ? 'Synthesizing...' : 'Loading the pronouncing dictionary...');
    updateControls();
    return request;
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


// Add a streamed block to a member's track - and play it, if the player is playing. Playback starts once every member
// has its first block (or shortly after the first block, if a member is slow), so no speaker is clipped.
function addBlock(target, member, samples, sampleRate) {
    const buffer = audioContext.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(samples, 0);
    const track = target.tracks[member];
    const block = {buffer, 'start': track.length};
    track.blocks.push(block);
    track.length += buffer.duration;
    if (player.request !== target) {
        return;
    }
    if (player.playing) {
        scheduleAhead();
    } else if (target.autoplay) {
        const allStarted = target.tracks.every((other) => other.blocks.length !== 0);
        if (allStarted) {
            target.autoplay = false;
            clearTimeout(player.startTimer);
            player.startTimer = null;
            startPlayback(0);
        } else if (player.startTimer === null) {
            player.startTimer = setTimeout(() => {
                player.startTimer = null;
                if (target.autoplay && player.request === target) {
                    target.autoplay = false;
                    startPlayback(0);
                }
            }, 400);
        }
    }
}


// Schedule a block for the current playback - skipping what is before the position
function scheduleBlock(member, block) {
    const target = player.request;
    const blockStart = target.members[member].offset + block.start;
    const blockEnd = blockStart + block.buffer.duration;
    if (blockEnd <= player.anchorPosition) {
        return;
    }
    let when = player.anchorTime + (blockStart - player.anchorPosition);
    let offset = 0;
    const earliest = audioContext.currentTime + 0.005;
    if (when < earliest) {
        offset = earliest - when;
        when = earliest;
    }
    if (offset >= block.buffer.duration) {
        return;
    }
    const source = audioContext.createBufferSource();
    source.buffer = block.buffer;
    source.connect(target.nodes[member]);
    source.start(when, offset);
    source.onended = () => player.sources.delete(source);
    player.sources.add(source);
}


// Schedule each member's blocks that start within the schedule-ahead window
function scheduleAhead() {
    const horizon = position() + scheduleAheadSeconds;
    player.request.tracks.forEach((track, member) => {
        const offset = player.request.members[member].offset;
        while (player.cursors[member] < track.blocks.length && offset + track.blocks[player.cursors[member]].start < horizon) {
            scheduleBlock(member, track.blocks[player.cursors[member]]);
            player.cursors[member]++;
        }
    });
}


// Stop the scheduled sources
function stopSources() {
    for (const source of player.sources) {
        source.onended = null;
        source.stop();
    }
    player.sources.clear();
}


// Play the player's request from a timeline position
function startPlayback(at) {
    stopSources();
    audioContext.resume();
    player.anchorPosition = at;
    player.anchorTime = audioContext.currentTime + (player.request.members.length > 1 ? 0.1 : 0.05);
    player.playing = true;
    player.cursors = player.request.tracks.map((track, member) => {
        const offset = player.request.members[member].offset;
        const index = track.blocks.findIndex((block) => offset + block.start + block.buffer.duration > at);
        return index === -1 ? track.blocks.length : index;
    });
    scheduleAhead();
    setStatus(player.request.members.length > 1 ? 'The crowd is speaking...' : 'Speaking...');
    updateControls();
}


// Pause the player, keeping its position
function pausePlayback() {
    if (player.playing) {
        player.anchorPosition = Math.min(position(), duration());
        player.playing = false;
        stopSources();
        setStatus('Paused.');
    }
    if (player.request !== null) {
        player.request.autoplay = false;
    }
    updateControls();
}


// Move the player's position - continuing to play from there if playing
function seek(to) {
    const clamped = Math.max(0, Math.min(to, duration()));
    if (player.playing) {
        startPlayback(clamped);
    } else {
        player.anchorPosition = clamped;
        updateControls();
    }
}


// The player's clock - schedules ahead, updates the time, and stops at the end of finished audio
function tick() {
    if (player.playing) {
        scheduleAhead();
    }
    if (player.playing && !isBusy(player.request) && position() >= duration() + 0.05) {
        player.playing = false;
        player.anchorPosition = duration();
        stopSources();
        setStatus('Done.');
    }
    updateControls();
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


// Speak - stream the text into a new player request, playing as soon as the audio starts arriving
function onSpeak() {
    stopSources();
    player.playing = false;
    clearTimeout(player.startTimer);
    player.startTimer = null;
    if (audioContext === null) {
        audioContext = new AudioContext();
    }
    audioContext.resume();
    connectOutput(roomElement.checked, crowdElement.checked);
    const target = startRequest(true, false);
    target.nodes = target.members.map((member) => memberNode(member));
    target.autoplay = true;
    player.request = target;
    player.anchorPosition = 0;
    updateControls();
}


// Stop - stop playback (back to the start) and abandon any synthesis in progress
function onStop() {
    stopSources();
    player.playing = false;
    player.anchorPosition = 0;
    if (player.request !== null) {
        player.request.autoplay = false;
    }
    if (isBusy()) {
        restartWorkers();
    }
    setStatus('Stopped.');
    updateControls();
}


// Play/Pause - pause, or play from the position (from the start, if at the end); with nothing to play, Speak
function onPlayPause() {
    if (player.request === null || duration() === 0) {
        onSpeak();
    } else if (player.playing) {
        pausePlayback();
    } else {
        const atEnd = player.anchorPosition >= duration() - 0.05 && !isBusy(player.request);
        startPlayback(atEnd ? 0 : player.anchorPosition);
    }
}


// Skip back or ahead by seconds (each skip button's data-skip)
function onSkip(seconds) {
    if (player.request !== null) {
        player.request.autoplay = false;
        seek(position() + seconds);
    }
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
    startRequest(false, true);
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
playPauseButton.addEventListener('click', onPlayPause);
for (const button of skipButtons) {
    button.addEventListener('click', () => onSkip(Number(button.dataset.skip)));
}
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
setInterval(tick, 100);
updateControls();
