// The performance lab's front end - polls the backend (tools/perf/server.js) for commands, runs them, and posts their
// results. Commands:
// - {'type': 'timing', ...options} - a timing (tools/perf/timing.js's options, plus where: 'worker' - the default, as
//   PicoShelf synthesizes - or 'main')
// - {'type': 'pipeline', pipelines, lib, texts, voice} - that many pipelines at once, each a network worker and a render
//   worker (tools/perf/labpipe.js) - the aggregate speed and each pipeline's
// - {'type': 'parallel', workers, ...options} - that many timings (stream only) in workers at once - the aggregate speed
//   (every worker's audio over the wall time) and each worker's
// - {'type': 'bench', repeats} - fixed workloads repeated, on the main thread and in a worker (tools/perf/timing.js
//   runBench) - whether the JIT compiles them
// - {'type': 'reload'} - reload the page (and so the lab's own code)
// - {'type': 'info'} - the device's details
// Keep the phone awake and the page in front while it runs (Settings > Display & Brightness > Auto-Lock).

import {defaultTexts, runBench, runTiming} from './timing.js';


// The client's id (kept across reloads)
function clientId() {
    let id;
    try {
        id = localStorage.getItem('perfLabClient');
        if (!id) {
            id = `client-${Math.random().toString(36).slice(2, 8)}`;
            localStorage.setItem('perfLabClient', id);
        }
    } catch {
        id = `client-${Math.random().toString(36).slice(2, 8)}`;
    }
    return id;
}
const client = clientId();


// The page's status and log
const statusElement = document.getElementById('status');
const logElement = document.getElementById('log');
function setStatus(text) {
    statusElement.textContent = `${client} - ${text}`;
}
function log(text) {
    logElement.textContent = `${new Date().toLocaleTimeString()} ${text}\n${logElement.textContent}`.slice(0, 20000);
}


// The device's details
function deviceInfo() {
    return {
        'agent': navigator.userAgent, 'cores': navigator.hardwareConcurrency ?? null, 'memory': navigator.deviceMemory ?? null,
        'screen': `${screen.width}x${screen.height}@${devicePixelRatio}`, 'visible': document.visibilityState
    };
}


// Run a timing in a new worker (a fresh module map and JIT) - its result
function workerTiming(options) {
    return new Promise((resolve, reject) => {
        const worker = new Worker(new URL('labworker.js', import.meta.url), {'type': 'module'});
        worker.onmessage = (event) => {
            worker.terminate();
            if (event.data.type === 'result') {
                resolve(event.data.result);
            } else {
                reject(new Error(event.data.message));
            }
        };
        worker.onerror = (event) => {
            worker.terminate();
            reject(new Error(event.message || 'worker error'));
        };
        worker.postMessage(options);
    });
}


// Run a pipeline - a network worker and a render worker joined by a channel - {network, render} (their done messages)
function pipelineTiming(options) {
    const channel = new MessageChannel();
    const run = (role, port) => new Promise((resolve, reject) => {
        const worker = new Worker(new URL('labpipe.js', import.meta.url), {'type': 'module'});
        worker.onmessage = (event) => {
            worker.terminate();
            resolve(event.data);
        };
        worker.onerror = (event) => {
            worker.terminate();
            reject(new Error(event.message || 'worker error'));
        };
        worker.postMessage({...options, role, port}, [port]);
    });
    return Promise.all([run('render', channel.port2), run('network', channel.port1)]).then(([render, network]) => ({network, render}));
}


// A timing's summary line
function timingSummary(result) {
    const last = result.runs.filter((run) => run.pass === result.passes - 1);
    const parts = last.map((run) => {
        const stages = run.stages ? `network ${run.stages.networkSpeed}x render ${run.stages.renderSpeed}x` : '';
        const stream = run.stream ? ` stream ${run.stream.streamSpeed}x` : '';
        return `${run.voice}/${run.text}: ${stages}${stream}`;
    });
    return `timing ${result.lib}: ${parts.join('; ')} (speed check ${result.speedCheck.ms} ms)`;
}


// Run a command - its result
async function runCommand(command) {
    if (command.type === 'timing') {
        const {where = 'worker', ...options} = command;
        delete options.type;
        delete options.id;
        delete options.client;
        const start = performance.now();
        const result = where === 'main' ? await runTiming(options) : await workerTiming(options);
        result.where = where;
        result.wallMs = Math.round(performance.now() - start);
        log(timingSummary(result));
        return result;
    } else if (command.type === 'pipeline') {
        // Pipelines at once - each a network worker and a render worker (tools/perf/labpipe.js); every pipeline's audio
        // over the longest pipeline's time (after the models load)
        const {pipelines = 1, lib = '/lib', texts = defaultTexts, voice = 'male'} = command;
        const results = await Promise.all(Array.from({'length': pipelines}, () => pipelineTiming({lib, texts, voice})));
        const audio = results.reduce((sum, result) => sum + result.render.audio, 0);
        const wallMs = Math.max(...results.map((result) => result.render.wallMs));
        const summary = {pipelines, 'audio': Math.round(10 * audio) / 10, 'aggregateSpeed': Math.round(100 * audio * 1000 / wallMs) / 100,
            'pipelineSpeeds': results.map((result) => Math.round(100 * result.render.audio * 1000 / result.render.wallMs) / 100),
            'networkSpeeds': results.map((result) => Math.round(100 * result.render.audio * 1000 / result.network.networkMs) / 100),
            'renderSpeeds': results.map((result) => Math.round(100 * result.render.audio * 1000 / result.render.renderMs) / 100),
            'firstMs': results.map((result) => Math.round(result.render.firstMs))};
        log(`pipeline ${pipelines}: aggregate ${summary.aggregateSpeed}x, each ${summary.pipelineSpeeds.join(' ')}x`);
        return summary;
    } else if (command.type === 'parallel') {
        // Workers timings at once - the aggregate speed: every worker's audio over the wall time
        const {workers = 2, ...options} = command;
        delete options.type;
        delete options.id;
        delete options.client;
        const start = performance.now();
        const results = await Promise.all(Array.from({'length': workers}, () => workerTiming({...options, 'stages': false})));
        const wallMs = performance.now() - start;
        const audio = results.reduce((total, result) => total + result.runs.reduce((sum, run) => sum + run.stream.audio, 0), 0);
        const work = results.reduce((total, result) => total + result.runs.reduce((sum, run) => sum + run.stream.streamMs, 0), 0);
        const summary = {workers, 'wallMs': Math.round(wallMs), 'audio': Math.round(audio * 10) / 10,
            'aggregateSpeed': Math.round(10 * audio * 1000 / wallMs) / 10, 'perWorkerSpeed': Math.round(10 * audio * 1000 / work) / 10};
        log(`parallel ${workers}: aggregate ${summary.aggregateSpeed}x, each ${summary.perWorkerSpeed}x`);
        return {...summary, results};
    } else if (command.type === 'bench') {
        const result = {'main': runBench(command), 'worker': await workerTiming({...command, 'bench': true})};
        log(`bench: main speed check ${result.main.speedCheck.join(' ')} ms; worker ${result.worker.speedCheck.join(' ')} ms`);
        return result;
    } else if (command.type === 'info') {
        return deviceInfo();
    }
    throw new Error(`unknown command ${command.type}`);
}


// Post a result
async function postResult(result) {
    await fetch('/api/result', {'method': 'POST', 'body': JSON.stringify({client, 'device': deviceInfo(), ...result})});
}


// The poll loop - each command run in turn
async function pollLoop() {
    for (;;) {
        let command;
        try {
            setStatus('waiting for a command');
            const response = await fetch(`/api/next?client=${encodeURIComponent(client)}&t=${Date.now()}`, {'cache': 'no-store'});
            command = await response.json();
        } catch (error) {
            setStatus(`can't reach the backend (${error.message}) - retrying`);
            await new Promise((resolve) => {
                setTimeout(resolve, 2000);
            });
            continue;
        }
        if (command.type === 'none') {
            continue;
        }
        if (command.type === 'reload') {
            await postResult({'command': command.id, 'type': 'reload'});
            location.reload();
            return;
        }
        setStatus(`running ${command.type} ${command.id}`);
        log(`command ${command.id}: ${command.type}`);
        try {
            await postResult({'command': command.id, 'type': command.type, 'result': await runCommand(command)});
        } catch (error) {
            log(`command ${command.id} failed: ${error.message}`);
            await postResult({'command': command.id, 'type': command.type, 'error': error.message});
        }
    }
}


// The buttons
document.getElementById('timing').onclick = () => {
    setStatus('running a local timing');
    runCommand({'type': 'timing'}).catch((error) => log(`local timing failed: ${error.message}`));
};
document.getElementById('reload').onclick = () => location.reload();


// The audio routes - a second of a 440 Hz tone straight to Web Audio's output, or through a media element (a
// MediaStream destination as an audio element's source - PicoShelf's route on iPhones)
function playTone(viaMedia) {
    const context = new AudioContext();
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    gain.gain.value = 0.3;
    oscillator.frequency.value = 440;
    oscillator.connect(gain);
    if (viaMedia) {
        const stream = context.createMediaStreamDestination();
        gain.connect(stream);
        const element = document.createElement('audio');
        element.setAttribute('playsinline', '');
        element.srcObject = stream.stream;
        element.play().then(() => log('tone: media element playing'), (error) => log(`tone: media element refused (${error.message})`));
    } else {
        gain.connect(context.destination);
    }
    context.resume().then(() => log(`tone: audio ${context.state}, secure context ${window.isSecureContext}`));
    oscillator.start();
    oscillator.stop(context.currentTime + 1);
    postResult({'type': 'tone', 'route': viaMedia ? 'media' : 'direct', 'state': context.state, 'secure': window.isSecureContext});
}
document.getElementById('toneDirect').onclick = () => playTone(false);
document.getElementById('toneMedia').onclick = () => playTone(true);


log(`started - ${navigator.userAgent}`);
pollLoop();
