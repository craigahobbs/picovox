// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import * as train from '../tools/dashboard/train.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


// A small stage graph - data (CPU), a check after it (ordering only), three GPU stages, and a final CPU stage - and its make
// targets
const manifest = {
    'target': 'train',
    'targets': {'train': ['check', 'package'], 'train-g2p': ['check'], 'train-voice': ['package']},
    'stages': [
        {'stage': 'check', 'on': 'cpu', 'minutes': 10, 'needs': [], 'after': ['data']},
        {'stage': 'data', 'on': 'cpu', 'minutes': 5, 'needs': []},
        {'stage': 'voice-a', 'on': 'gpu', 'minutes': 20, 'needs': ['data'], 'run': 'voice-a'},
        {'stage': 'voice-b', 'on': 'gpu', 'minutes': 10, 'needs': ['voice-a']},
        {'stage': 'eval', 'on': 'gpu', 'minutes': 5, 'needs': ['voice-a']},
        {'stage': 'package', 'on': 'cpu', 'minutes': 1, 'needs': ['eval', 'voice-b']}
    ]
};
const time = (clock) => train.parseLocalTime(`2026-10-16T${clock}`);


test('train.parseLocalTime', () => {
    assert.equal(train.parseLocalTime('2026-10-16T08:20:51'), new Date(2026, 9, 16, 8, 20, 51).getTime());
    assert(Number.isNaN(train.parseLocalTime('08:20:51')));
});


test('train.stageOrder', () => {
    assert.deepEqual(train.stageOrder(manifest).map((stage) => stage.stage), ['data', 'check', 'voice-a', 'voice-b', 'eval', 'package']);
});


test('train.targetStages', () => {
    // A target's stages and every stage those need
    assert.equal(train.targetStages(manifest, 'train').size, 6);
    assert.deepEqual([...train.targetStages(manifest, 'train-g2p')], ['check']);
    assert.deepEqual([...train.targetStages(manifest, 'train-voice')].sort(), ['data', 'eval', 'package', 'voice-a', 'voice-b']);
});


test('train.parseRunTarget', () => {
    assert.equal(train.parseRunTarget(''), null);
    assert.deepEqual(train.parseRunTarget([
        '{"event": "target", "target": "train", "time": 100}',
        '{"stage": "data", "event": "start", "time": 101}',
        'not json',
        '{"event": "target", "target": "train-voice", "time": 200}'
    ].join('\n')), {'target': 'train-voice', 'time': 200000});
});


test('train.stageStates, a later run', () => {
    // An earlier run's unfinished attempts are stale - waiting in the later run; its finished stages stay done
    const events = train.parseStageEvents([
        '{"stage": "data", "event": "start", "time": 100}',
        '{"stage": "data", "event": "end", "time": 160}',
        '{"stage": "check", "event": "start", "time": 160}',
        '{"stage": "check", "event": "fail", "time": 170, "status": 2}',
        '{"stage": "voice-a", "event": "start", "time": 160}'
    ].join('\n'));
    const states = train.stageStates(manifest, {events, 'alive': true, 'started': 300000,
        'target': 'train'});
    assert.equal(states.get('data').status, 'done');
    assert.equal(states.get('check').status, 'waiting');
    assert.equal(states.get('voice-a').status, 'waiting');
});


test('train.parseStageEvents', () => {
    const text = [
        '{"stage": "data", "event": "start", "time": 100}',
        '{"stage": "data", "event": "end", "time": 160}',
        '{"stage": "voice-a", "event": "start", "time": 170}',
        '{"stage": "voice-a", "event": "fail", "time": 180, "status": 1}',
        '{"stage": "eval", "event": "fail", "time": 190, "signal": "terminated"}',
        '{"stage": "voice-b", "event": "end", "time": 195}',
        '{"stage": "package", "event": "start", "ti',
        ''
    ].join('\n');
    assert.deepEqual(Object.fromEntries(train.parseStageEvents(text)), {
        'data': {'start': 100000, 'end': 160000, 'fail': null, 'failure': null},
        'voice-a': {'start': 170000, 'end': null, 'fail': 180000, 'failure': 'exit status 1'},
        'eval': {'start': null, 'end': null, 'fail': 190000, 'failure': 'terminated'}
    });
});


test('train.sessionStart', () => {
    const minute = 60000;
    const event = (start, end, fail = null) => ({start, end, fail, 'failure': null});
    // (two sessions - the second starts after more than the gap with nothing running; overlapping and unfinished
    // attempts run on, and an attempt without a start is ignored)
    const events = new Map([
        ['data', event(0, 10 * minute)],
        ['voice-a', event(5 * minute, 40 * minute)],
        ['eval', event(41 * minute, null, 50 * minute)],
        ['voice-b', event(200 * minute, null)],
        ['package', event(205 * minute, 206 * minute)],
        ['check', event(null, null, 3 * minute)]
    ]);
    assert.equal(train.sessionStart(events, 300 * minute, 60 * minute), 200 * minute);
    // (within the gap - one session)
    assert.equal(train.sessionStart(events, 300 * minute, 160 * minute), 0);
    // (an unfinished attempt runs to now - nothing after it starts a session)
    assert.equal(train.sessionStart(new Map([['data', event(0, null)], ['voice-a', event(100 * minute, 110 * minute)]]),
        300 * minute, 60 * minute), 0);
    assert.equal(train.sessionStart(new Map(), 0, 60 * minute), null);
});


test('train.parseUtilLog', () => {
    // Each header names its sampler's columns (a line before any header, or not a sample, is skipped)
    const text = [
        '2026-10-16T23:59:20 50 90 40.0 voice-a',
        'time cpu_busy% gpu% mem_used% stages',
        '2026-10-16T23:59:40 50 90 40.0 voice-a',
        '2026-10-17T00:00:10 60 80 40.0 voice-a',
        'garbage',
        '2026-10-17T00:00:40 70 0 40.5 voice-a',
        'time cpu_busy% gpu% mem_used% system_w power_in_w thermal stages',
        '2026-10-17T00:01:00 10 90 41.0 35.3 48.3 2 voice-a',
        '2026-10-17T00:01:20 10 90 41.0 - - - voice-a',
        ''
    ].join('\n');
    const samples = train.parseUtilLog(text);
    assert.deepEqual(samples, [
        {'time': time('23:59:40'), 'cpu': 50, 'gpu': 90, 'mem': 40, 'system': NaN, 'powerIn': NaN, 'thermal': NaN},
        {'time': time('23:59:40') + 30000, 'cpu': 60, 'gpu': 80, 'mem': 40, 'system': NaN, 'powerIn': NaN, 'thermal': NaN},
        {'time': time('23:59:40') + 60000, 'cpu': 70, 'gpu': 0, 'mem': 40.5, 'system': NaN, 'powerIn': NaN, 'thermal': NaN},
        {'time': time('23:59:40') + 80000, 'cpu': 10, 'gpu': 90, 'mem': 41, 'system': 35.3, 'powerIn': 48.3, 'thermal': 2},
        {'time': time('23:59:40') + 100000, 'cpu': 10, 'gpu': 90, 'mem': 41, 'system': NaN, 'powerIn': NaN, 'thermal': NaN}
    ]);
    assert.deepEqual(train.meanUtilization(samples.slice(0, 3), -Infinity, Infinity), {'cpu': 60, 'gpu': 170 / 3, 'count': 3});
    assert.deepEqual(train.meanUtilization(samples, 0, 1), {'cpu': NaN, 'gpu': NaN, 'count': 0});
});


test('train.logProgress', () => {
    const progress = {'log': 'train.log', 'pattern': 'step (\\d+)/\\1 done', 'total': 4};
    const text = 'step 5/10\nstep 10/10 done\nfold 1: step 10/10 done\nstep 3/10 done\n';
    assert.deepEqual(train.logProgress(text, progress), {'done': 2, 'total': 4, 'fraction': 0.5});
    assert.deepEqual(train.logProgress('', progress), {'done': 0, 'total': 4, 'fraction': 0});
    assert.equal(train.logProgress('step 1/1 done\n'.repeat(5), progress).fraction, 1);
});


test('train.logProgress, phases', () => {
    const progress = {'phases': [
        {'name': 'make', 'header': '^make: \\d+ items, (\\d+) to make$', 'step': '^  (\\d+)/(\\d+) made, (\\d+) s$', 'secondsEach': 2},
        {'name': 'check', 'header': '^check: (?:(\\d+) to check|current)$', 'step': '^  (\\d+)/(\\d+) checked, (\\d+) s$', 'secondsEach': 1,
            'sameCount': true},
        {'name': 'tidy', 'header': '^tidy: (\\d+)$', 'step': '^  (\\d+)/(\\d+) tidied, (\\d+) s$', 'secondsEach': 60}
    ]};
    // Before the first header - none
    assert.equal(train.logProgress('gpulock: waiting\n', progress), null);
    // The first phase, before its first step - its count at its secondsEach, and the later phases' (check's the same
    // count, tidy's none - its count known when it starts)
    assert.deepEqual(train.logProgress('make: 100 items, 30 to make\n', progress),
        {'done': 0, 'total': 30, 'phase': 'make', 'minutesLeft': (30 * 2 + 30 * 1) / 60});
    // At its pace - 10 of 30 in 40 s: 80 s to go, then check's 30 at a second each
    assert.deepEqual(train.logProgress('make: 100 items, 30 to make\n  10/30 made, 40 s\n', progress),
        {'done': 10, 'total': 30, 'phase': 'make', 'minutesLeft': (80 + 30) / 60});
    // A later phase - its own count and pace; a header without a count (current) - none to do
    const text = 'make: 100 items, 30 to make\n  30/30 made, 120 s\ncheck: 20 to check\n  5/20 checked, 10 s\n';
    assert.deepEqual(train.logProgress(text, progress), {'done': 5, 'total': 20, 'phase': 'check', 'minutesLeft': 30 / 60});
    assert.deepEqual(train.logProgress('make: 100 items, 0 to make\ncheck: current\n', progress),
        {'done': 0, 'total': 0, 'phase': 'check', 'minutesLeft': 0});
    assert.deepEqual(train.logProgress(`${text}tidy: 2\n  1/2 tidied, 50 s\n`, progress),
        {'done': 1, 'total': 2, 'phase': 'tidy', 'minutesLeft': 50 / 60});
    // The latest attempt's lines - from its first phase's last header
    assert.deepEqual(train.logProgress(`${text}make: 100 items, 4 to make\n`, progress),
        {'done': 0, 'total': 4, 'phase': 'make', 'minutesLeft': (4 * 2 + 4) / 60});
});


test('train.runManifest', () => {
    const firstRun = {'firstRun': {'stage': 'corpus', 'pattern': '^make: \\d+ items, (\\d+) to make$'},
        'stages': [{'stage': 'corpus', 'minutes': 1, 'firstMinutes': 60}, {'stage': 'train', 'minutes': 20}]};
    // Making the training speech (the latest generate line's count) - the first run's minutes
    const made = train.runManifest(firstRun, 'make: 10 items, 0 to make\nmake: 10 items, 4 to make\n');
    assert.deepEqual(made.stages.map((stage) => stage.minutes), [60, 20]);
    // Cached (none to make, in the latest line), not known yet, or no firstRun - the manifest as is
    assert.equal(train.runManifest(firstRun, 'make: 10 items, 4 to make\nmake: 10 items, 0 to make\n'), firstRun);
    assert.equal(train.runManifest(firstRun, 'gpulock: waiting\n'), firstRun);
    assert.equal(train.runManifest(firstRun, undefined), firstRun);
    const plain = {'stages': firstRun.stages};
    assert.equal(train.runManifest(plain, 'make: 10 items, 4 to make\n'), plain);
});


test('train.stageMinutesLeft', () => {
    const stage = {'stage': 'context', 'minutes': 25};
    const state = {'status': 'running', 'start': 0, 'end': null};
    const minutes = 60000;
    // By its progress - 20 of 55 parts in 40 minutes
    assert.equal(train.stageMinutesLeft(stage, state, 40 * minutes, {'done': 20, 'total': 55, 'fraction': 20 / 55}), 70);
    // Done, and nearly done (at least a minute)
    assert.equal(train.stageMinutesLeft(stage, state, 40 * minutes, {'done': 55, 'total': 55, 'fraction': 1}), 0);
    assert.equal(train.stageMinutesLeft(stage, state, 40 * minutes, {'done': 54, 'total': 55, 'fraction': 54 / 55}), 1);
    // No progress yet, or none to measure - its expected minutes less its time so far (at least a minute)
    assert.equal(train.stageMinutesLeft(stage, state, 10 * minutes, {'done': 0, 'total': 55, 'fraction': 0}), 15);
    assert.equal(train.stageMinutesLeft(stage, state, 60 * minutes, undefined), 1);
    // By its phases' rest (at least a minute), and before its first phase's header (null) - its expected minutes
    assert.equal(train.stageMinutesLeft(stage, state, 10 * minutes, {'done': 3, 'total': 9, 'phase': 'make', 'minutesLeft': 7.5}), 7.5);
    assert.equal(train.stageMinutesLeft(stage, state, 10 * minutes, {'done': 9, 'total': 9, 'phase': 'make', 'minutesLeft': 0}), 1);
    assert.equal(train.stageMinutesLeft(stage, state, 10 * minutes, null), 15);
});


test('train.throttled', () => {
    const sample = (thermal) => ({'time': 0, 'cpu': 10, 'gpu': 90, 'mem': 40, 'system': 40, 'powerIn': 40, thermal});
    assert.equal(train.throttled(sample(0)), false);
    assert.equal(train.throttled(sample(1)), true);
    assert.equal(train.throttled(sample(2)), true);
    assert.equal(train.throttled(sample(NaN)), false);
    assert.equal(train.thermalLevels[2], 'heavy');
});


test('train.stageStates, events', () => {
    const events = train.parseStageEvents([
        '{"stage": "data", "event": "start", "time": 100}',
        '{"stage": "data", "event": "end", "time": 160}',
        '{"stage": "check", "event": "start", "time": 160}',
        '{"stage": "voice-a", "event": "start", "time": 160}',
        '{"stage": "voice-a", "event": "end", "time": 200}',
        '{"stage": "voice-b", "event": "start", "time": 200}',
        '{"stage": "eval", "event": "start", "time": 200}',
        '{"stage": "eval", "event": "fail", "time": 210, "status": 2}'
    ].join('\n'));
    const run = {events, 'alive': true, 'queued': new Set(['voice-b'])};
    const states = train.stageStates(manifest, run);
    assert.deepEqual(Object.fromEntries(states), {
        'data': {'status': 'done', 'start': 100000, 'end': 160000},
        'check': {'status': 'running', 'start': 160000, 'end': null},
        'voice-a': {'status': 'done', 'start': 160000, 'end': 200000},
        'voice-b': {'status': 'queued', 'start': 200000, 'end': null},
        'eval': {'status': 'failed', 'start': 200000, 'end': 210000, 'failure': 'exit status 2'},
        'package': {'status': 'waiting', 'start': null, 'end': null}
    });

    // Make stopped: the started stages stopped; a target leaves the others out
    const stopped = train.stageStates(manifest, {events, 'alive': false, 'target': 'train-g2p'});
    assert.equal(stopped.get('check').status, 'stopped');
    assert.equal(stopped.get('voice-b').status, 'stopped');
    assert.equal(stopped.get('package').status, 'skipped');
});


test('train.forecast', () => {
    const now = time('10:00:00');
    const states = new Map([
        ['data', {'status': 'done', 'start': time('09:00:00'), 'end': time('09:05:00')}],
        ['check', {'status': 'running', 'start': time('09:55:00'), 'end': null}],
        ['voice-a', {'status': 'running', 'start': time('09:05:00'), 'end': null}],
        ['voice-b', {'status': 'waiting', 'start': null, 'end': null}],
        ['eval', {'status': 'waiting', 'start': null, 'end': null}],
        ['package', {'status': 'waiting', 'start': null, 'end': null}]
    ]);
    const remaining = (stage, state) => (state.status === 'running' ? 5 : stage.minutes);
    const plan = train.forecast(manifest, states, remaining, now);
    // (voice-a until 10:05, then voice-b and eval one at a time on the GPU, then package)
    assert.equal(plan.finishes.get('voice-a'), time('10:05:00'));
    assert.equal(plan.finishes.get('voice-b'), time('10:15:00'));
    assert.equal(plan.finishes.get('eval'), time('10:20:00'));
    assert.equal(plan.eta, time('10:21:00'));
    assert.deepEqual(plan.path, ['data', 'voice-a', 'voice-b', 'eval', 'package']);

    // From the start - the critical path's stages one after another (the check beside them)
    const waiting = new Map([...states].map(([name]) => [name, {'status': 'waiting', 'start': null, 'end': null}]));
    const fresh = train.forecast(manifest, waiting, remaining, now);
    assert.equal(fresh.eta, now + 41 * 60000);
    assert.equal(fresh.finishes.get('check'), now + 15 * 60000);

    // Skipped stages aren't scheduled; nothing to schedule - the ETA is now
    const skipped = new Map([...states].map(([name]) => [name, {'status': 'skipped', 'start': null, 'end': null}]));
    assert.deepEqual(train.forecast(manifest, skipped, remaining, now), {'finishes': new Map(), 'eta': now, 'path': []});
});


test('train.listenModels', () => {
    // The manifest's samples stages, each waiting until its samples are in the index (the reference first, then the
    // models in the order of the training stages they follow)
    const listenManifest = {'stages': [
        ...manifest.stages,
        {'stage': 'samples-voice-b', 'on': 'cpu', 'minutes': 1, 'needs': ['voice-b'], 'samples': 'voice-b'},
        {'stage': 'samples-voice-a', 'on': 'cpu', 'minutes': 1, 'needs': ['voice-a'], 'samples': 'voice-a'},
        {'stage': 'samples-reference', 'on': 'cpu', 'minutes': 1, 'needs': [], 'samples': 'reference'}
    ]};
    const states = new Map([['samples-voice-a', {'status': 'running'}], ['samples-voice-b', {'status': 'waiting'}]]);
    const voiceA = {'stage': 'voice-a', 'file': 'voice-a-female-ishmael.wav', 'mos': 3.9};
    const reference = {'stage': 'reference', 'file': 'reference-female-ishmael.wav', 'mos': 4.0};
    assert.deepEqual(train.listenModels(listenManifest, states, [voiceA, reference]), [
        {'key': 'reference', 'stage': 'samples-reference', 'status': 'done', 'samples': [reference]},
        {'key': 'voice-a', 'stage': 'samples-voice-a', 'status': 'done', 'samples': [voiceA]},
        {'key': 'voice-b', 'stage': 'samples-voice-b', 'status': 'waiting', 'samples': []}
    ]);
    assert.deepEqual(train.listenModels(listenManifest, states, null).map((model) => [model.key, model.status]),
        [['reference', 'waiting'], ['voice-a', 'running'], ['voice-b', 'waiting']]);

    // A samples stage not in the run's make target isn't listed, unless it has samples
    states.set('samples-voice-b', {'status': 'skipped'});
    states.set('samples-voice-a', {'status': 'skipped'});
    assert.deepEqual(train.listenModels(listenManifest, states, [voiceA]).map((model) => [model.key, model.status]),
        [['reference', 'waiting'], ['voice-a', 'done']]);
});

