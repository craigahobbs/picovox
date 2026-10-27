// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import * as train from '../tools/dashboard/train.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


// A small stage graph - data (CPU), a check after it (ordering only), two GPU stages, and a final CPU stage
const manifest = {
    'target': 'train',
    'stages': [
        {'stage': 'check', 'on': 'cpu', 'minutes': 10, 'needs': [], 'after': ['data']},
        {'stage': 'data', 'on': 'cpu', 'minutes': 5, 'needs': []},
        {'stage': 'teacher', 'on': 'gpu', 'minutes': 20, 'needs': ['data'], 'run': 'teacher'},
        {'stage': 'student', 'on': 'gpu', 'minutes': 10, 'needs': ['teacher']},
        {'stage': 'eval', 'on': 'gpu', 'minutes': 5, 'needs': ['teacher']},
        {'stage': 'package', 'on': 'cpu', 'minutes': 1, 'needs': ['eval', 'student']}
    ]
};
const time = (clock) => train.parseLocalTime(`2026-10-16 ${clock}`);


test('train.parseLocalTime', () => {
    assert.equal(train.parseLocalTime('2026-10-16 08:20:51'), new Date(2026, 9, 16, 8, 20, 51).getTime());
    assert.equal(train.parseLocalTime('2026-10-16T08:20:51'), new Date(2026, 9, 16, 8, 20, 51).getTime());
    assert(Number.isNaN(train.parseLocalTime('08:20:51')));
});


test('train.stageOrder', () => {
    assert.deepEqual(train.stageOrder(manifest).map((stage) => stage.stage), ['data', 'check', 'teacher', 'student', 'eval', 'package']);
});


test('train.targetStages', () => {
    assert.deepEqual([...train.targetStages(manifest, 'student')].sort(), ['data', 'student', 'teacher']);
    assert.equal(train.targetStages(manifest, 'train').size, 6);

    // A manifest's named targets - their stages and every stage those need
    const named = {...manifest, 'targets': {'train-check': ['check'], 'train-voice': ['package']}};
    assert.deepEqual([...train.targetStages(named, 'train-check')], ['check']);
    assert.deepEqual([...train.targetStages(named, 'train-voice')].sort(), ['data', 'eval', 'package', 'student', 'teacher']);
    assert.equal(train.targetStages(named, 'unknown').size, 6);
});


test('train.parseRunTarget', () => {
    assert.equal(train.parseRunTarget(''), null);
    assert.deepEqual(train.parseRunTarget([
        '{"event": "target", "target": "train", "time": 100}',
        '{"stage": "data", "event": "start", "time": 101}',
        'not json',
        '{"event": "target", "target": "train-check", "time": 200}'
    ].join('\n')), {'target': 'train-check', 'time': 200000});
});


test('train.stageStates, a later run', () => {
    // An earlier run's unfinished attempts are stale - waiting in the later run; its finished stages stay done
    const events = train.parseStageEvents([
        '{"stage": "data", "event": "start", "time": 100}',
        '{"stage": "data", "event": "end", "time": 160}',
        '{"stage": "check", "event": "start", "time": 160}',
        '{"stage": "check", "event": "fail", "time": 170, "status": 2}',
        '{"stage": "teacher", "event": "start", "time": 160}'
    ].join('\n'));
    const states = train.stageStates(manifest, {events, 'alive': true, 'started': 300000,
        'target': 'train'});
    assert.equal(states.get('data').status, 'done');
    assert.equal(states.get('check').status, 'waiting');
    assert.equal(states.get('teacher').status, 'waiting');
});


test('train.parseStageEvents', () => {
    const text = [
        '{"stage": "data", "event": "start", "time": 100}',
        '{"stage": "data", "event": "end", "time": 160}',
        '{"stage": "teacher", "event": "start", "time": 170}',
        '{"stage": "teacher", "event": "fail", "time": 180, "status": 1}',
        '{"stage": "eval", "event": "fail", "time": 190, "signal": "terminated"}',
        '{"stage": "student", "event": "end", "time": 195}',
        '{"stage": "package", "event": "start", "ti',
        ''
    ].join('\n');
    assert.deepEqual(Object.fromEntries(train.parseStageEvents(text)), {
        'data': {'start': 100000, 'end': 160000, 'fail': null, 'failure': null},
        'teacher': {'start': 170000, 'end': null, 'fail': 180000, 'failure': 'exit status 1'},
        'eval': {'start': null, 'end': null, 'fail': 190000, 'failure': 'terminated'}
    });
});


test('train.parseUtilLog', () => {
    const text = [
        'time cpu_busy% gpu% stage',
        '23:59:40 50 90 picovox_train.trainmulti',
        '00:00:10 60 80 picovox_train.trainmulti',
        'garbage',
        'time cpu_busy% gpu% mem_used% stages',
        '2026-10-17T00:00:40 70 0 40.5 teacher',
        'time cpu_busy% gpu% mem_used% system_w power_in_w thermal stages',
        '2026-10-17T00:01:00 10 90 41.0 35.3 48.3 2 teacher-a',
        '2026-10-17T00:01:20 10 90 41.0 - - - teacher-a',
        ''
    ].join('\n');
    const samples = train.parseUtilLog(text, time('08:00:00'));
    assert.deepEqual(samples, [
        {'time': time('23:59:40'), 'cpu': 50, 'gpu': 90, 'mem': NaN, 'system': NaN, 'powerIn': NaN, 'thermal': NaN},
        {'time': time('23:59:40') + 30000, 'cpu': 60, 'gpu': 80, 'mem': NaN, 'system': NaN, 'powerIn': NaN, 'thermal': NaN},
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
        '{"stage": "teacher", "event": "start", "time": 160}',
        '{"stage": "teacher", "event": "end", "time": 200}',
        '{"stage": "student", "event": "start", "time": 200}',
        '{"stage": "eval", "event": "start", "time": 200}',
        '{"stage": "eval", "event": "fail", "time": 210, "status": 2}'
    ].join('\n'));
    const run = {events, 'alive': true, 'queued': new Set(['student'])};
    const states = train.stageStates(manifest, run);
    assert.deepEqual(Object.fromEntries(states), {
        'data': {'status': 'done', 'start': 100000, 'end': 160000},
        'check': {'status': 'running', 'start': 160000, 'end': null},
        'teacher': {'status': 'done', 'start': 160000, 'end': 200000},
        'student': {'status': 'queued', 'start': 200000, 'end': null},
        'eval': {'status': 'failed', 'start': 200000, 'end': 210000, 'failure': 'exit status 2'},
        'package': {'status': 'waiting', 'start': null, 'end': null}
    });

    // Make stopped: the started stages stopped; a target leaves the others out
    const stopped = train.stageStates(manifest, {events, 'alive': false, 'target': 'student'});
    assert.equal(stopped.get('check').status, 'stopped');
    assert.equal(stopped.get('student').status, 'stopped');
    assert.equal(stopped.get('package').status, 'skipped');
});


test('train.forecast', () => {
    const now = time('10:00:00');
    const states = new Map([
        ['data', {'status': 'done', 'start': time('09:00:00'), 'end': time('09:05:00')}],
        ['check', {'status': 'running', 'start': time('09:55:00'), 'end': null}],
        ['teacher', {'status': 'running', 'start': time('09:05:00'), 'end': null}],
        ['student', {'status': 'waiting', 'start': null, 'end': null}],
        ['eval', {'status': 'waiting', 'start': null, 'end': null}],
        ['package', {'status': 'waiting', 'start': null, 'end': null}]
    ]);
    const remaining = (stage, state) => (state.status === 'running' ? 5 : stage.minutes);
    const plan = train.forecast(manifest, states, remaining, now);
    // (teacher until 10:05, then student and eval one at a time on the GPU, then package)
    assert.equal(plan.finishes.get('teacher'), time('10:05:00'));
    assert.equal(plan.finishes.get('student'), time('10:15:00'));
    assert.equal(plan.finishes.get('eval'), time('10:20:00'));
    assert.equal(plan.eta, time('10:21:00'));
    assert.deepEqual(plan.path, ['data', 'teacher', 'student', 'eval', 'package']);

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
        {'stage': 'samples-student', 'on': 'cpu', 'minutes': 1, 'needs': ['student'], 'samples': 'student'},
        {'stage': 'samples-teacher', 'on': 'cpu', 'minutes': 1, 'needs': ['teacher'], 'samples': 'teacher'},
        {'stage': 'samples-reference', 'on': 'cpu', 'minutes': 1, 'needs': [], 'samples': 'reference'}
    ]};
    const states = new Map([['samples-teacher', {'status': 'running'}], ['samples-student', {'status': 'waiting'}]]);
    const teacher = {'stage': 'teacher', 'file': 'teacher-a-ishmael.wav', 'mos': 3.9};
    const reference = {'stage': 'reference', 'file': 'reference-a-ishmael.wav', 'mos': 4.0};
    assert.deepEqual(train.listenModels(listenManifest, states, [teacher, reference]), [
        {'key': 'reference', 'stage': 'samples-reference', 'status': 'done', 'samples': [reference]},
        {'key': 'teacher', 'stage': 'samples-teacher', 'status': 'done', 'samples': [teacher]},
        {'key': 'student', 'stage': 'samples-student', 'status': 'waiting', 'samples': []}
    ]);
    assert.deepEqual(train.listenModels(listenManifest, states, null).map((model) => [model.key, model.status]),
        [['reference', 'waiting'], ['teacher', 'running'], ['student', 'waiting']]);

    // A samples stage not in the run's make target isn't listed, unless it has samples
    states.set('samples-student', {'status': 'skipped'});
    states.set('samples-teacher', {'status': 'skipped'});
    assert.deepEqual(train.listenModels(listenManifest, states, [teacher]).map((model) => [model.key, model.status]),
        [['reference', 'waiting'], ['teacher', 'done']]);
});

