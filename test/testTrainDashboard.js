// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

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


test('train.parseMakeLog', () => {
    const text = [
        'before any start',
        '2026-10-16 08:20:51 start: make -j4 train in a clean worktree',
        'make: Entering directory `/repo\'',
        'make: Leaving directory `/repo\'',
        '2026-10-16 09:42:51 restart: the voice chain (make -j4 build/train/student.done)',
        'make: Entering directory `/repo\'',
        '2026-10-16 10:15:50 start: the check (make check.done, its log elsewhere)',
        'make: Leaving directory `/other\'',
        'make: Leaving directory `/repo\'',
        '2026-10-16 10:20:00 start: something else'
    ].join('\n');
    const segments = train.parseMakeLog(text);
    assert.deepEqual(segments.map(({start, target, directory, ended}) => [start, target, directory, ended]), [
        [time('08:20:51'), 'train', '/repo', true],
        [time('09:42:51'), 'student', '/repo', true],
        [time('10:15:50'), 'check', null, false],
        [time('10:20:00'), 'train', null, false]
    ]);
    assert.equal(segments[0].text, 'make: Entering directory `/repo\'\nmake: Leaving directory `/repo\'');
});


test('train.parseUtilLog', () => {
    const text = [
        'time cpu_busy% gpu% stage',
        '23:59:40 50 90 nn.trainmulti',
        '00:00:10 60 80 nn.trainmulti',
        'garbage',
        'time cpu_busy% gpu% mem_used% stages',
        '2026-10-17T00:00:40 70 0 40.5 teacher',
        ''
    ].join('\n');
    const samples = train.parseUtilLog(text, time('08:00:00'));
    assert.deepEqual(samples, [
        {'time': time('23:59:40'), 'cpu': 50, 'gpu': 90, 'mem': NaN},
        {'time': time('23:59:40') + 30000, 'cpu': 60, 'gpu': 80, 'mem': NaN},
        {'time': time('23:59:40') + 60000, 'cpu': 70, 'gpu': 0, 'mem': 40.5}
    ]);
    assert.deepEqual(train.meanUtilization(samples, -Infinity, Infinity), {'cpu': 60, 'gpu': 170 / 3, 'count': 3});
    assert.deepEqual(train.meanUtilization(samples, 0, 1), {'cpu': NaN, 'gpu': NaN, 'count': 0});
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
    const run = {events, 'stamps': new Map(), 'segments': [], 'alive': true, 'queued': new Set(['student'])};
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
    const stopped = train.stageStates(manifest, {events, 'stamps': new Map(), 'segments': [], 'alive': false, 'target': 'student'});
    assert.equal(stopped.get('check').status, 'stopped');
    assert.equal(stopped.get('student').status, 'stopped');
    assert.equal(stopped.get('package').status, 'skipped');
});


test('train.stageStates, stamps and make\'s log', () => {
    const segments = train.parseMakeLog([
        '2026-10-16 08:00:00 start: make -j4 train',
        'make: Entering directory `/repo\'',
        'make: *** [build/train/check.done] Error 1',
        'make: Leaving directory `/repo\'',
        '2026-10-16 09:00:00 restart: make -j4 build/train/student.done',
        'make: Entering directory `/repo\''
    ].join('\n'));
    const stamps = new Map([['data', time('09:10:00')], ['teacher', time('09:30:00')]]);
    const states = train.stageStates(manifest, {'events': new Map(), stamps, segments, 'errorTexts': ['make: *** [x/eval.done] Error 2']});
    assert.deepEqual(Object.fromEntries(states), {
        'data': {'status': 'done', 'start': time('09:00:00'), 'end': time('09:10:00')},
        'check': {'status': 'skipped', 'start': null, 'end': null},
        'teacher': {'status': 'done', 'start': time('09:10:00'), 'end': time('09:30:00')},
        'student': {'status': 'running', 'start': time('09:30:00'), 'end': null},
        'eval': {'status': 'failed', 'start': null, 'end': null, 'failure': 'make reported an error'},
        'package': {'status': 'skipped', 'start': null, 'end': null}
    });

    // A stamp without a make run or dependencies - its start unknown; the latest run's failure; no runs - every stage
    const alone = train.stageStates(manifest, {'events': new Map(), 'stamps': new Map([['data', 5000]]), 'segments': []});
    assert.deepEqual(alone.get('data'), {'status': 'done', 'start': null, 'end': 5000});
    const early = train.stageStates(manifest, {'events': new Map(), 'stamps': new Map([['teacher', 5000]]), 'segments': []});
    assert.deepEqual(early.get('teacher'), {'status': 'done', 'start': null, 'end': 5000});
    assert.equal(alone.get('package').status, 'waiting');
    const failed = train.stageStates(manifest, {'events': new Map(), 'stamps': new Map(), 'segments': segments.slice(0, 1)});
    assert.equal(failed.get('check').status, 'failed');
    assert.equal(failed.get('data').status, 'waiting');

    // An open run with no start in its log (a stage with no dependencies)
    const open = [{'start': -Infinity, 'target': 'train', 'directory': null, 'ended': false, 'text': ''}];
    const opened = train.stageStates(manifest, {'events': new Map(), 'stamps': new Map(), 'segments': open});
    assert.deepEqual(opened.get('data'), {'status': 'running', 'start': null, 'end': null});
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
