// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {mergeIndex, parseArgs, parseScores, sampleFile, updateIndex, withLock} from '../tools/trainsamplesindex.js';
import {mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, utimesSync, writeFileSync} from 'node:fs';
import {strict as assert} from 'node:assert';
import {join} from 'node:path';
import test from 'node:test';
import {tmpdir} from 'node:os';


test('trainsamplesindex.parseArgs', () => {
    assert.deepEqual(parseArgs(['--stage', 'teacher-a', '--checkpoint', 'build/train/runs/teacher-a/last.pt', '--out', 'samples']), {
        'stage': 'teacher-a', 'checkpoint': 'build/train/runs/teacher-a/last.pt', 'model': null, 'run': 'teacher-a',
        'out': 'samples', 'settings': 'train/data/train.json'
    });
    assert.deepEqual(parseArgs(['--stage', 'reference', '--model', 'lib/models/voiceModel.js', '--out', 'samples',
        '--settings', 'other.json']), {
        'stage': 'reference', 'checkpoint': null, 'model': 'lib/models/voiceModel.js', 'run': null, 'out': 'samples',
        'settings': 'other.json'
    });
    assert.equal(parseArgs(['--stage', 'polish', '--checkpoint', 'best.pt', '--run', 'distill', '--out', 'samples']).run, 'distill');
    assert.throws(() => parseArgs(['--stage']), /incomplete argument --stage/);
    assert.throws(() => parseArgs(['--bogus', 'value']), /unknown or incomplete argument --bogus/);
    assert.throws(() => parseArgs(['--stage', 'polish', '--out', 'samples']), /usage/);
    assert.throws(() => parseArgs(['--stage', 'polish', '--model', 'a.js', '--checkpoint', 'b.pt', '--out', 'samples']), /usage/);
    assert.throws(() => parseArgs(['--model', 'a.js', '--out', 'samples']), /usage/);
});


test('trainsamplesindex.sampleFile', () => {
    assert.equal(sampleFile('teacher-a', 'arctic-slt', 'ishmael'), 'teacher-a-arctic-slt-ishmael.wav');
});


test('trainsamplesindex.parseScores', () => {
    const output = '{"file": "/tmp/samples/a.wav", "seconds": 2.0, "mos": 3.5}\nnot JSON\nnull\n{"file": "b.wav"}\n';
    assert.deepEqual(parseScores(output), {'a.wav': 3.5, 'b.wav': null});
});


test('trainsamplesindex.mergeIndex', () => {
    const index = [
        {'stage': 'reference', 'file': 'r.wav'}, {'stage': 'teacher-a', 'file': 'old.wav'}, {'stage': 'student', 'file': 's.wav'}
    ];
    assert.deepEqual(mergeIndex(index, 'teacher-a', [{'stage': 'teacher-a', 'file': 'new.wav'}]), [
        {'stage': 'reference', 'file': 'r.wav'}, {'stage': 'student', 'file': 's.wav'}, {'stage': 'teacher-a', 'file': 'new.wav'}
    ]);
    assert.deepEqual(mergeIndex(null, 'polish', [{'stage': 'polish'}]), [{'stage': 'polish'}]);
});


test('trainsamplesindex.withLock', () => {
    const dir = mkdtempSync(join(tmpdir(), 'trainsamples-'));
    try {
        const lock = join(dir, 'index.lock');

        // Held for the callback, released after (also when it throws)
        assert.equal(withLock(lock, () => readdirSync(dir).join()), 'index.lock');
        assert.deepEqual(readdirSync(dir), []);
        assert.throws(() => withLock(lock, () => {
            throw new Error('failed');
        }), /failed/);
        assert.deepEqual(readdirSync(dir), []);

        // Another holder - waited for (released meanwhile, or before its age is read), timed out, or taken over when stale
        const never = () => assert.fail('the lock was held');
        mkdirSync(lock);
        let clock = Date.now();
        const now = () => clock;
        const waits = [];
        assert.equal(withLock(lock, () => 'waited', {now, 'wait': (ms) => {
            waits.push(ms);
            rmSync(lock, {'recursive': true});
        }}), 'waited');
        assert.deepEqual(waits, [100]);
        mkdirSync(lock);
        let calls = 0;
        const releasing = () => {
            calls += 1;
            if (calls === 2) {
                rmSync(lock, {'recursive': true});
            }
            return clock;
        };
        assert.equal(withLock(lock, () => 'released', {'now': releasing, 'wait': () => null}), 'released');
        mkdirSync(lock);
        assert.throws(() => withLock(lock, never, {now, 'timeoutMs': 1000, 'wait': () => {
            clock += 600;
        }}), /timed out waiting/);
        assert.throws(() => withLock(lock, never, {'timeoutMs': 50}), /timed out waiting/);
        utimesSync(lock, new Date(clock - 600000), new Date(clock - 600000));
        assert.equal(withLock(lock, () => 'stale', {now}), 'stale');

        // A lock path that can't be made
        assert.throws(() => withLock(join(dir, 'missing', 'index.lock'), never), /ENOENT/);
        assert.throws(never, /the lock was held/);
    } finally {
        rmSync(dir, {'recursive': true, 'force': true});
    }
});


test('trainsamplesindex.updateIndex', () => {
    const dir = mkdtempSync(join(tmpdir(), 'trainsamples-'));
    try {
        // A new index, then a stage's samples replaced, the others kept
        updateIndex(dir, 'reference', [{'stage': 'reference', 'file': 'r.wav'}]);
        updateIndex(dir, 'polish', [{'stage': 'polish', 'file': 'p1.wav'}]);
        updateIndex(dir, 'polish', [{'stage': 'polish', 'file': 'p2.wav'}]);
        assert.deepEqual(JSON.parse(readFileSync(join(dir, 'index.json'), 'utf8')),
            [{'stage': 'reference', 'file': 'r.wav'}, {'stage': 'polish', 'file': 'p2.wav'}]);
        assert.deepEqual(readdirSync(dir), ['index.json']);

        // An unreadable index starts over
        writeFileSync(join(dir, 'index.json'), 'not JSON');
        updateIndex(dir, 'student', [{'stage': 'student', 'file': 's.wav'}]);
        assert.deepEqual(JSON.parse(readFileSync(join(dir, 'index.json'), 'utf8')), [{'stage': 'student', 'file': 's.wav'}]);
    } finally {
        rmSync(dir, {'recursive': true, 'force': true});
    }
});
