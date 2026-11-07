// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {mergeIndex, parseArgs, sampleFile, updateIndex, withLock} from '../tools/trainsamplesindex.js';
import {mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, utimesSync, writeFileSync} from 'node:fs';
import {strict as assert} from 'node:assert';
import {join} from 'node:path';
import test from 'node:test';
import {tmpdir} from 'node:os';


test('trainsamplesindex.parseArgs', () => {
    assert.deepEqual(parseArgs(['--stage', 'voice-a', '--checkpoint', 'build/train/runs/voice-a/last.pt', '--out', 'samples']), {
        'stage': 'voice-a', 'checkpoint': 'build/train/runs/voice-a/last.pt', 'model': null, 'out': 'samples'
    });
    assert.deepEqual(parseArgs(['--stage', 'reference', '--model', 'lib/models/voiceModel.js', '--out', 'samples']), {
        'stage': 'reference', 'checkpoint': null, 'model': 'lib/models/voiceModel.js', 'out': 'samples'
    });
    assert.throws(() => parseArgs(['--stage']), /incomplete argument --stage/);
    assert.throws(() => parseArgs(['--run', 'voice-b']), /unknown or incomplete argument --run/);
    assert.throws(() => parseArgs(['--stage', 'voice-b', '--out', 'samples']), /usage/);
    assert.throws(() => parseArgs(['--stage', 'voice-b', '--model', 'a.js', '--checkpoint', 'b.pt', '--out', 'samples']), /usage/);
    assert.throws(() => parseArgs(['--model', 'a.js', '--out', 'samples']), /usage/);
});


test('trainsamplesindex.sampleFile', () => {
    assert.equal(sampleFile('voice-a', 'female', 'ishmael'), 'voice-a-female-ishmael.wav');
});


test('trainsamplesindex.mergeIndex', () => {
    const index = [
        {'stage': 'reference', 'file': 'r.wav'}, {'stage': 'voice-a', 'file': 'old.wav'}, {'stage': 'voice-b', 'file': 'b.wav'}
    ];
    assert.deepEqual(mergeIndex(index, 'voice-a', [{'stage': 'voice-a', 'file': 'new.wav'}]), [
        {'stage': 'reference', 'file': 'r.wav'}, {'stage': 'voice-b', 'file': 'b.wav'}, {'stage': 'voice-a', 'file': 'new.wav'}
    ]);
    assert.deepEqual(mergeIndex(null, 'voice-b', [{'stage': 'voice-b'}]), [{'stage': 'voice-b'}]);
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
        updateIndex(dir, 'voice-b', [{'stage': 'voice-b', 'file': 'b1.wav'}]);
        updateIndex(dir, 'voice-b', [{'stage': 'voice-b', 'file': 'b2.wav'}]);
        assert.deepEqual(JSON.parse(readFileSync(join(dir, 'index.json'), 'utf8')),
            [{'stage': 'reference', 'file': 'r.wav'}, {'stage': 'voice-b', 'file': 'b2.wav'}]);
        assert.deepEqual(readdirSync(dir), ['index.json']);

        // An unreadable index starts over
        writeFileSync(join(dir, 'index.json'), 'not JSON');
        updateIndex(dir, 'voice-a', [{'stage': 'voice-a', 'file': 'a.wav'}]);
        assert.deepEqual(JSON.parse(readFileSync(join(dir, 'index.json'), 'utf8')), [{'stage': 'voice-a', 'file': 'a.wav'}]);
    } finally {
        rmSync(dir, {'recursive': true, 'force': true});
    }
});
