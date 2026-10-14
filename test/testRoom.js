import {convolve, roomImpulse} from '../lib/room.js';
import {strict as assert} from 'node:assert';
import {makeRandom} from '../lib/klatt.js';
import test from 'node:test';


// Direct convolution, for reference
function directConvolve(samples, impulse) {
    const out = new Float64Array(samples.length + impulse.length - 1);
    for (let ix = 0; ix < samples.length; ix++) {
        for (let jx = 0; jx < impulse.length; jx++) {
            out[ix + jx] += samples[ix] * impulse[jx];
        }
    }
    return out;
}


// The energy of samples in a range
function energy(samples, start, end) {
    let sum = 0;
    for (let ix = start; ix < end; ix++) {
        sum += samples[ix] * samples[ix];
    }
    return sum;
}


test('roomImpulse, defaults', () => {
    const sampleRate = 16000;
    const impulse = roomImpulse(sampleRate);
    assert.ok(impulse instanceof Float32Array);
    assert.equal(impulse.length, Math.floor(0.35 * sampleRate));

    // The direct sound, then silence until the first reflection (3 ms)
    assert.equal(impulse[0], 1);
    for (let ix = 1; ix < Math.floor(0.003 * sampleRate); ix++) {
        assert.equal(impulse[ix], 0);
    }

    // Early reflections before the diffuse tail (12 ms) - at most eight, each at most 1.5 times the wet level
    const wet = 10 ** (-16 / 20);
    const tailStart = Math.floor(0.012 * sampleRate);
    const early = impulse.subarray(1, tailStart).filter((value) => value !== 0);
    assert.ok(early.length > 0 && early.length <= 8);
    for (const value of early) {
        assert.ok(Math.abs(value) <= 1.5 * wet + 1e-6);
    }

    // The tail decays - the last tenth is far quieter than the first tenth after the tail starts
    const tenth = Math.floor((impulse.length - tailStart) / 10);
    const first = energy(impulse, tailStart + 0.003 * sampleRate, tailStart + tenth);
    const last = energy(impulse, impulse.length - tenth, impulse.length);
    assert.ok(first > 0 && last > 0);
    assert.ok(last < 1e-3 * first);
});


test('roomImpulse, options', () => {
    const sampleRate = 8000;

    // Repeatable by seed
    assert.deepEqual(roomImpulse(sampleRate, {'seed': 3}), roomImpulse(sampleRate, {'seed': 3}));
    assert.notDeepEqual(roomImpulse(sampleRate, {'seed': 3}), roomImpulse(sampleRate, {'seed': 4}));

    // The length is the RT60
    assert.equal(roomImpulse(sampleRate, {'rt60': 0.5}).length, 4000);

    // The wet level scales everything but the direct sound
    const loud = roomImpulse(sampleRate, {'wetDb': -10});
    const quiet = roomImpulse(sampleRate, {'wetDb': -30});
    assert.equal(loud[0], 1);
    assert.equal(quiet[0], 1);
    for (let ix = 1; ix < loud.length; ix++) {
        assert.ok(Math.abs(quiet[ix] * 10 - loud[ix]) < 1e-6, `sample ${ix}`);
    }
});


test('convolve, matches direct convolution', () => {
    const random = makeRandom(11);
    for (const [sampleCount, impulseCount] of [[1, 1], [5, 3], [100, 7], [1000, 64], [300, 200]]) {
        const samples = new Float32Array(sampleCount).map(() => random.gauss());
        const impulse = new Float32Array(impulseCount).map(() => random.gauss());
        const expected = directConvolve(samples, impulse);
        const out = convolve(samples, impulse);
        assert.ok(out instanceof Float32Array);
        assert.equal(out.length, sampleCount + impulseCount - 1);
        for (let ix = 0; ix < out.length; ix++) {
            assert.ok(Math.abs(out[ix] - expected[ix]) < 1e-4 * (1 + Math.abs(expected[ix])), `${sampleCount}, ${impulseCount}, ${ix}`);
        }
    }
});


test('convolve, identity and delay', () => {
    const samples = Float32Array.from([0.5, -0.25, 1, 0, 0.125]);

    // A unit impulse is the identity
    const identity = convolve(samples, Float32Array.from([1]));
    for (let ix = 0; ix < samples.length; ix++) {
        assert.ok(Math.abs(identity[ix] - samples[ix]) < 1e-7);
    }

    // A delayed, scaled impulse delays and scales
    const delayed = convolve(samples, Float32Array.from([0, 0, 0.5]));
    assert.equal(delayed.length, 7);
    assert.ok(Math.abs(delayed[0]) < 1e-7 && Math.abs(delayed[1]) < 1e-7);
    for (let ix = 0; ix < samples.length; ix++) {
        assert.ok(Math.abs(delayed[ix + 2] - 0.5 * samples[ix]) < 1e-7);
    }

    // No samples is only the impulse's tail of silence
    assert.deepEqual(convolve(new Float32Array(0), Float32Array.from([1, 0.5])), new Float32Array(1));
});


test('convolve, room ambience', () => {
    // The room's direct sound passes through, followed by its reverberation
    const sampleRate = 8000;
    const impulse = roomImpulse(sampleRate, {'rt60': 0.2});
    const click = new Float32Array(100);
    click[0] = 1;
    const out = convolve(click, impulse);
    for (let ix = 0; ix < impulse.length; ix++) {
        assert.ok(Math.abs(out[ix] - impulse[ix]) < 1e-5, `sample ${ix}`);
    }
});
