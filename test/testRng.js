// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {STREAM_ASPIRATION, STREAM_FRICATION, STREAM_JITTER, STREAM_SHIMMER, gaussAt, streamSeed, uniformAt} from '../lib/rng.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


test('rng.streamSeed', () => {
    // nn/rng.py's stream_seed
    assert.equal(streamSeed(1, 1), 1278455945);
    assert.equal(streamSeed(1000, 2), 2545871315);
    assert.equal(streamSeed(5, 3), 2274654832);

    // Seeds are 32-bit
    assert.equal(streamSeed(2 ** 32 + 5, 3), streamSeed(5, 3));
    assert.equal(streamSeed(-1, 1), streamSeed(2 ** 32 - 1, 1));

    // The streams differ
    const seeds = [STREAM_ASPIRATION, STREAM_FRICATION, STREAM_JITTER, STREAM_SHIMMER].map((stream) => streamSeed(1, stream));
    assert.deepEqual(seeds.slice(0, 1), [1278455945]);
    assert.equal(new Set(seeds).size, 4);
});


test('rng.uniformAt', () => {
    // nn/rng.py's uniform
    assert.deepEqual([0, 1, 2, 3, 4, 5].map((index) => uniformAt(7, index)), [
        0.011704753153026104, 0.06195825757458806, 0.97690763277933, 0.6990287057124078, 0.5214452685322613, 0.4055216880515218
    ]);
    assert.equal(uniformAt(7, 1e6), 0.17535980022512376);

    // [0, 1), about uniform
    let sum = 0;
    for (let index = 0; index < 10000; index++) {
        const value = uniformAt(99, index);
        assert(value >= 0 && value < 1);
        sum += value;
    }
    assert(Math.abs(sum / 10000 - 0.5) < 0.01);
});


test('rng.gaussAt', () => {
    // nn/rng.py's gauss
    assert.equal(gaussAt(7, 0), -0.4574043168616386);
    assert.equal(gaussAt(123, 45), -1.1194866375645351);
    assert.deepEqual([10, 11, 12].map((index) => gaussAt(42, index)), [-0.7347769063645522, -0.1463601602974729, 1.3256268005328773]);

    // About unit variance, zero mean
    let sum = 0;
    let squares = 0;
    for (let index = 0; index < 10000; index++) {
        const value = gaussAt(5, index);
        sum += value;
        squares += value * value;
    }
    assert(Math.abs(sum / 10000) < 0.05);
    assert(Math.abs(squares / 10000 - 1) < 0.05);
});
