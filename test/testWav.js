import {encodeWav, readWav, writeWav} from '../lib/wav.js';
import {mkdtempSync, rmSync, writeFileSync} from 'node:fs';
import {Buffer} from 'node:buffer';
import {strict as assert} from 'node:assert';
import {join} from 'node:path';
import process from 'node:process';
import test from 'node:test';
import {tmpdir} from 'node:os';


// Run a function with a temporary directory, removed afterward
function withTempDir(callback) {
    const dir = mkdtempSync(join(tmpdir(), 'voice-synth-wav-'));
    try {
        callback(dir);
    } finally {
        rmSync(dir, {'recursive': true, 'force': true});
    }
}


// Build WAV file bytes from chunks - [[id, Uint8Array], ...]
function buildWav(chunks) {
    const parts = [];
    for (const [id, body] of chunks) {
        const header = Buffer.alloc(8);
        header.write(id, 0, 'ascii');
        header.writeUInt32LE(body.length, 4);
        parts.push(header, Buffer.from(body));
        if (body.length % 2) {
            parts.push(Buffer.alloc(1));
        }
    }
    const content = Buffer.concat(parts);
    const riff = Buffer.alloc(12);
    riff.write('RIFF', 0, 'ascii');
    riff.writeUInt32LE(4 + content.length, 4);
    riff.write('WAVE', 8, 'ascii');
    return Buffer.concat([riff, content]);
}


// A PCM fmt chunk body
function fmtChunk(channels, sampleRate, bits) {
    const body = Buffer.alloc(16);
    body.writeUInt16LE(1, 0);
    body.writeUInt16LE(channels, 2);
    body.writeUInt32LE(sampleRate, 4);
    body.writeUInt32LE(sampleRate * channels * bits / 8, 8);
    body.writeUInt16LE(channels * bits / 8, 12);
    body.writeUInt16LE(bits, 14);
    return body;
}


// A 16-bit data chunk body
function dataChunk(values) {
    const body = Buffer.alloc(values.length * 2);
    values.forEach((value, ix) => {
        body.writeInt16LE(value, ix * 2);
    });
    return body;
}


test('encodeWav, header', () => {
    const bytes = encodeWav(Float32Array.from([0, 0.5, -0.5]), 22050);
    assert.ok(bytes instanceof Uint8Array);
    assert.equal(bytes.length, 50);
    const buffer = Buffer.from(bytes.buffer);
    assert.equal(buffer.toString('ascii', 0, 4), 'RIFF');
    assert.equal(buffer.readUInt32LE(4), 42);
    assert.equal(buffer.toString('ascii', 8, 16), 'WAVEfmt ');
    assert.equal(buffer.readUInt32LE(16), 16);
    assert.equal(buffer.readUInt16LE(20), 1);
    assert.equal(buffer.readUInt16LE(22), 1);
    assert.equal(buffer.readUInt32LE(24), 22050);
    assert.equal(buffer.readUInt32LE(28), 44100);
    assert.equal(buffer.readUInt16LE(32), 2);
    assert.equal(buffer.readUInt16LE(34), 16);
    assert.equal(buffer.toString('ascii', 36, 40), 'data');
    assert.equal(buffer.readUInt32LE(40), 6);
});


test('encodeWav, samples and clipping', () => {
    const bytes = encodeWav(Float32Array.from([0, 1, -1, 0.5, 2, -3, 1 / 32767]), 8000);
    const buffer = Buffer.from(bytes.buffer);
    const values = Array.from({'length': 7}, (_value, ix) => buffer.readInt16LE(44 + ix * 2));
    assert.deepEqual(values, [0, 32767, -32767, 16384, 32767, -32767, 1]);

    // Empty
    const empty = encodeWav(new Float32Array(0), 8000);
    assert.equal(empty.length, 44);
    assert.equal(Buffer.from(empty.buffer).readUInt32LE(40), 0);
});


test('writeWav and readWav, round trip', () => {
    withTempDir((dir) => {
        const path = join(dir, 'test.wav');
        const samples = new Float32Array(1001).map((_value, ix) => 0.8 * Math.sin(ix / 10));
        writeWav(path, samples, 16000);
        const result = readWav(path);
        assert.equal(result.sampleRate, 16000);
        assert.equal(result.samples.length, samples.length);
        for (let ix = 0; ix < samples.length; ix++) {
            // 16-bit quantization, and the 32767 / 32768 scale difference
            assert.ok(Math.abs(result.samples[ix] - samples[ix]) < 1e-4, `sample ${ix}`);
        }

        // Writing what was read is the same file content
        const again = join(dir, 'again.wav');
        writeWav(again, result.samples, result.sampleRate);
        const reread = readWav(again);
        for (let ix = 0; ix < samples.length; ix++) {
            assert.ok(Math.abs(reread.samples[ix] - result.samples[ix]) <= 1 / 32768);
        }
    });
});


test('readWav, stereo and extra chunks', () => {
    withTempDir((dir) => {
        // A stereo file with an odd-sized (padded) LIST chunk before the data - the first channel is read
        const path = join(dir, 'stereo.wav');
        writeFileSync(path, buildWav([
            ['fmt ', fmtChunk(2, 48000, 16)],
            ['LIST', Buffer.from('abc')],
            ['data', dataChunk([16384, -1, -32768, 2, 8192, 3])]
        ]));
        const result = readWav(path);
        assert.equal(result.sampleRate, 48000);
        assert.deepEqual(Array.from(result.samples), [0.5, -1, 0.25]);
    });
});


test('readWav, data chunk without a fmt chunk', () => {
    withTempDir((dir) => {
        // The defaults - mono 16-bit, with an unknown (0) sample rate
        const path = join(dir, 'nofmt.wav');
        writeFileSync(path, buildWav([['data', dataChunk([-16384, 4096])]]));
        const result = readWav(path);
        assert.equal(result.sampleRate, 0);
        assert.deepEqual(Array.from(result.samples), [-0.5, 0.125]);
    });
});


test('readWav, unsupported bit depth', () => {
    withTempDir((dir) => {
        const path = join(dir, 'bits8.wav');
        writeFileSync(path, buildWav([['fmt ', fmtChunk(1, 8000, 8)], ['data', Buffer.from([128, 129])]]));
        assert.throws(() => readWav(path), {'message': 'Unsupported WAV bits per sample: 8'});
    });
});


test('readWav, missing data chunk', () => {
    withTempDir((dir) => {
        const path = join(dir, 'nodata.wav');
        writeFileSync(path, buildWav([['fmt ', fmtChunk(1, 8000, 16)], ['LIST', Buffer.from('info')]]));
        assert.throws(() => readWav(path), {'message': `No WAV data in ${path}`});

        // A truncated file
        const truncated = join(dir, 'truncated.wav');
        writeFileSync(truncated, buildWav([]).subarray(0, 10));
        assert.throws(() => readWav(truncated), {'message': `No WAV data in ${truncated}`});
    });
});


test('readWav and writeWav, require Node', () => {
    // Without Node's getBuiltinModule (as in a browser), file reading and writing throw
    const {getBuiltinModule} = process;
    process.getBuiltinModule = undefined;
    try {
        assert.throws(() => readWav('test.wav'), {'message': 'WAV file reading and writing requires Node'});
        assert.throws(() => writeWav('test.wav', new Float32Array(1), 8000), {'message': 'WAV file reading and writing requires Node'});
    } finally {
        process.getBuiltinModule = getBuiltinModule;
    }

    // Encoding needs no file system
    assert.equal(encodeWav(new Float32Array(1), 8000).length, 46);
});
