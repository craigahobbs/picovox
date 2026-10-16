import {encodeWav, writeWav} from '../lib/wav.js';
import {mkdtempSync, readFileSync, rmSync} from 'node:fs';
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


test('writeWav', () => {
    withTempDir((dir) => {
        // The file is the encoded bytes
        const path = join(dir, 'test.wav');
        const samples = new Float32Array(1001).map((_value, ix) => 0.8 * Math.sin(ix / 10));
        writeWav(path, samples, 16000);
        assert.deepEqual(new Uint8Array(readFileSync(path)), encodeWav(samples, 16000));
    });
});

test('writeWav, requires Node', () => {
    // Without Node's getBuiltinModule (as in a browser), file writing throws
    const {getBuiltinModule} = process;
    process.getBuiltinModule = undefined;
    try {
        assert.throws(() => writeWav('test.wav', new Float32Array(1), 8000), {'message': 'WAV file writing requires Node'});
    } finally {
        process.getBuiltinModule = getBuiltinModule;
    }

    // Encoding needs no file system
    assert.equal(encodeWav(new Float32Array(1), 8000).length, 46);
});
