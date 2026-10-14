// WAV file reading and writing - 16-bit PCM, mono

import {readFileSync, writeFileSync} from 'node:fs';


// Write a mono Float32Array (-1 to 1) as a 16-bit PCM WAV file
export function writeWav(path, samples, sampleRate) {
    const dataBytes = samples.length * 2;
    const buffer = Buffer.alloc(44 + dataBytes);
    buffer.write('RIFF', 0);
    buffer.writeUInt32LE(36 + dataBytes, 4);
    buffer.write('WAVE', 8);
    buffer.write('fmt ', 12);
    buffer.writeUInt32LE(16, 16);
    buffer.writeUInt16LE(1, 20);
    buffer.writeUInt16LE(1, 22);
    buffer.writeUInt32LE(sampleRate, 24);
    buffer.writeUInt32LE(sampleRate * 2, 28);
    buffer.writeUInt16LE(2, 32);
    buffer.writeUInt16LE(16, 34);
    buffer.write('data', 36);
    buffer.writeUInt32LE(dataBytes, 40);
    for (let ix = 0; ix < samples.length; ix++) {
        const value = Math.max(-1, Math.min(1, samples[ix]));
        buffer.writeInt16LE(Math.round(value * 32767), 44 + ix * 2);
    }
    writeFileSync(path, buffer);
}


// Read a 16-bit PCM WAV file (mono, or the first channel) - returns {samples, sampleRate}
export function readWav(path) {
    const buffer = readFileSync(path);
    let offset = 12;
    let sampleRate = 0;
    let channels = 1;
    let bits = 16;
    while (offset + 8 <= buffer.length) {
        const id = buffer.toString('ascii', offset, offset + 4);
        const size = buffer.readUInt32LE(offset + 4);
        if (id === 'fmt ') {
            channels = buffer.readUInt16LE(offset + 10);
            sampleRate = buffer.readUInt32LE(offset + 12);
            bits = buffer.readUInt16LE(offset + 22);
        } else if (id === 'data') {
            if (bits !== 16) {
                throw new Error(`Unsupported WAV bits per sample: ${bits}`);
            }
            const count = Math.floor(size / (2 * channels));
            const samples = new Float32Array(count);
            for (let ix = 0; ix < count; ix++) {
                samples[ix] = buffer.readInt16LE(offset + 8 + ix * 2 * channels) / 32768;
            }
            return {samples, sampleRate};
        }
        offset += 8 + size + (size % 2);
    }
    throw new Error(`No WAV data in ${path}`);
}
