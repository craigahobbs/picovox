// WAV file reading and writing - 16-bit PCM, mono

// Node's file system module, if present (no static node:fs import, so the module also loads in the browser)
function nodeFs() {
    const fs = globalThis.process?.getBuiltinModule?.('node:fs');
    if (!fs) {
        throw new Error('WAV file reading and writing requires Node');
    }
    return fs;
}


// Encode a mono Float32Array (-1 to 1) as 16-bit PCM WAV file bytes (Uint8Array)
export function encodeWav(samples, sampleRate) {
    const dataBytes = samples.length * 2;
    const bytes = new Uint8Array(44 + dataBytes);
    const view = new DataView(bytes.buffer);
    const writeAscii = (text, offset) => {
        for (let ix = 0; ix < text.length; ix++) {
            bytes[offset + ix] = text.charCodeAt(ix);
        }
    };
    writeAscii('RIFF', 0);
    view.setUint32(4, 36 + dataBytes, true);
    writeAscii('WAVE', 8);
    writeAscii('fmt ', 12);
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeAscii('data', 36);
    view.setUint32(40, dataBytes, true);
    for (let ix = 0; ix < samples.length; ix++) {
        const value = Math.max(-1, Math.min(1, samples[ix]));
        view.setInt16(44 + ix * 2, Math.round(value * 32767), true);
    }
    return bytes;
}


// Write a mono Float32Array (-1 to 1) as a 16-bit PCM WAV file (Node)
export function writeWav(path, samples, sampleRate) {
    nodeFs().writeFileSync(path, encodeWav(samples, sampleRate));
}


// Read a 16-bit PCM WAV file (mono, or the first channel) - returns {samples, sampleRate} (Node)
export function readWav(path) {
    const buffer = nodeFs().readFileSync(path);
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
