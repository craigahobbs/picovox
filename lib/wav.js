// WAV encoding - 16-bit PCM, mono


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
