// Render WAV files as stacked spectrogram PNG images (with optional pitch tracks) for visual comparison
//
// node tools/spectrogram.js out.png a.wav [b.wav ...]

import {pitchTrack, spectrogram} from '../lib/analysis.js';
import {Buffer} from 'node:buffer';
import {deflateSync} from 'node:zlib';
import process from 'node:process';
import {readWav} from '../lib/wav.js';
import {writeFileSync} from 'node:fs';


// Encode an RGB image as a PNG
function encodePng(width, height, rgb) {
    const crcTable = new Int32Array(256);
    for (let ix = 0; ix < 256; ix++) {
        let value = ix;
        for (let kx = 0; kx < 8; kx++) {
            value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
        }
        crcTable[ix] = value;
    }
    const crc = (buffer) => {
        let value = -1;
        for (const byte of buffer) {
            value = crcTable[(value ^ byte) & 0xff] ^ (value >>> 8);
        }
        return (value ^ -1) >>> 0;
    };
    const chunk = (type, data) => {
        const length = Buffer.alloc(4);
        length.writeUInt32BE(data.length);
        const typeData = Buffer.concat([Buffer.from(type), data]);
        const crcBuffer = Buffer.alloc(4);
        crcBuffer.writeUInt32BE(crc(typeData));
        return Buffer.concat([length, typeData, crcBuffer]);
    };
    const raw = Buffer.alloc((width * 3 + 1) * height);
    for (let row = 0; row < height; row++) {
        raw[row * (width * 3 + 1)] = 0;
        rgb.copy(raw, row * (width * 3 + 1) + 1, row * width * 3, (row + 1) * width * 3);
    }
    const header = Buffer.alloc(13);
    header.writeUInt32BE(width, 0);
    header.writeUInt32BE(height, 4);
    header[8] = 8;
    header[9] = 2;
    return Buffer.concat([
        Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
        chunk('IHDR', header),
        chunk('IDAT', deflateSync(raw)),
        chunk('IEND', Buffer.alloc(0))
    ]);
}


// A dark-to-bright color map
function colorMap(value) {
    const level = Math.max(0, Math.min(1, value));
    return [
        Math.round(255 * Math.min(1, level * 1.6)),
        Math.round(255 * Math.max(0, Math.min(1, level * 1.6 - 0.4))),
        Math.round(255 * Math.max(0, level * 2 - 1))
    ];
}


// Render spectrograms of WAV files, stacked, to a PNG - each 0 to maxHz, with the pitch track in cyan
export function renderSpectrograms(pngPath, wavPaths, {maxHz = 8000, pixelsPerSecond = 200, rowHeight = 220, rangeDb = 70} = {}) {
    const rows = wavPaths.map((path) => {
        const {samples, sampleRate} = readWav(path);
        const frames = spectrogram(samples, sampleRate, {'fftSize': 1024});
        return {samples, sampleRate, frames, 'pitch': pitchTrack(samples, sampleRate)};
    });
    const maxSeconds = Math.max(...rows.map((row) => row.samples.length / row.sampleRate));
    const width = Math.max(200, Math.round(maxSeconds * pixelsPerSecond));
    const height = rows.length * (rowHeight + 4);
    const rgb = Buffer.alloc(width * height * 3, 20);
    rows.forEach((row, ixRow) => {
        const top = ixRow * (rowHeight + 4);
        const peak = Math.max(...row.frames.map((frame) => Math.max(...frame.db)));
        const binHz = row.sampleRate / 1024;
        for (let column = 0; column < width; column++) {
            const time = column / pixelsPerSecond;
            const frame = row.frames[Math.min(row.frames.length - 1, Math.round(time / 0.005))];
            if (!frame || time > row.samples.length / row.sampleRate) {
                continue;
            }
            for (let line = 0; line < rowHeight; line++) {
                const hz = (rowHeight - 1 - line) / rowHeight * maxHz;
                const bin = Math.min(frame.db.length - 1, Math.round(hz / binHz));
                const value = (frame.db[bin] - (peak - rangeDb)) / rangeDb;
                const [red, green, blue] = colorMap(value);
                const offset = ((top + line) * width + column) * 3;
                rgb[offset] = red;
                rgb[offset + 1] = green;
                rgb[offset + 2] = blue;
            }
        }
        // The pitch track, plotted against 0-500 Hz over the row's height
        for (const point of row.pitch) {
            if (point.f0 > 0) {
                const column = Math.round(point.time * pixelsPerSecond);
                const line = top + Math.round(rowHeight - 1 - point.f0 / 500 * (rowHeight - 1));
                for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
                    const offset = ((line + dy) * width + column + dx) * 3;
                    if (column + dx < width && offset + 2 < rgb.length) {
                        rgb[offset] = 0;
                        rgb[offset + 1] = 255;
                        rgb[offset + 2] = 255;
                    }
                }
            }
        }
    });
    writeFileSync(pngPath, encodePng(width, height, rgb));
}


if (process.argv[1] && process.argv[1].endsWith('spectrogram.js')) {
    const [pngPath, ...wavPaths] = process.argv.slice(2);
    renderSpectrograms(pngPath, wavPaths);
}
