// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

import {openZip, readZipDirectory, readZipEnd, readZipEntry, readZipText, zipEntrySpans, zipTailSize} from '../lib/zip.js';
import {strict as assert} from 'node:assert';
import {deflateRawSync} from 'node:zlib';
import test from 'node:test';


// Build a ZIP file's bytes from [{name, data (string or Uint8Array), method (0 stored, 8 deflated; default 8),
// extra (local and central extra field length)}], with an optional archive comment
function buildZip(files, comment = '') {
    const encoder = new TextEncoder();
    const locals = [];
    const centrals = [];
    let offset = 0;
    for (const file of files) {
        const nameBytes = encoder.encode(file.name);
        const raw = typeof file.data === 'string' ? encoder.encode(file.data) : file.data;
        const method = file.method ?? 8;
        const data = method === 8 ? new Uint8Array(deflateRawSync(raw)) : raw;
        const extra = file.extra ?? 0;

        // The local header, name, extra field, and data
        const local = new Uint8Array(30 + nameBytes.length + extra + data.length);
        const localView = new DataView(local.buffer);
        localView.setUint32(0, 0x04034b50, true);
        localView.setUint16(8, method, true);
        localView.setUint32(18, data.length, true);
        localView.setUint32(22, raw.length, true);
        localView.setUint16(26, nameBytes.length, true);
        localView.setUint16(28, extra, true);
        local.set(nameBytes, 30);
        local.set(data, 30 + nameBytes.length + extra);
        locals.push(local);

        // The central directory header, with an extra field and comment
        const central = new Uint8Array(46 + nameBytes.length + extra + 2);
        const centralView = new DataView(central.buffer);
        centralView.setUint32(0, 0x02014b50, true);
        centralView.setUint16(10, method, true);
        centralView.setUint32(20, data.length, true);
        centralView.setUint32(24, raw.length, true);
        centralView.setUint16(28, nameBytes.length, true);
        centralView.setUint16(30, extra, true);
        centralView.setUint16(32, 2, true);
        centralView.setUint32(42, offset, true);
        central.set(nameBytes, 46);
        centrals.push(central);
        offset += local.length;
    }

    // The end-of-central-directory record
    const commentBytes = encoder.encode(comment);
    const directorySize = centrals.reduce((sum, central) => sum + central.length, 0);
    const end = new Uint8Array(22 + commentBytes.length);
    const endView = new DataView(end.buffer);
    endView.setUint32(0, 0x06054b50, true);
    endView.setUint16(8, files.length, true);
    endView.setUint16(10, files.length, true);
    endView.setUint32(12, directorySize, true);
    endView.setUint32(16, offset, true);
    end.set(commentBytes, 22);
    return concatBytes([...locals, ...centrals, end]);
}


// Concatenate Uint8Arrays
function concatBytes(parts) {
    const bytes = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
    let offset = 0;
    for (const part of parts) {
        bytes.set(part, offset);
        offset += part.length;
    }
    return bytes;
}


// A test archive - deflated and stored entries, and an entry with extra fields
const testFiles = [
    {'name': 'mimetype', 'data': 'application/epub+zip', 'method': 0},
    {'name': 'a.xhtml', 'data': 'Hello, ZIP! '.repeat(50)},
    {'name': 'image.png', 'data': new Uint8Array(100).fill(7), 'method': 0, 'extra': 4},
    {'name': 'b.xhtml', 'data': 'Café ☕'}
];


test('zipTailSize', () => {
    assert.equal(zipTailSize, 65536);
});


test('openZip', async () => {
    const bytes = buildZip(testFiles, 'a comment');
    const zip = openZip(bytes);
    assert.deepEqual([...zip.entries.keys()], ['mimetype', 'a.xhtml', 'image.png', 'b.xhtml']);
    assert.deepEqual(zip.entries.get('mimetype'), {'name': 'mimetype', 'method': 0, 'compressedSize': 20, 'size': 20, 'offset': 0});
    assert.equal(zip.entries.get('a.xhtml').method, 8);
    assert.equal(zip.entries.get('a.xhtml').size, 600);
    assert.ok(zip.entries.get('a.xhtml').compressedSize < 600);
    assert.deepEqual(zip.chunks, [{'start': 0, bytes}]);

    // Read the entries
    assert.equal(await readZipText(zip, 'mimetype'), 'application/epub+zip');
    assert.equal(await readZipText(zip, 'a.xhtml'), 'Hello, ZIP! '.repeat(50));
    assert.equal(await readZipText(zip, 'b.xhtml'), 'Café ☕');
    assert.deepEqual(await readZipEntry(zip, 'image.png'), new Uint8Array(100).fill(7));
});


test('openZip, large', async () => {
    // A file larger than the tail - the directory is found in the last zipTailSize bytes
    const big = new Uint8Array(zipTailSize + 1000);
    for (let ix = 0; ix < big.length; ix++) {
        big[ix] = (ix * 7919) % 251;
    }
    const zip = openZip(buildZip([{'name': 'big.bin', 'data': big, 'method': 0}, {'name': 'small.txt', 'data': 'small'}]));
    assert.deepEqual(await readZipEntry(zip, 'big.bin'), big);
    assert.equal(await readZipText(zip, 'small.txt'), 'small');
});


test('openZip, empty', () => {
    const zip = openZip(buildZip([]));
    assert.equal(zip.entries.size, 0);
});


test('readZipEnd', () => {
    const bytes = buildZip(testFiles);

    // The whole file as the tail
    const end = readZipEnd(bytes, 0, bytes.length);
    assert.equal(end.count, 4);
    assert.equal(end.size, 4 * 48 + 'mimetype'.length + 'a.xhtml'.length + 'image.png'.length + 4 + 'b.xhtml'.length);
    assert.equal(end.offset + end.size, bytes.length - 22);

    // A partial tail
    const tailStart = bytes.length - 40;
    assert.deepEqual(readZipEnd(bytes.subarray(tailStart), tailStart, bytes.length), end);
});


test('readZipEnd, not found', () => {
    assert.throws(() => readZipEnd(new Uint8Array(100), 0, 100), {'message': 'ZIP end of central directory not found'});
    assert.throws(() => readZipEnd(new Uint8Array(10), 0, 10), {'message': 'ZIP end of central directory not found'});
});


test('readZipEnd, invalid records', () => {
    // A record whose directory runs past the record - skipped
    const bytes = buildZip(testFiles);
    const badRecord = new Uint8Array(22);
    const badView = new DataView(badRecord.buffer);
    badView.setUint32(0, 0x06054b50, true);
    badView.setUint32(12, 1000000, true);
    const withBad = concatBytes([bytes, badRecord]);
    assert.deepEqual(readZipEnd(withBad, 0, withBad.length), readZipEnd(bytes, 0, bytes.length));

    // A record whose directory runs past the file's size
    assert.throws(() => readZipEnd(bytes, 1000000, 10), {'message': 'ZIP end of central directory not found'});
});


test('readZipDirectory, invalid', () => {
    assert.throws(() => readZipDirectory(new Uint8Array(46), 1), {'message': 'Invalid ZIP central directory'});
    assert.deepEqual(readZipDirectory(new Uint8Array(0), 0), new Map());
});


test('zipEntrySpans', () => {
    const entries = new Map([
        ['c.xhtml', {'name': 'c.xhtml', 'offset': 300}],
        ['a.xhtml', {'name': 'a.xhtml', 'offset': 0}],
        ['b.xhtml', {'name': 'b.xhtml', 'offset': 100}],
        ['image.png', {'name': 'image.png', 'offset': 200}],
        ['d.xhtml', {'name': 'd.xhtml', 'offset': 5000}],
        ['e.png', {'name': 'e.png', 'offset': 6000}],
        ['f.xhtml', {'name': 'f.xhtml', 'offset': 7000}]
    ]);
    const isNeeded = (name) => name.endsWith('.xhtml');

    // No merging
    assert.deepEqual(zipEntrySpans(entries, 8000, isNeeded, 0), [
        {'start': 0, 'end': 100},
        {'start': 100, 'end': 200},
        {'start': 300, 'end': 5000},
        {'start': 5000, 'end': 6000},
        {'start': 7000, 'end': 8000}
    ]);

    // Merge gaps smaller than 1000 bytes
    assert.deepEqual(zipEntrySpans(entries, 8000, isNeeded, 1000), [
        {'start': 0, 'end': 6000},
        {'start': 7000, 'end': 8000}
    ]);

    // Nothing needed
    assert.deepEqual(zipEntrySpans(entries, 8000, () => false, 1000), []);
});


test('readZipEntry, chunks', async () => {
    // Entries from partial chunks, as range requests download them
    const bytes = buildZip(testFiles);
    const {entries} = openZip(bytes);
    const aEntry = entries.get('a.xhtml');
    const bEntry = entries.get('b.xhtml');
    const zip = {entries, 'chunks': [
        {'start': aEntry.offset, 'bytes': bytes.slice(aEntry.offset, entries.get('image.png').offset)},
        {'start': bEntry.offset, 'bytes': bytes.slice(bEntry.offset)}
    ]};
    assert.equal(await readZipText(zip, 'a.xhtml'), 'Hello, ZIP! '.repeat(50));
    assert.equal(await readZipText(zip, 'b.xhtml'), 'Café ☕');

    // Not downloaded
    await assert.rejects(readZipText(zip, 'mimetype'), {'message': 'ZIP entry not downloaded: mimetype'});

    // Not downloaded - only part of the local header
    const partial = {entries, 'chunks': [{'start': 0, 'bytes': bytes.slice(0, 20)}]};
    await assert.rejects(readZipText(partial, 'mimetype'), {'message': 'ZIP entry not downloaded: mimetype'});

    // Truncated
    const truncated = {entries, 'chunks': [{'start': aEntry.offset, 'bytes': bytes.slice(aEntry.offset, aEntry.offset + 50)}]};
    await assert.rejects(readZipText(truncated, 'a.xhtml'), {'message': 'ZIP entry truncated: a.xhtml'});
});


test('readZipEntry, errors', async () => {
    const bytes = buildZip(testFiles);
    const zip = openZip(bytes);

    // Missing
    await assert.rejects(readZipEntry(zip, 'missing.xhtml'), {'message': 'ZIP entry not found: missing.xhtml'});

    // Invalid local header
    const badZip = {'entries': zip.entries, 'chunks': [{'start': 0, 'bytes': new Uint8Array(bytes.length)}]};
    await assert.rejects(readZipEntry(badZip, 'a.xhtml'), {'message': 'Invalid ZIP local header: a.xhtml'});

    // Unsupported compression method
    const methodZip = {'entries': new Map([['a.xhtml', {...zip.entries.get('a.xhtml'), 'method': 12}]]), 'chunks': zip.chunks};
    await assert.rejects(readZipEntry(methodZip, 'a.xhtml'), {'message': 'Unsupported ZIP compression method 12: a.xhtml'});

    // Invalid deflate data
    const corrupt = bytes.slice();
    const aEntry = zip.entries.get('a.xhtml');
    corrupt.fill(0xff, aEntry.offset + 30 + 7, aEntry.offset + 30 + 7 + aEntry.compressedSize);
    await assert.rejects(readZipEntry({'entries': zip.entries, 'chunks': [{'start': 0, 'bytes': corrupt}]}, 'a.xhtml'));
});
