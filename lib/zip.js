// ZIP reading - the central directory and the entries' data (stored or deflated), from a whole file's bytes or from
// byte ranges of it (so a browser can download only the entries it needs)
//
// A zip reader is {entries, chunks}: entries is a Map of name to {name, method, compressedSize, size, offset}, and
// chunks is an array of {start, bytes} - the downloaded parts of the file, by byte offset.


// The size of the ZIP tail read to find the end-of-central-directory record (and, usually, the directory)
export const zipTailSize = 65536;


// Find the end-of-central-directory record in the ZIP's tail (the bytes from tailStart to the end of a file of size
// bytes) - returns {offset, size, count}, the central directory's offset, size, and entry count
export function readZipEnd(tail, tailStart, size) {
    const view = new DataView(tail.buffer, tail.byteOffset, tail.byteLength);
    for (let ix = tail.length - 22; ix >= 0; ix--) {
        if (view.getUint32(ix, true) === 0x06054b50) {
            const count = view.getUint16(ix + 10, true);
            const directorySize = view.getUint32(ix + 12, true);
            const offset = view.getUint32(ix + 16, true);
            if (offset + directorySize <= tailStart + ix && offset + directorySize <= size) {
                return {offset, 'size': directorySize, count};
            }
        }
    }
    throw new Error('ZIP end of central directory not found');
}


// Read the ZIP central directory - returns a Map of name to {name, method, compressedSize, size, offset}
export function readZipDirectory(bytes, count) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const decoder = new TextDecoder();
    const entries = new Map();
    let pos = 0;
    for (let ix = 0; ix < count; ix++) {
        if (view.getUint32(pos, true) !== 0x02014b50) {
            throw new Error('Invalid ZIP central directory');
        }
        const nameLength = view.getUint16(pos + 28, true);
        const name = decoder.decode(bytes.subarray(pos + 46, pos + 46 + nameLength));
        entries.set(name, {
            name,
            'method': view.getUint16(pos + 10, true),
            'compressedSize': view.getUint32(pos + 20, true),
            'size': view.getUint32(pos + 24, true),
            'offset': view.getUint32(pos + 42, true)
        });
        pos += 46 + nameLength + view.getUint16(pos + 30, true) + view.getUint16(pos + 32, true);
    }
    return entries;
}


// Open a whole ZIP file's bytes - returns a zip reader
export function openZip(bytes) {
    const tailStart = Math.max(0, bytes.length - zipTailSize);
    const directory = readZipEnd(bytes.subarray(tailStart), tailStart, bytes.length);
    const entries = readZipDirectory(bytes.subarray(directory.offset, directory.offset + directory.size), directory.count);
    return {entries, 'chunks': [{'start': 0, bytes}]};
}


// The byte spans [start, end) holding the entries for which isNeeded(name) is true - each entry's span runs from its
// local header to the next entry (or the central directory at directoryOffset), and spans closer than mergeGap bytes
// are merged
export function zipEntrySpans(entries, directoryOffset, isNeeded, mergeGap) {
    const byOffset = [...entries.values()].sort((entryA, entryB) => entryA.offset - entryB.offset);
    const spans = [];
    for (const [ix, entry] of byOffset.entries()) {
        if (isNeeded(entry.name)) {
            const end = ix + 1 < byOffset.length ? byOffset[ix + 1].offset : directoryOffset;
            const last = spans.at(-1);
            if (last && entry.offset - last.end < mergeGap) {
                last.end = end;
            } else {
                spans.push({'start': entry.offset, end});
            }
        }
    }
    return spans;
}


// Read a ZIP entry's bytes (Uint8Array) - throws if it's missing, not downloaded, or not stored or deflated
export async function readZipEntry(zip, name) {
    const entry = zip.entries.get(name);
    if (!entry) {
        throw new Error(`ZIP entry not found: ${name}`);
    }

    // Find the chunk holding the entry and skip its local header
    const chunk = zip.chunks.find((chunkFind) => entry.offset >= chunkFind.start &&
        entry.offset + 30 <= chunkFind.start + chunkFind.bytes.length);
    if (!chunk) {
        throw new Error(`ZIP entry not downloaded: ${name}`);
    }
    const local = entry.offset - chunk.start;
    const view = new DataView(chunk.bytes.buffer, chunk.bytes.byteOffset, chunk.bytes.byteLength);
    if (view.getUint32(local, true) !== 0x04034b50) {
        throw new Error(`Invalid ZIP local header: ${name}`);
    }
    const dataStart = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true);
    const data = chunk.bytes.subarray(dataStart, dataStart + entry.compressedSize);
    if (data.length !== entry.compressedSize) {
        throw new Error(`ZIP entry truncated: ${name}`);
    }

    // Stored or deflated
    if (entry.method === 0) {
        return data;
    }
    if (entry.method === 8) {
        const stream = new Blob([data]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
        return new Uint8Array(await new Response(stream).arrayBuffer());
    }
    throw new Error(`Unsupported ZIP compression method ${entry.method}: ${name}`);
}


// Read a ZIP entry as UTF-8 text - throws as readZipEntry does
export async function readZipText(zip, name) {
    return new TextDecoder().decode(await readZipEntry(zip, name));
}
