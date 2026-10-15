// The Standard Ebooks source - read a catalog book's EPUB as plain text for reading aloud
//
// Standard Ebooks (https://standardebooks.org) serves its EPUB downloads with Access-Control-Allow-Origin: *, so the
// EPUB (the catalog's book.text.url) is fetched directly from the browser, unzipped (lib/zip.js), and converted to text
// and chapters (lib/epub.js).

import {convertEpub, isEpubTextEntry} from '../../../lib/epub.js';
import {openZip, readZipDirectory, readZipEnd, zipEntrySpans, zipTailSize} from '../../../lib/zip.js';


// Error messages
const errorOffline = 'Couldn\'t reach Standard Ebooks. Check your connection and try again.';
const errorUnavailable = 'This book isn\'t available from Standard Ebooks.';
const errorUnreadable = 'This book from Standard Ebooks couldn\'t be read.';

// Files at most this size are downloaded whole - range requests save little and cost round trips
const rangeMinimumSize = 1500000;

// Range-request gaps smaller than this are fetched rather than starting another request
const rangeMergeGap = 65536;


// The source
export const source = {
    'id': 'standardebooks',
    'name': 'Standard Ebooks',
    load
};


// The default progress callback
function noProgress() {
    // no progress reporting
}


// A catalog book's text for reading aloud and display - {text, chapters, title, author}; onProgress(loadedBytes,
// totalBytes or null) while downloading
async function load(book, onProgress = noProgress) {
    const url = book?.text?.url;
    if (!url) {
        throw new Error(errorUnavailable);
    }

    // Download the EPUB's text files and convert them - any unexpected failure means the file isn't a readable EPUB
    try {
        return await convertEpub(await downloadEpub(url, onProgress));
    } catch (error) {
        if (error.message === errorOffline || error.message === errorUnavailable) {
            throw error;
        }
        throw new Error(errorUnreadable, {'cause': error});
    }
}


// fetch from the site, with a user-presentable error if the network fails
async function fetchSite(url, options = {}) {
    try {
        return await fetch(url, options);
    } catch {
        throw new Error(errorOffline);
    }
}


// Download an EPUB's text files (everything but images, styles, and fonts) - returns a zip reader (lib/zip.js)
//
// Illustrated editions can be large (Alice's Adventures in Wonderland is 18 MB, almost all images), so when the
// browser allows it, only the ZIP directory and the text entries are fetched with range requests. Anything
// unexpected falls back to downloading the whole file.
async function downloadEpub(epubUrl, onProgress) {
    // Get the size with a HEAD request - the site serves an HTML page (not the EPUB) for a missing file
    const head = await fetchSite(epubUrl, {'method': 'HEAD'});
    if (!head.ok || !isEpubType(head)) {
        throw new Error(errorUnavailable);
    }

    // Try range requests - any failure (e.g. a browser that preflights Range, which the site doesn't allow)
    // falls back to the whole file
    const size = Number(head.headers.get('Content-Length'));
    if (Number.isSafeInteger(size) && size > rangeMinimumSize) {
        try {
            const zip = await downloadEpubRanges(epubUrl, size, onProgress);
            if (zip !== null) {
                return zip;
            }
        } catch {
            // fall back
        }
    }

    // Download the whole file
    const response = await fetchSite(epubUrl);
    if (!response.ok) {
        throw new Error(errorUnavailable);
    }
    return openZip(await readBody(response, onProgress));
}


// Is the response an EPUB (and not the site's HTML "thank you" or error page)?
function isEpubType(response) {
    return /epub|zip|octet-stream/i.test(response.headers.get('Content-Type') ?? '');
}


// Download the ZIP directory and text entries with range requests - returns null if the server doesn't honor
// ranges (the caller downloads the whole file)
async function downloadEpubRanges(epubUrl, size, onProgress) {
    // Fetch the tail of the file, which holds the end-of-central-directory record and (usually) the directory -
    // if the server ignores the range and sends the whole file, use it
    onProgress(0, null);
    const tailStart = Math.max(0, size - zipTailSize);
    const tailResponse = await fetch(epubUrl, {'headers': {'Range': `bytes=${tailStart}-${size - 1}`}});
    if (tailResponse.status === 200) {
        return openZip(await readBody(tailResponse, onProgress));
    }
    const tail = await readRange(tailResponse, tailStart, size);
    if (tail === null) {
        return null;
    }
    const directory = readZipEnd(tail, tailStart, size);
    let directoryBytes;
    if (directory.offset >= tailStart) {
        directoryBytes = tail.subarray(directory.offset - tailStart, directory.offset - tailStart + directory.size);
    } else {
        directoryBytes = await fetchRange(epubUrl, directory.offset, directory.offset + directory.size);
        if (directoryBytes === null) {
            return null;
        }
    }
    const entries = readZipDirectory(directoryBytes, directory.count);

    // Fetch the text entries' spans, reporting progress over their total size
    const spans = zipEntrySpans(entries, directory.offset, isEpubTextEntry, rangeMergeGap);
    const total = spans.reduce((sum, span) => sum + span.end - span.start, 0);
    let loaded = 0;
    const chunks = [];
    for (const span of spans) {
        const spanLoaded = loaded;
        const bytes = await fetchRange(epubUrl, span.start, span.end, (count) => onProgress(spanLoaded + count, total));
        if (bytes === null) {
            return null;
        }
        loaded += bytes.length;
        chunks.push({'start': span.start, bytes});
    }
    onProgress(total, total);
    return {entries, chunks};
}


// Fetch the bytes [start, end) - returns null if the server didn't return the range
async function fetchRange(url, start, end, onCount = noProgress) {
    return readRange(await fetch(url, {'headers': {'Range': `bytes=${start}-${end - 1}`}}), start, end, onCount);
}


// Read a range response's bytes [start, end) - returns null if it isn't the range
async function readRange(response, start, end, onCount = noProgress) {
    if (response.status !== 206 || Number(response.headers.get('Content-Length')) !== end - start) {
        await response.body?.cancel();
        return null;
    }
    const bytes = await readBody(response, (count) => onCount(count));
    return bytes.length === end - start ? bytes : null;
}


// Read a response body, reporting progress - onProgress(loadedBytes, totalBytes or null); returns a Uint8Array
async function readBody(response, onProgress) {
    const total = Number(response.headers.get('Content-Length')) || null;
    const reader = response.body.getReader();
    const parts = [];
    let loaded = 0;
    onProgress(0, total);
    for (;;) {
        let result;
        try {
            result = await reader.read();
        } catch {
            throw new Error(errorOffline);
        }
        if (result.done) {
            break;
        }
        parts.push(result.value);
        loaded += result.value.length;
        onProgress(loaded, total);
    }

    // Concatenate the parts
    const bytes = new Uint8Array(loaded);
    let offset = 0;
    for (const part of parts) {
        bytes.set(part, offset);
        offset += part.length;
    }
    return bytes;
}
