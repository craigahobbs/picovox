// The Project Gutenberg source - read a catalog book's plain text for reading aloud
//
// gutenberg.org doesn't send CORS headers, so the text (the catalog's book.text.url) is fetched from the GITenberg
// mirror on GitHub (raw.githubusercontent.com sends Access-Control-Allow-Origin: *), decoded with the catalog's
// book.text.encoding, and prepared for reading aloud (lib/gutenberg.js): no license boilerplate, no table of contents
// or markup, paragraphs unwrapped, and chapter headings found and listed.

import {decodeText, prepareText} from '../../../lib/gutenberg.js';


// Error messages
const errorOffline = 'Couldn\'t download the book - check your connection and try again.';
const errorUnavailable = 'This book isn\'t available from GITenberg, the Project Gutenberg mirror this app reads.';


// The source
export const source = {
    'id': 'gutenberg',
    'name': 'Project Gutenberg',
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
    let response;
    try {
        response = await fetch(url);
    } catch {
        throw new Error(errorOffline);
    }
    if (response.status === 404) {
        throw new Error(errorUnavailable);
    }
    if (!response.ok) {
        throw new Error(`The book couldn't be downloaded (error ${response.status}) - try again later.`);
    }
    const bytes = await readBody(response, onProgress);
    return prepareText(decodeText(bytes, book.text.encoding ?? 'utf-8'), book);
}


// Read a response body, reporting progress - the total is unknown (null) for a cross-origin response, since GitHub
// compresses text and its Content-Length is then the compressed size
async function readBody(response, onProgress) {
    const encoding = response.headers.get('Content-Encoding');
    const isLengthKnown = response.type !== 'cors' || encoding === 'identity';
    const contentLength = isLengthKnown ? (Number(response.headers.get('Content-Length')) || null) : null;
    const chunks = [];
    let loaded = 0;
    onProgress(0, contentLength);
    if (!response.body) {
        const buffer = new Uint8Array(await response.arrayBuffer());
        onProgress(buffer.length, buffer.length);
        return buffer;
    }
    const reader = response.body.getReader();
    let total = contentLength;
    for (;;) {
        let result;
        try {
            result = await reader.read();
        } catch {
            throw new Error(errorOffline);
        }
        const {done, value} = result;
        if (done) {
            break;
        }
        chunks.push(value);
        loaded += value.length;
        if (total !== null && loaded > total) {
            total = null;
        }
        onProgress(loaded, total);
    }
    onProgress(loaded, loaded);
    const bytes = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.length;
    }
    return bytes;
}
