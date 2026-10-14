// Check the Standard Ebooks source (web/sources/standardebooks.js) in headless Chrome over CDP - load catalog books
// (each with its EPUB's text URL), printing per book: title, author, text length, chapters, the first 4 and last 2
// chapter titles, the first 300 characters, load time, and download size
//
// node tools/standardebookscheck.mjs [options] [book id or EPUB URL ...]
//
//   --catalog <file>  check the Standard Ebooks books of a catalog (e.g. data/catalog.json) instead
//   --limit <count>   with --catalog, only the first count books
//   --live            fetch from standardebooks.org (default: serve standardebooks.org from the cache in
//                     out/standardebooks/cache, downloading each URL once - be considerate to the site)
//   --no-range        the cache ignores Range requests (returns the whole file) - checks the whole-file fallback
//   --save            save each book's text to out/standardebooks/text/<id>.txt
//
// Book ids are site paths, e.g. /ebooks/herman-melville/moby-dick (the "/ebooks/" prefix is optional) - the EPUB's URL
// is the site's download URL for the id. An EPUB URL's book id is its path before "/downloads/".

import {dirname, join} from 'node:path';
import {existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {Buffer} from 'node:buffer';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import process from 'node:process';
import {spawn} from 'node:child_process';


// Print a line
function print(text = '') {
    process.stdout.write(`${text}\n`);
}


// Options and books
const args = process.argv.slice(2);
const options = new Set(args.filter((arg) => arg.startsWith('--')));
const optionValue = (name) => {
    const index = args.indexOf(name);
    return index !== -1 && index + 1 < args.length ? args[index + 1] : null;
};
const catalogPath = optionValue('--catalog');
const limit = Number(optionValue('--limit') ?? Infinity);
const defaultBooks = [
    'herman-melville/moby-dick',
    'jane-austen/pride-and-prejudice',
    'lewis-carroll/alices-adventures-in-wonderland/john-tenniel',
    'john-keats/poetry',
    'william-shakespeare/hamlet',
    'ernest-hemingway/short-fiction',
    'marcus-aurelius/meditations/george-long',
    'homer/the-odyssey/william-cullen-bryant',
    'fyodor-dostoevsky/crime-and-punishment/constance-garnett',
    'edmund-burke/reflections-on-the-revolution-in-france'
];
const bookArgs = args.filter((arg, ix) => !arg.startsWith('--') && !['--catalog', '--limit'].includes(args[ix - 1]));


// A catalog book from a book id or EPUB URL
function bookFromArg(arg) {
    if (/^https?:/.test(arg)) {
        const id = new URL(arg).pathname.replace(/\/downloads\/.*$/, '');
        return {'source': 'standardebooks', id, 'title': '', 'author': '', 'text': {'url': arg, 'format': 'epub'}};
    }
    const id = arg.startsWith('/ebooks/') ? arg : `/ebooks/${arg.replace(/^\//, '')}`;
    const fileName = `${id.slice('/ebooks/'.length).split('/').join('_')}.epub`;
    const url = `https://standardebooks.org${id}/downloads/${fileName}?source=download`;
    return {'source': 'standardebooks', id, 'title': '', 'author': '', 'text': {url, 'format': 'epub'}};
}
let books;
if (catalogPath !== null) {
    books = JSON.parse(readFileSync(catalogPath, 'utf8')).books.filter((book) => book.source === 'standardebooks').slice(0, limit);
} else {
    books = (bookArgs.length !== 0 ? bookArgs : defaultBooks).map(bookFromArg);
}

// Paths
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const outDir = join(root, 'out', 'standardebooks');
const cacheDir = join(outDir, 'cache');
const textDir = join(outDir, 'text');
mkdirSync(cacheDir, {'recursive': true});
mkdirSync(textDir, {'recursive': true});
writeFileSync(join(outDir, 'check.html'), '<!doctype html><meta charset="utf-8"><title>Standard Ebooks check</title>\n');

// Serve the repository
const port = 8700 + Math.floor(Math.random() * 200);
const server = spawn('python3', ['-m', 'http.server', String(port), '--bind', '127.0.0.1'], {'cwd': root, 'stdio': 'ignore'});

// Launch headless Chrome
const debugPort = 9400 + Math.floor(Math.random() * 200);
const profileDir = mkdtempSync(join(outDir, 'chrome-'));
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
    '--headless=new', '--disable-gpu', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profileDir}`, 'about:blank'
], {'stdio': 'ignore'});

// Stop the server and Chrome on exit
const cleanup = () => {
    chrome.kill();
    server.kill();
    try {
        rmSync(profileDir, {'recursive': true, 'force': true});
    } catch {
        // ignore
    }
};
process.on('exit', cleanup);
process.on('SIGINT', () => process.exit(1));

// Connect to the page
const sleep = (ms) => new Promise((resolve) => {
    setTimeout(resolve, ms);
});
let targets = [];
for (let ix = 0; ix < 50 && targets.length === 0; ix++) {
    await sleep(200);
    try {
        targets = (await (await fetch(`http://127.0.0.1:${debugPort}/json`)).json()).filter((target) => target.type === 'page');
    } catch {
        // not yet
    }
}
const ws = new WebSocket(targets[0].webSocketDebuggerUrl);
await new Promise((resolve) => {
    ws.addEventListener('open', resolve);
});
let messageId = 0;
const pending = new Map();
const logs = [];
const send = (method, params = {}) => new Promise((resolve) => {
    messageId += 1;
    pending.set(messageId, resolve);
    ws.send(JSON.stringify({'id': messageId, method, params}));
});

// Network accounting - bytes received from standardebooks.org and request count
let networkBytes = 0;
let networkRequests = 0;
const siteRequests = new Set();

// Handle CDP messages
ws.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
        pending.get(message.id)(message.result ?? message.error);
        pending.delete(message.id);
    } else if (message.method === 'Runtime.consoleAPICalled') {
        logs.push(`console.${message.params.type}: ${message.params.args.map((arg) => arg.value ?? arg.description).join(' ')}`);
    } else if (message.method === 'Runtime.exceptionThrown') {
        logs.push(`exception: ${message.params.exceptionDetails.exception?.description ?? message.params.exceptionDetails.text}`);
    } else if (message.method === 'Network.requestWillBeSent' &&
        message.params.request.url.startsWith('https://standardebooks.org/')) {
        siteRequests.add(message.params.requestId);
        networkRequests += 1;
    } else if (message.method === 'Network.dataReceived' && siteRequests.has(message.params.requestId)) {
        networkBytes += message.params.encodedDataLength || message.params.dataLength;
    } else if (message.method === 'Fetch.requestPaused') {
        fulfillFromCache(message.params).catch((error) => {
            logs.push(`cache error: ${error.message}`);
            send('Fetch.failRequest', {'requestId': message.params.requestId, 'errorReason': 'Failed'});
        });
    }
});


// Serve a standardebooks.org request from the cache, downloading the URL once (whole, without Range)
async function fulfillFromCache({requestId, request}) {
    const key = createHash('sha1').update(request.url).digest('hex');
    const bodyPath = join(cacheDir, `${key}.body`);
    const metaPath = join(cacheDir, `${key}.json`);
    if (!existsSync(metaPath)) {
        const response = await fetch(request.url, {'redirect': 'manual', 'headers': {'User-Agent': 'voice-synth source check'}});
        const body = Buffer.from(await response.arrayBuffer());
        writeFileSync(bodyPath, body);
        writeFileSync(metaPath, JSON.stringify({
            'url': request.url,
            'status': response.status,
            'contentType': response.headers.get('content-type'),
            'location': response.headers.get('location')
        }));
        cacheDownloads += 1;
        cacheDownloadBytes += body.length;
    }
    const meta = JSON.parse(readFileSync(metaPath, 'utf8'));
    let body = readFileSync(bodyPath);
    let {status} = meta;
    const headers = [
        {'name': 'Access-Control-Allow-Origin', 'value': '*'},
        {'name': 'Cache-Control', 'value': 'no-store'}
    ];
    if (meta.contentType) {
        headers.push({'name': 'Content-Type', 'value': meta.contentType});
    }
    if (meta.location) {
        headers.push({'name': 'Location', 'value': new URL(meta.location, request.url).href});
    }

    // Ranges, as the site serves them
    const rangeHeader = Object.entries(request.headers).find(([name]) => name.toLowerCase() === 'range')?.[1];
    const rangeMatch = /^bytes=(\d+)-(\d*)$/.exec(rangeHeader ?? '');
    if (rangeMatch && status === 200 && !options.has('--no-range')) {
        const start = Number(rangeMatch[1]);
        const end = Math.min(rangeMatch[2] === '' ? body.length - 1 : Number(rangeMatch[2]), body.length - 1);
        headers.push({'name': 'Content-Range', 'value': `bytes ${start}-${end}/${body.length}`});
        body = body.subarray(start, end + 1);
        status = 206;
    }
    headers.push({'name': 'Content-Length', 'value': String(body.length)});
    if (request.method === 'HEAD') {
        body = Buffer.alloc(0);
    }
    await send('Fetch.fulfillRequest', {requestId, 'responseCode': status, 'responseHeaders': headers, 'body': body.toString('base64')});
}
let cacheDownloads = 0;
let cacheDownloadBytes = 0;


// Evaluate an expression in the page
async function evaluate(expression) {
    const result = await send('Runtime.evaluate', {expression, 'awaitPromise': true, 'returnByValue': true});
    if (result.exceptionDetails) {
        return {'error': result.exceptionDetails.exception?.description ?? result.exceptionDetails.text};
    }
    return result.result.value;
}


// Open the check page and enable interception
await send('Runtime.enable');
await send('Network.enable', {'maxTotalBufferSize': 0, 'maxResourceBufferSize': 0});
if (!options.has('--live')) {
    await send('Fetch.enable', {'patterns': [{'urlPattern': 'https://standardebooks.org/*', 'requestStage': 'Request'}]});
}
for (let ix = 0; ix < 40; ix++) {
    await send('Page.navigate', {'url': `http://127.0.0.1:${port}/out/standardebooks/check.html`});
    await sleep(250);
    if (await evaluate('document.title') === 'Standard Ebooks check') {
        break;
    }
}
const imported = await evaluate(
    `import('/web/sources/standardebooks.js').then((module) => { window.source = module.source; return true; })`
);
if (imported !== true) {
    print(`import failed: ${JSON.stringify(imported)}`);
    process.exit(1);
}
print(`Source: ${await evaluate('JSON.stringify({id: source.id, name: source.name})')}`);
print(options.has('--live') ? 'Fetching live from standardebooks.org'
    : `Serving standardebooks.org from ${cacheDir}${options.has('--no-range') ? ' (ranges ignored)' : ''}`);


// Load checks
for (const book of books) {
    const {id} = book;
    networkBytes = 0;
    networkRequests = 0;
    const start = Date.now();
    const result = await evaluate(`(async () => {
        const progress = [];
        try {
            const result = await source.load(${JSON.stringify(book)}, (loaded, total) => progress.push([loaded, total]));
            return {...result, progress};
        } catch (error) {
            return {'error': error.message, progress};
        }
    })()`);
    const ms = Date.now() - start;
    print(`\n${id}`);
    if (result.error) {
        print(`  ERROR: ${result.error}`);
        continue;
    }
    const {text, chapters} = result;
    const lastProgress = result.progress.at(-1) ?? [];
    print(`  title: ${result.title} | author: ${result.author}`);
    print(`  text: ${text.length} chars, ${text.split('\n\n').length} paragraphs | chapters: ${chapters.length} ` +
        `(level 2: ${chapters.filter((chapter) => chapter.level === 2).length})`);
    const chapterLine = (chapter) => `${'  '.repeat(chapter.level)}[${chapter.level}] ${chapter.title}`;
    print(`  first chapters:\n${chapters.slice(0, 4).map((chapter) => `  ${chapterLine(chapter)}`).join('\n')}`);
    print(`  last chapters:\n${chapters.slice(-2).map((chapter) => `  ${chapterLine(chapter)}`).join('\n')}`);
    print(`  start: ${JSON.stringify(text.slice(0, 300))}`);
    print(`  load: ${ms} ms | download: ${(networkBytes / 1e6).toFixed(2)} MB in ${networkRequests} requests | ` +
        `progress: ${result.progress.length} calls, last ${JSON.stringify(lastProgress)}`);

    // Check the text conventions
    const problems = [];
    for (const chapter of chapters) {
        if (text.slice(chapter.char, chapter.char + chapter.title.length) !== chapter.title ||
            (chapter.char !== 0 && text.slice(chapter.char - 2, chapter.char) !== '\n\n')) {
            problems.push(`chapter offset mismatch: ${chapter.title}`);
        }
    }
    if (/\n{3,}/.test(text)) {
        problems.push('three or more newlines');
    }
    if (/[<>]/.test(text)) {
        const markupIndex = text.search(/[<>]/);
        problems.push(`markup characters: ${JSON.stringify(text.slice(Math.max(0, markupIndex - 40), markupIndex + 40))}`);
    }
    if (/[\u2060\uFEFF\u00AD\u200A\u00A0]/.test(text)) {
        problems.push('invisible or unusual space characters');
    }
    if (/ \n|\n | {2}/.test(text)) {
        problems.push('untrimmed spaces');
    }
    if (/standard ebooks|public domain|uncopyright|colophon/i.test(text)) {
        const match = /standard ebooks|public domain|uncopyright|colophon/i.exec(text);
        problems.push(`boilerplate? ${JSON.stringify(text.slice(Math.max(0, match.index - 60), match.index + 60))}`);
    }
    if (lastProgress[0] !== lastProgress[1] && lastProgress[1] !== null) {
        problems.push('progress did not finish');
    }
    if (problems.length !== 0) {
        print(`  PROBLEMS:\n${problems.slice(0, 10).map((problem) => `    ${problem}`).join('\n')}`);
    }
    if (options.has('--save')) {
        writeFileSync(join(textDir, `${id.slice('/ebooks/'.length).replaceAll('/', '_')}.txt`), text);
    }
}


// Summary
if (!options.has('--live')) {
    print(`\nCache: downloaded ${cacheDownloads} new URLs (${(cacheDownloadBytes / 1e6).toFixed(2)} MB)`);
}
print(`Console/exceptions: ${logs.length !== 0 ? `\n${logs.join('\n')}` : 'none'}`);
ws.close();
process.exit(0);
