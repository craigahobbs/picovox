// Check the Project Gutenberg source (static/audiobook/sources/gutenberg.js) in headless Chrome - load catalog books (each with its
// GITenberg text URL and encoding) and print each book's title, author, text length, chapters, opening text, load
// time, and download size. GITenberg's raw texts are cached under out/gutenberg/cache, so a rerun doesn't fetch them
// again (the cache serves whole, uncompressed bodies); --no-cache fetches everything live, without interception, as a
// browser would.
//
// node tools/gutenbergcheck.mjs [--no-cache] [--catalog <file> [--limit <count>]] [gutenbergId or text URL ...]
//
//   --catalog <file>  check the Gutenberg books of a catalog (e.g. data/catalog.json) - or, with ids, those books
//   --limit <count>   with --catalog, only the first count books
//
// Without a catalog, ids are looked up in the default books (16 books with known GITenberg URLs). A text URL is a
// raw.githubusercontent.com/GITenberg/<repo>/master/<file> URL - its encoding is Windows-1252 for a "-8.txt" file,
// otherwise UTF-8.

import {existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {Buffer} from 'node:buffer';
import {createHash} from 'node:crypto';
import {createServer} from 'node:net';
import {fileURLToPath} from 'node:url';
import {join} from 'node:path';
import process from 'node:process';
import {spawn} from 'node:child_process';
import {tmpdir} from 'node:os';


const root = fileURLToPath(new URL('..', import.meta.url));
const chromePath = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const cacheDir = join(root, 'out', 'gutenberg', 'cache');
const gitenbergURL = 'https://raw.githubusercontent.com/GITenberg/';
const args = process.argv.slice(2);
const useCache = !args.includes('--no-cache');
const optionValue = (name) => {
    const index = args.indexOf(name);
    return index !== -1 && index + 1 < args.length ? args[index + 1] : null;
};
const catalogPath = optionValue('--catalog');
const limit = Number(optionValue('--limit') ?? Infinity);
const bookArgs = args.filter((arg, ix) => !arg.startsWith('--') && !['--catalog', '--limit'].includes(args[ix - 1]));


// Print a line
function print(text = '') {
    process.stdout.write(`${text}\n`);
}


// The default books - novels, an illustrated edition, short stories, poetry, plays, nonfiction, and a recent release
const defaultBooks = [
    {'id': '2701', 'title': 'Moby Dick; Or, The Whale', 'author': 'Herman Melville', 'encoding': 'utf-8',
        'url': `${gitenbergURL}Moby-Dick--Or-The-Whale_2701/master/2701-0.txt`},
    {'id': '1342', 'title': 'Pride and Prejudice', 'author': 'Jane Austen', 'encoding': 'utf-8',
        'url': `${gitenbergURL}Pride-and-Prejudice_1342/master/1342-0.txt`},
    {'id': '11', 'title': 'Alice\'s Adventures in Wonderland', 'author': 'Lewis Carroll', 'encoding': 'utf-8',
        'url': `${gitenbergURL}Alice-s-Adventures-in-Wonderland_11/master/11-0.txt`},
    {'id': '84', 'title': 'Frankenstein; or, the modern prometheus', 'author': 'Mary Wollstonecraft Shelley', 'encoding': 'utf-8',
        'url': `${gitenbergURL}Frankenstein_84/master/84-0.txt`},
    {'id': '1661', 'title': 'The Adventures of Sherlock Holmes', 'author': 'Arthur Conan Doyle', 'encoding': 'utf-8',
        'url': `${gitenbergURL}The-Adventures-of-Sherlock-Holmes_1661/master/1661.txt`},
    {'id': '98', 'title': 'A Tale of Two Cities', 'author': 'Charles Dickens', 'encoding': 'utf-8',
        'url': `${gitenbergURL}A-Tale-of-Two-Cities_98/master/98-0.txt`},
    {'id': '345', 'title': 'Dracula', 'author': 'Bram Stoker', 'encoding': 'utf-8',
        'url': `${gitenbergURL}Dracula_345/master/345.txt`},
    {'id': '100', 'title': 'The Complete Works of William Shakespeare', 'author': 'William Shakespeare', 'encoding': 'utf-8',
        'url': `${gitenbergURL}The-Complete-Works-of-William-Shakespeare_100/master/100-0.txt`},
    {'id': '1524', 'title': 'Hamlet', 'author': 'William Shakespeare', 'encoding': 'utf-8',
        'url': `${gitenbergURL}Hamlet-Prince-of-Denmark_1524/master/1524-0.txt`},
    {'id': '844', 'title': 'The Importance of Being Earnest', 'author': 'Oscar Wilde', 'encoding': 'utf-8',
        'url': `${gitenbergURL}The-Importance-of-Being-Earnest--A-Trivial-Comedy-for-Serious-People_844/master/844.txt`},
    {'id': '1322', 'title': 'Leaves of Grass', 'author': 'Walt Whitman', 'encoding': 'utf-8',
        'url': `${gitenbergURL}Leaves-of-Grass_1322/master/1322.txt`},
    {'id': '12242', 'title': 'Poems by Emily Dickinson, Three Series, Complete', 'author': 'Emily Dickinson', 'encoding': 'utf-8',
        'url': `${gitenbergURL}Poems-by-Emily-Dickinson-Three-Series-Complete_12242/master/12242.txt`},
    {'id': '2814', 'title': 'Dubliners', 'author': 'James Joyce', 'encoding': 'utf-8',
        'url': `${gitenbergURL}Dubliners_2814/master/2814-0.txt`},
    {'id': '205', 'title': 'Walden, and On The Duty Of Civil Disobedience', 'author': 'Henry David Thoreau', 'encoding': 'utf-8',
        'url': `${gitenbergURL}Walden-and-On-The-Duty-Of-Civil-Disobedience_205/master/205-0.txt`},
    {'id': '28885', 'title': 'Alice\'s Adventures in Wonderland', 'author': 'Lewis Carroll', 'encoding': 'utf-8',
        'url': `${gitenbergURL}Alice-s-Adventures-in-WonderlandIllustrated-by-Arthur-Rackham.-With-a-Proem-by-Austin-Dobson_28885/` +
            'master/28885.txt'},
    {'id': '75000', 'title': 'Folk tales from Tibet', 'author': '', 'encoding': 'utf-8',
        'url': `${gitenbergURL}Folk-tales-from-Tibet-b-With-illustrations-by-a-Tibetan-artist-and-some-verses-from-Tibetan-l__75000/` +
            'master/75000-0.txt'}
].map(({id, title, author, encoding, url}) => (
    {'source': 'gutenberg', id, title, author, 'text': {url, 'format': 'gutenberg-text', encoding}}
));


// The books to check - a catalog's, ids in the catalog or the default books, or text URLs
const catalogBooks = catalogPath !== null
    ? JSON.parse(readFileSync(catalogPath, 'utf8')).books.filter((book) => book.source === 'gutenberg')
    : defaultBooks;
let books;
if (bookArgs.length === 0) {
    books = catalogBooks.slice(0, limit);
} else {
    books = bookArgs.map((arg) => {
        if (/^https?:/.test(arg)) {
            const id = /_(\d+)\/[^/]+\/[^/]+$/.exec(arg)?.[1] ?? arg;
            const encoding = /-8\.txt$/.test(arg) ? 'windows-1252' : 'utf-8';
            return {'source': 'gutenberg', id, 'title': '', 'author': '', 'text': {'url': arg, 'format': 'gutenberg-text', encoding}};
        }
        return catalogBooks.find((book) => book.id === arg) ?? null;
    });
    for (const [ix, book] of books.entries()) {
        if (book === null) {
            print(`Unknown book ${bookArgs[ix]} - give its text URL, or a catalog that has it`);
            process.exit(1);
        }
    }
}


// A free local port
function freePort() {
    return new Promise((resolve) => {
        const server = createServer();
        server.listen(0, '127.0.0.1', () => {
            const {port} = server.address();
            server.close(() => resolve(port));
        });
    });
}

const sleep = (ms) => new Promise((resolve) => {
    setTimeout(resolve, ms);
});


// Start the static server and headless Chrome
const httpPort = await freePort();
const debugPort = await freePort();
const profile = mkdtempSync(join(tmpdir(), 'gutenbergcheck-'));
const server = spawn('python3', ['-m', 'http.server', String(httpPort), '--bind', '127.0.0.1'], {'cwd': root, 'stdio': 'ignore'});
const chrome = spawn(chromePath, ['--headless=new', '--disable-gpu', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
    'about:blank'], {'stdio': 'ignore'});

// Stop the server and Chrome on exit
let stopped = false;
function stop() {
    if (!stopped) {
        stopped = true;
        chrome.kill();
        server.kill();
        try {
            rmSync(profile, {'recursive': true, 'force': true});
        } catch {
            // Chrome may still hold the profile
        }
    }
}
process.on('exit', stop);
process.on('SIGINT', () => process.exit(130));


try {
    // Connect to the page target
    let targets = [];
    for (let ix = 0; ix < 50 && targets.length === 0; ix++) {
        await sleep(200);
        try {
            targets = (await (await fetch(`http://127.0.0.1:${debugPort}/json`)).json()).filter((target) => target.type === 'page');
        } catch {
            // Not listening yet
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
        messageId++;
        pending.set(messageId, resolve);
        ws.send(JSON.stringify({'id': messageId, method, params}));
    });

    // Serve remote requests from the cache, or fetch them and cache the responses
    mkdirSync(cacheDir, {'recursive': true});
    const stats = {'cached': 0, 'live': 0};
    const cachePath = (url) => join(cacheDir, createHash('sha1').update(url).digest('hex').slice(0, 20));
    const onRequestPaused = async (params) => {
        const {requestId, request, responseStatusCode, responseHeaders} = params;
        const path = cachePath(request.url);
        if (responseStatusCode === undefined) {
            // The request stage - from the cache, or on to the network
            if (request.method === 'OPTIONS') {
                await send('Fetch.continueRequest', {requestId});
                return;
            }
            if (existsSync(`${path}.json`)) {
                const meta = JSON.parse(readFileSync(`${path}.json`, 'utf8'));
                const body = readFileSync(`${path}.body`);
                stats.cached++;
                await send('Fetch.fulfillRequest', {requestId, 'responseCode': meta.status, 'responseHeaders': [
                    {'name': 'Content-Type', 'value': meta.contentType},
                    {'name': 'Content-Length', 'value': String(body.length)},
                    {'name': 'Access-Control-Allow-Origin', 'value': '*'}
                ], 'body': body.toString('base64')});
                return;
            }
            stats.live++;
            await send('Fetch.continueRequest', {requestId, 'interceptResponse': true});
            return;
        }

        // The response stage - cache a successful or not-found response
        if (responseStatusCode === 200 || responseStatusCode === 404) {
            const result = await send('Fetch.getResponseBody', {requestId});
            if (result && result.body !== undefined) {
                const body = result.base64Encoded ? Buffer.from(result.body, 'base64') : Buffer.from(result.body, 'utf8');
                const contentTypeHeader = responseHeaders.find((header) => header.name.toLowerCase() === 'content-type');
                const contentType = contentTypeHeader?.value || 'text/plain';
                writeFileSync(`${path}.body`, body);
                writeFileSync(`${path}.json`, JSON.stringify({'url': request.url, 'status': responseStatusCode, contentType}));
            }
        }
        await send('Fetch.continueRequest', {requestId});
    };
    ws.addEventListener('message', (event) => {
        const message = JSON.parse(event.data);
        if (message.id && pending.has(message.id)) {
            pending.get(message.id)(message.result);
            pending.delete(message.id);
        } else if (message.method === 'Fetch.requestPaused') {
            onRequestPaused(message.params).catch((error) => logs.push(`requestPaused: ${error.message}`));
        } else if (message.method === 'Runtime.exceptionThrown' ||
            (message.method === 'Runtime.consoleAPICalled' && message.params.type === 'error')) {
            logs.push(JSON.stringify(message.params).slice(0, 300));
        }
    });
    await send('Runtime.enable');
    if (useCache) {
        await send('Fetch.enable', {'patterns': [{'urlPattern': 'https://raw.githubusercontent.com/*', 'requestStage': 'Request'}]});
    }
    const evaluate = async (expression) => {
        const result = await send('Runtime.evaluate', {expression, 'awaitPromise': true, 'returnByValue': true});
        if (result.exceptionDetails) {
            throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
        }
        return result.result.value;
    };

    // Load a page from the server, so the module imports from the page's origin
    await send('Page.navigate', {'url': `http://127.0.0.1:${httpPort}/package.json`});
    await sleep(500);
    await evaluate(`import('/static/audiobook/sources/gutenberg.js').then((module) => { window.gutenberg = module; return true; })`);
    print(`Source: ${await evaluate('JSON.stringify({id: gutenberg.source.id, name: gutenberg.source.name})')}\n`);

    // Load each book - a second apart when fetching live, to be considerate to GitHub
    for (const book of books) {
        const statsLive = stats.live;
        const result = await evaluate(`(async () => {
            const progress = [];
            const start = performance.now();
            try {
                const result = await gutenberg.source.load(${JSON.stringify(book)}, (loaded, total) => progress.push([loaded, total]));
                const ms = performance.now() - start;
                const chapters = result.chapters;
                const badOffsets = chapters.filter((chapter) => !result.text.startsWith(chapter.title, chapter.char)).length;
                return {'title': result.title, 'author': result.author, 'length': result.text.length, 'chapters': chapters.length,
                    'levels': [1, 2].map((level) => chapters.filter((chapter) => chapter.level === level).length),
                    'first': chapters.slice(0, 4).map((chapter) => chapter.title),
                    'last': chapters.slice(-2).map((chapter) => chapter.title),
                    'opening': result.text.slice(0, 300), ms, 'bytes': progress.length ? progress.at(-1)[0] : 0,
                    'progressCalls': progress.length, 'totalKnown': progress.filter(([, total]) => total !== null).length, badOffsets};
            } catch (error) {
                return {'error': error.message, 'ms': performance.now() - start};
            }
        })()`);
        print(`== ${book.id}: ${book.title} | ${book.author}`);
        if (result.error) {
            print(`   error (${Math.round(result.ms)} ms): ${result.error}\n`);
        } else {
            print(`   title: ${result.title} | author: ${result.author}`);
            const badOffsets = result.badOffsets ? `, ${result.badOffsets} BAD OFFSETS` : '';
            print(`   text ${result.length} chars, ${result.chapters} chapters ` +
                `(${result.levels[0]} level 1, ${result.levels[1]} level 2)${badOffsets}`);
            print(`   load ${Math.round(result.ms)} ms, download ${result.bytes} bytes, ${result.progressCalls} progress calls ` +
                `(${result.totalKnown} with a total)`);
            print(`   first chapters: ${result.first.join(' | ')}`);
            print(`   last chapters: ${result.last.join(' | ')}`);
            print(`   text: ${JSON.stringify(result.opening)}\n`);
        }
        if (!useCache || stats.live !== statsLive) {
            await sleep(1000);
        }
    }
    if (useCache) {
        print(`requests: ${stats.cached} from the cache, ${stats.live} live`);
    }
    print(`console errors and exceptions: ${logs.length ? `\n${logs.join('\n')}` : 'none'}`);
    ws.close();
} finally {
    stop();
}
process.exit(0);
