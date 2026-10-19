// The performance lab's backend - serves the repository (the front end, tools/perf/index.html, imports the package's
// lib/ - or a variant of it, build/perf/variants/<name>/lib/ - straight from it) to a phone on the local network, queues
// commands for the front ends, which poll for them, and records their results in build/perf/results.jsonl.
//
//     node tools/perf/server.js [--port 8090] [--root DIR]  # then open the printed URL on the phone (DIR: serve another
//                                                          # app - PicoShelf, say - instead of the repository)
//     node tools/perf/send.js timing '{"lib": "lib"}'     # queue a command and print its result (tools/perf/send.js)
//
// The API (JSON):
// - GET /api/next?client=ID - the client's next command (waits up to 20 s - long polling), else {'type': 'none'}
// - POST /api/result - a client's result ({client, command, ...}), appended to build/perf/results.jsonl
// - POST /api/command - queue a command ({type, ...} - for every client, or {client} for one) - returns its id
// - GET /api/result?id=N - the result of command N (waits up to 20 s), else {'pending': true}
// - GET /api/clients - the clients and when each last polled
// - POST /log - an app's log (PicoShelf's ?timing&log=URL timeline), appended to build/perf/devlog.jsonl

import {appendFileSync, createReadStream, existsSync, mkdirSync, statSync} from 'node:fs';
import {dirname, extname, join, normalize, resolve} from 'node:path';
import {Buffer} from 'node:buffer';
import {createServer} from 'node:http';
import {fileURLToPath} from 'node:url';
import {networkInterfaces} from 'node:os';
import process from 'node:process';


// The repository, the results file, and the long poll's wait (ms)
const repository = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const resultsPath = join(repository, 'build/perf/results.jsonl');
const devlogPath = join(repository, 'build/perf/devlog.jsonl');
let servedRoot = repository;
const waitMs = 20000;


// The served files' content types
const contentTypes = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.wav': 'audio/wav'
};


// The queue - commands by id, each client's pending command ids, the clients' last polls, the results by command id, and
// the waiters (long polls) for a client's command or a command's result
const commands = new Map();
const pending = new Map();
const clients = new Map();
const results = new Map();
const commandWaiters = new Map();
const resultWaiters = new Map();
let nextId = 1;


// Read a request's JSON body
function readJson(request) {
    return new Promise((resolveBody, reject) => {
        const chunks = [];
        request.on('data', (chunk) => chunks.push(chunk));
        request.on('end', () => {
            try {
                resolveBody(JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}'));
            } catch (error) {
                reject(error);
            }
        });
        request.on('error', reject);
    });
}


// Send a JSON response
function sendJson(response, value, status = 200) {
    response.writeHead(status, {'Content-Type': contentTypes['.json'], 'Cache-Control': 'no-store'});
    response.end(JSON.stringify(value));
}


// Queue a command for one client or every known client - its id
export function queueCommand(command) {
    const id = nextId++;
    const entry = {...command, id};
    commands.set(id, entry);
    const targets = command.client ? [command.client] : [...clients.keys()];
    for (const client of targets) {
        if (!pending.has(client)) {
            pending.set(client, []);
        }
        pending.get(client).push(id);
        const waiter = commandWaiters.get(client);
        if (waiter) {
            commandWaiters.delete(client);
            waiter();
        }
    }
    return id;
}


// A client's next command, or null
function takeCommand(client) {
    const queue = pending.get(client) ?? [];
    const id = queue.shift();
    return id === undefined ? null : commands.get(id);
}


// Wait (up to waitMs) for a value - check() is tried now and whenever wake is called, null meaning not yet
function waitFor(waiters, key, check) {
    return new Promise((resolveValue) => {
        const now = check();
        if (now !== null) {
            resolveValue(now);
            return;
        }
        const timer = setTimeout(() => {
            waiters.delete(key);
            resolveValue(null);
        }, waitMs);
        waiters.set(key, () => {
            clearTimeout(timer);
            resolveValue(check());
        });
    });
}


// Record a client's result
function recordResult(result) {
    mkdirSync(dirname(resultsPath), {'recursive': true});
    appendFileSync(resultsPath, `${JSON.stringify({...result, 'received': new Date().toISOString()})}\n`);
    if (result.command !== undefined) {
        results.set(result.command, result);
        const waiter = resultWaiters.get(result.command);
        if (waiter) {
            resultWaiters.delete(result.command);
            waiter();
        }
    }
}


// Serve a repository file (no caching - a reload picks up every change)
function serveFile(response, urlPath) {
    const path = normalize(join(servedRoot, decodeURIComponent(urlPath)));
    if (!path.startsWith(servedRoot) || !existsSync(path) || !statSync(path).isFile()) {
        response.writeHead(404, {'Content-Type': 'text/plain'});
        response.end('not found');
        return;
    }
    response.writeHead(200, {'Content-Type': contentTypes[extname(path)] ?? 'application/octet-stream', 'Cache-Control': 'no-store'});
    createReadStream(path).pipe(response);
}


// Handle a request
async function handle(request, response) {
    const url = new URL(request.url, 'http://localhost');
    const client = url.searchParams.get('client');
    if (url.pathname === '/' && servedRoot === repository) {
        response.writeHead(302, {'Location': '/tools/perf/index.html'});
        response.end();
    } else if (url.pathname === '/' || url.pathname.endsWith('/')) {
        serveFile(response, `${url.pathname}index.html`);
    } else if (url.pathname === '/log' && request.method === 'POST') {
        const body = await new Promise((resolveBody) => {
            const chunks = [];
            request.on('data', (chunk) => chunks.push(chunk));
            request.on('end', () => resolveBody(Buffer.concat(chunks).toString('utf8')));
        });
        mkdirSync(dirname(devlogPath), {'recursive': true});
        appendFileSync(devlogPath, `${JSON.stringify({'received': new Date().toISOString(), 'body': body})}\n`);
        sendJson(response, {'ok': true});
    } else if (url.pathname === '/api/next' && client) {
        clients.set(client, {'lastPoll': Date.now(), 'agent': request.headers['user-agent'] ?? ''});
        if (!pending.has(client)) {
            pending.set(client, []);
        }
        const command = await waitFor(commandWaiters, client, () => takeCommand(client));
        clients.set(client, {'lastPoll': Date.now(), 'agent': request.headers['user-agent'] ?? ''});
        sendJson(response, command ?? {'type': 'none'});
    } else if (url.pathname === '/api/result' && request.method === 'POST') {
        recordResult(await readJson(request));
        sendJson(response, {'ok': true});
    } else if (url.pathname === '/api/command' && request.method === 'POST') {
        sendJson(response, {'id': queueCommand(await readJson(request)), 'clients': [...clients.keys()]});
    } else if (url.pathname === '/api/result') {
        const id = Number(url.searchParams.get('id'));
        const result = await waitFor(resultWaiters, id, () => results.get(id) ?? null);
        sendJson(response, result ?? {'pending': true});
    } else if (url.pathname === '/api/clients') {
        sendJson(response, Object.fromEntries([...clients].map(([key, value]) => [key, {...value, 'ago': Date.now() - value.lastPoll}])));
    } else {
        serveFile(response, url.pathname);
    }
}


// The local network's IPv4 addresses
function localAddresses() {
    return Object.values(networkInterfaces()).flat().filter((entry) => entry.family === 'IPv4' && !entry.internal)
        .map((entry) => entry.address);
}


// Start the server
function main(argv) {
    const portIndex = argv.indexOf('--port');
    const port = portIndex >= 0 ? Number(argv[portIndex + 1]) : 8090;
    const rootIndex = argv.indexOf('--root');
    if (rootIndex >= 0) {
        servedRoot = resolve(argv[rootIndex + 1]);
    }
    const server = createServer((request, response) => {
        handle(request, response).catch((error) => sendJson(response, {'error': error.message}, 500));
    });
    server.listen(port, '0.0.0.0', () => {
        for (const address of localAddresses()) {
            process.stdout.write(`perf lab: http://${address}:${port}/\n`);
        }
    });
}


if (process.argv[1] === fileURLToPath(import.meta.url)) {
    main(process.argv.slice(2));
}
