// Serve the repository statically, for the browser page (static/audiobook/index.html)
//
// node tools/serve.js [-p port]

import {extname, join, normalize, sep} from 'node:path';
import {createServer} from 'node:http';
import {fileURLToPath} from 'node:url';
import process from 'node:process';
import {readFile} from 'node:fs/promises';


const root = fileURLToPath(new URL('..', import.meta.url));
const args = process.argv.slice(2);
const port = args[0] === '-p' ? Number(args[1]) : 8000;
const types = {
    '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
    '.json': 'application/json', '.wav': 'audio/wav', '.png': 'image/png', '.dict': 'text/plain; charset=utf-8'
};


// Serve files under the repository root - a directory serves its index.html
createServer(async (request, response) => {
    const path = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    let file = normalize(join(root, path));
    if (!file.startsWith(root.replace(/[/\\]$/, '') + sep) && file !== root.replace(/[/\\]$/, '')) {
        response.writeHead(403).end();
        return;
    }
    if (path.endsWith('/')) {
        file = join(file, 'index.html');
    }
    try {
        const body = await readFile(file);
        response.writeHead(200, {'Content-Type': types[extname(file)] || 'application/octet-stream'}).end(body);
    } catch {
        response.writeHead(404).end('Not found');
    }
}).listen(port, () => {
    process.stdout.write(`http://localhost:${port}/static/audiobook/\n`);
});
