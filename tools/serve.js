// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// The development servers' static files (tools/trainab.js, tools/perf/server.js) - the served files' content types, and a
// file sent from under a root directory:
//
//     import {sendFile} from './serve.js';
//     sendFile(response, resolve('build/voicelab'), url.pathname);

import {createReadStream, existsSync, statSync} from 'node:fs';
import {extname, join, normalize, sep} from 'node:path';


// The served files' content types
export const contentTypes = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.wav': 'audio/wav'
};


// A URL path's file path under a root directory - null for a malformed escape
function filePath(root, urlPath) {
    try {
        return normalize(join(root, decodeURIComponent(urlPath)));
    } catch {
        return null;
    }
}


// Send a file under a root directory (an absolute path - no caching, so a reload picks up every change) - only a file under
// the root (not a directory, nor a file of a sibling directory whose name starts with the root's), else a 404
export function sendFile(response, root, urlPath) {
    const path = filePath(root, urlPath);
    if (path === null || !path.startsWith(root + sep) || !existsSync(path) || !statSync(path).isFile()) {
        response.writeHead(404, {'Content-Type': 'text/plain'});
        response.end('not found');
        return;
    }
    response.writeHead(200, {'Content-Type': contentTypes[extname(path)] ?? 'application/octet-stream', 'Cache-Control': 'no-store'});
    createReadStream(path).pipe(response);
}
