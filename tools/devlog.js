// A development log server - receives the app's startup timeline from a device (the app opened with
// ?timing&log=http://<this computer>:8790) and prints each new report, so a phone's behavior can be read from here
//
// node tools/devlog.js [port]

import {appendFileSync} from 'node:fs';
import {createServer} from 'node:http';
import process from 'node:process';


const port = Number(process.argv[2] || 8790);
const logPath = new URL('../build/devlog.jsonl', import.meta.url);
createServer((request, response) => {
    let body = '';
    request.on('data', (chunk) => {
        body += chunk;
    });
    request.on('end', () => {
        if (request.method === 'POST' && body !== '') {
            const line = {'time': new Date().toISOString(), 'from': request.socket.remoteAddress, 'report': body};
            appendFileSync(logPath, `${JSON.stringify(line)}\n`);
            process.stdout.write(`--- ${line.time} from ${line.from}\n${body}\n`);
        }
        response.writeHead(204, {'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': '*'});
        response.end();
    });
}).listen(port, () => process.stdout.write(`Listening on port ${port} - writing to ${logPath.pathname}\n`));
