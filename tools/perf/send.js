// Queue a command for the performance lab's front ends (tools/perf/server.js) and print its result - JSON, one line
//
//     node tools/perf/send.js TYPE ['{"option": ...}'] [--client ID] [--timeout SECONDS] [--port 8090]
//
// TYPE is a front-end command (tools/perf/lab.js): timing, reload, info. With no client, the command goes to every
// client that has polled.

/* eslint-disable no-console */

import process from 'node:process';


// Queue the command, then wait for its result
async function main(argv) {
    const option = (name, fallback) => {
        const index = argv.indexOf(name);
        return index >= 0 ? argv.splice(index, 2)[1] : fallback;
    };
    const port = Number(option('--port', '8090'));
    const timeout = Number(option('--timeout', '900'));
    const client = option('--client', null);
    const [type, settings = '{}'] = argv;
    const base = `http://localhost:${port}`;
    const queued = await (await fetch(`${base}/api/command`, {
        'method': 'POST', 'body': JSON.stringify({...JSON.parse(settings), type, ...(client ? {client} : {})})
    })).json();
    if (queued.clients.length === 0 && !client) {
        console.error('no clients have polled - open the lab on the phone first');
        return 1;
    }
    const deadline = Date.now() + 1000 * timeout;
    while (Date.now() < deadline) {
        const result = await (await fetch(`${base}/api/result?id=${queued.id}`)).json();
        if (!result.pending) {
            console.log(JSON.stringify(result));
            return result.error ? 1 : 0;
        }
    }
    console.error(`no result for command ${queued.id} in ${timeout} s`);
    return 1;
}


process.exitCode = await main(process.argv.slice(2));
