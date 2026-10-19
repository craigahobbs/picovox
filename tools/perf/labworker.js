// The performance lab's timing worker - runs one timing (tools/perf/timing.js) in a Web Worker, as PicoShelf synthesizes
// in one, and posts its result: in {'type': 'timing', ...options}; out {'type': 'result', result} or {'type': 'error',
// message}. The page starts a new worker for each timing - a fresh module map and JIT, so a reloaded lib is used.

import {runBench, runTiming} from './timing.js';


self.onmessage = (event) => {
    (event.data.bench ? Promise.resolve(runBench(event.data)) : runTiming(event.data)).then((result) => {
        self.postMessage({'type': 'result', result});
    }).catch((error) => {
        self.postMessage({'type': 'error', 'message': `${error.message} (${error.stack})`});
    });
};
