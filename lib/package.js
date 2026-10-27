// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// Packaged models - an acoustic model generated as a JavaScript module (picovox_train/package.py: its manifest, and its weights as
// base64, raw or deflated) loaded by a plain import, the same in Node, the browser, and a Web Worker (no fetch, no file
// system - only the built-in atob, and DecompressionStream for a deflated package):
//
//     import * as voice from './multi.js';
//     import {loadPackagedModel} from './lib/package.js';
//     const model = await loadPackagedModel(voice);     // as lib/model.js loadModel(manifest, buffer)

import {base64Bytes, inflate} from './g2p.js';
import {loadModel} from './model.js';


// A packaged module's weights - a Promise of the export's .bin bytes (an ArrayBuffer)
export function packagedWeights(module) {
    const bytes = base64Bytes(module.weights);
    if (module.encoding === 'raw') {
        return Promise.resolve(bytes.buffer);
    }
    if (module.encoding === 'deflate-raw') {
        return inflate(bytes).then((inflated) => inflated.buffer);
    }
    return Promise.reject(new Error(`unknown package encoding ${module.encoding}`));
}


// Load a packaged model (the generated module's namespace - {manifest, encoding, weights}) - a Promise of the model
export async function loadPackagedModel(module) {
    return loadModel(module.manifest, await packagedWeights(module));
}
