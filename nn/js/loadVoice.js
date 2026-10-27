// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// Load a voice model under Node - a packaged module (PATH.js, e.g. lib/models/voiceModel.js - lib/package.js) or an
// export (picovox_train.export's PATH.json and PATH.bin - lib/model.js):
//
//     import {loadVoice} from './nn/js/loadVoice.js';
//     const model = await loadVoice('lib/models/voiceModel.js');

import {loadModel} from '../../lib/model.js';
import {loadPackagedModel} from '../../lib/package.js';
import {pathToFileURL} from 'node:url';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';


// Load a packaged module or an export (its path without the extension)
export async function loadVoice(path) {
    if (path.endsWith('.js')) {
        return loadPackagedModel(await import(pathToFileURL(resolve(path)).href));
    }
    // (the weights' bytes as an ArrayBuffer of their own - a Node buffer may share a larger one)
    const bytes = readFileSync(`${path}.bin`);
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    return loadModel(JSON.parse(readFileSync(`${path}.json`, 'utf8')), buffer);
}
