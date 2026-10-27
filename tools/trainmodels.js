// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// make train-check's module hook - the package's tests with make train's trained models: an import of a generated model
// module (lib/models/g2pModel.js, voiceModel.js) resolves to the module of that name in TRAIN_MODELS (make train's
// TRAIN_DIR) instead - lib/models/ unchanged. NODE_OPTIONS carries it to the test runner's processes:
//
//     NODE_OPTIONS='--import=/path/to/tools/trainmodels.js' TRAIN_MODELS=build/train node --test

import {env} from 'node:process';
import {pathToFileURL} from 'node:url';
import {registerHooks} from 'node:module';
import {resolve} from 'node:path';


const modelsURL = new URL('../lib/models/', import.meta.url).href;
const trainedURL = pathToFileURL(`${resolve(env.TRAIN_MODELS)}/`).href;

registerHooks({
    resolve(specifier, context, nextResolve) {
        const resolved = nextResolve(specifier, context);
        return resolved.url.startsWith(modelsURL) ? {...resolved, 'url': `${trainedURL}${resolved.url.slice(modelsURL.length)}`} : resolved;
    }
});
