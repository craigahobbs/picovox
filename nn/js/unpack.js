// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

// Unpack a packaged model module (nn/package.py's generated JavaScript, e.g. lib/models/voiceModel.js) back into its
// export - the manifest (OUT.json) and the weights (OUT.bin) - so the export's tools (nn/js/speak.js, nn/panel.py) can
// speak the shipped model as packaged:
//
//     node nn/js/unpack.js lib/models/voiceModel.js OUT

import {resolve} from 'node:path';
import {packagedWeights} from './package.js';
import {pathToFileURL} from 'node:url';
import process from 'node:process';
import {writeFileSync} from 'node:fs';


const [modulePath, outPath] = process.argv.slice(2);
if (!modulePath || !outPath) {
    process.stderr.write('usage: node nn/js/unpack.js MODULE.js OUT\n');
    process.exit(2);
}
const module = await import(pathToFileURL(resolve(modulePath)).href);
writeFileSync(`${outPath}.json`, JSON.stringify(module.manifest));
writeFileSync(`${outPath}.bin`, new Uint8Array(await packagedWeights(module)));
