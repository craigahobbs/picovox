// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// Check a packaged model (nn/package.py's generated module) against its export (the .json manifest and .bin weights):
// the package and the loader import with no Node built-ins (any node: import from them fails, and
// process.getBuiltinModule is hidden while they load, as in a browser), every tensor loads bit-identically, the model
// speaks the same frames, and the same import works in a worker thread (as in a Web Worker).
//
//     node nn/js/packagecheck.js build/nn/packages/multi.js data/nn/multi
//
// It prints one JSON line - the tensors and frames compared, and the load times.

/* eslint-disable no-console */

import {Worker} from 'node:worker_threads';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import process from 'node:process';
import {readFileSync} from 'node:fs';
import {registerHooks} from 'node:module';


const [modulePath, exportPath] = process.argv.slice(2);
const moduleUrl = pathToFileURL(modulePath).href;

// Fail any node: or bare import from the package (lib/) and the model module
const browserRoots = [new URL('../../lib/', import.meta.url).href];
registerHooks({
    resolve(specifier, context, nextResolve) {
        const parent = context.parentURL ?? '';
        if ((browserRoots.some((root) => parent.startsWith(root)) || parent === moduleUrl) &&
            (specifier.startsWith('node:') || !/^[./]/u.test(specifier))) {
            throw new Error(`${parent} imports ${specifier}`);
        }
        return nextResolve(specifier, context);
    }
});


// A hash of a model's tensors (names, shapes, and float64 data)
function modelHash(model) {
    const hash = createHash('sha256');
    for (const name of Object.keys(model.tensors).sort()) {
        const {data, shape} = model.tensors[name];
        hash.update(`${name}:${shape.join(',')};`);
        hash.update(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    }
    return hash.digest('hex');
}


// The worker - import the package and the loader, and post the model's hash
const workerSource = `
import {parentPort, workerData} from 'node:worker_threads';
const {loadPackagedModel} = await import(workerData.loader);
const voice = await import(workerData.module);
const model = await loadPackagedModel(voice);
const {createHash} = await import('node:crypto');
const hash = createHash('sha256');
for (const name of Object.keys(model.tensors).sort()) {
    const {data, shape} = model.tensors[name];
    hash.update(name + ':' + shape.join(',') + ';');
    hash.update(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
}
parentPort.postMessage(hash.digest('hex'));
`;


async function main() {
    // The package and the loader, as a browser would load them
    const {getBuiltinModule} = process;
    process.getBuiltinModule = undefined;
    const startImport = performance.now();
    const voice = await import(moduleUrl);
    const {loadPackagedModel} = await import('../../lib/package.js');
    const importSeconds = (performance.now() - startImport) / 1000;
    const startLoad = performance.now();
    const packaged = await loadPackagedModel(voice);
    const loadSeconds = (performance.now() - startLoad) / 1000;
    // eslint-disable-next-line require-atomic-updates
    process.getBuiltinModule = getBuiltinModule;
    // (the frames' words pronounced by the G2P network, as the package pronounces)
    const {setPronouncer} = await import('../../lib/lexicon.js');
    const {loadG2p} = await import('../../lib/g2p.js');
    const {g2pModel} = await import('../../lib/models/g2pModel.js');
    setPronouncer(await loadG2p(g2pModel));

    // The export, loaded from its files
    const {loadModel, speakFrames} = await import('../../lib/model.js');
    const buffer = readFileSync(`${exportPath}.bin`);
    const exported = loadModel(JSON.parse(readFileSync(`${exportPath}.json`, 'utf8')),
        buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));

    // Every tensor bit-identical
    let tensors = 0;
    for (const [name, tensor] of Object.entries(exported.tensors)) {
        const other = packaged.tensors[name];
        if (!other || other.data.length !== tensor.data.length || other.data.some((value, ix) => !Object.is(value, tensor.data[ix]))) {
            throw new Error(`tensor ${name} differs`);
        }
        tensors++;
    }
    // (the manifest as exported - nn/package.py --provenance adds only its provenance record)
    const packagedManifest = {...packaged.manifest};
    delete packagedManifest.provenance;
    if (Object.keys(packaged.tensors).length !== tensors || JSON.stringify(packagedManifest) !== JSON.stringify(exported.manifest)) {
        throw new Error('the package\'s tensors or manifest differ');
    }

    // The same frames
    const texts = ['Call me Ishmael.', '"Where are you going?" she asked, and he said nothing at all.'];
    let frames = 0;
    for (const text of texts) {
        const first = speakFrames(exported, text);
        const second = speakFrames(packaged, text);
        if (first.frames.length !== second.frames.length ||
            first.frames.some((frame, ix) => frame.some((value, jx) => !Object.is(value, second.frames[ix][jx])))) {
            throw new Error(`frames differ: ${text}`);
        }
        frames += first.frames.length;
    }

    // The same model in a worker thread
    const worker = new Worker(new URL(`data:text/javascript,${encodeURIComponent(workerSource)}`),
        {'workerData': {'loader': new URL('../../lib/package.js', import.meta.url).href, 'module': moduleUrl}});
    const workerHash = await new Promise((resolve, reject) => {
        worker.once('message', resolve);
        worker.once('error', reject);
    });
    await worker.terminate();
    if (workerHash !== modelHash(exported)) {
        throw new Error('the worker\'s model differs');
    }
    console.log(JSON.stringify({'package': modulePath, 'encoding': voice.encoding, tensors, frames, 'identical': true, 'worker': true,
        importSeconds, loadSeconds}));
}

await main();
