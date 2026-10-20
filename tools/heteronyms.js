// Heteronym and names regression check - the sentences of data/nn/heteronyms-test.json and data/nn/names-test.json through
// the package's front end (lib/lexicon.js textToPhrases with the G2P network as the pronouncer), each expected reading
// scored: right when the word's phones are one of the reading's variants (or match one without stress and no rejected
// variant's). Kinds: "default" (the reading a context-free G2P should give), "context" (another reading - needs the
// neighboring words), "missing" (no dictionary variant), "names" (surnames, first names, and place names - verified
// pronunciations: the readers' audio, the dictionary's own derived forms, and well-established names). Prints one line and
// exits 1 when a kind's score is below its minimum (the make train gate): default 0.9, context 0.85 when the module has a
// context network (else 0), missing 0, names the shipped module's score when it was added (it can only go up).
//
// node tools/heteronyms.js [-m modulePath] [--min-default 0.9] [--min-context 0.85] [--min-missing 0] [--min-names N] [--show]

import {setPronouncer, textToPhrases} from '../lib/lexicon.js';
import {loadG2p} from '../lib/g2p.js';
import {pathToFileURL} from 'node:url';
import process from 'node:process';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';


// A phone string without stress digits
function unstressed(phones) {
    return phones.replace(/\d/g, '');
}


// Whether a word's phones are the expected reading
export function isReading(phones, {accept, reject}) {
    if (accept.includes(phones)) {
        return true;
    }
    return accept.some((variant) => unstressed(variant) === unstressed(phones)) &&
        !reject.some((variant) => unstressed(variant) === unstressed(phones));
}


// The phones of a word's nth occurrence in a text, through the front end (null if the word isn't found)
export function wordPhones(text, word, occurrence = 1) {
    const words = textToPhrases(text).flatMap((phrase) => phrase.words).filter((item) => item.text === word);
    const found = words[occurrence - 1];
    return found ? found.phones.map(({ph, stress}) => (stress === null ? ph : `${ph}${stress}`)).join(' ') : null;
}


// Score the regression set with the current pronouncer - {kinds: {kind: {right, total}}, misses: [...]}
export function scoreHeteronyms(items) {
    const kinds = {};
    const misses = [];
    for (const item of items) {
        const phones = wordPhones(item.text, item.word, item.occurrence);
        const right = phones !== null && isReading(phones, item);
        kinds[item.kind] ??= {'right': 0, 'total': 0};
        kinds[item.kind].total += 1;
        kinds[item.kind].right += right ? 1 : 0;
        if (!right) {
            misses.push({...item, phones});
        }
    }
    return {kinds, misses};
}


// The gate's minimums - the default reading, the context-dependent readings (when the module has a context network; a
// context-free G2P gives one reading per word), and the names (the shipped module's score - raised to 0.91 with the
// dictionary fixes, 67/73, and to 0.93 with Don Quixote's names, 80/86)
export const minimums = {'default': 0.9, 'context': 0.85, 'missing': 0, 'names': 0.93};


// The one-line result - kinds' scores against their minimums, and whether any is below
export function resultLine(kinds, kindMinimums) {
    const parts = [];
    const below = [];
    for (const [kind, {right, total}] of Object.entries(kinds)) {
        const share = right / total;
        if (share < kindMinimums[kind]) {
            below.push(kind);
        }
        parts.push(`${kind} ${right}/${total} (${(100 * share).toFixed(1)}%, min ${kindMinimums[kind]})`);
    }
    const status = below.length ? ` - FAILED (${below.join(', ')})` : ' - ok';
    return {'line': `heteronyms: ${parts.join(', ')}${status}`, 'failed': below.length !== 0};
}


// Main - load the G2P module, score, print one line (and the misses with --show) - returns true when a kind is below its
// minimum
async function main() {
    const args = process.argv.slice(2);
    const option = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
    const modulePath = option('-m', null);
    const {g2pModel} = await import(modulePath ? pathToFileURL(resolve(modulePath)).href : '../lib/models/g2pModel.js');
    setPronouncer(await loadG2p(g2pModel));
    const items = ['heteronyms-test.json', 'names-test.json'].flatMap(
        (name) => JSON.parse(readFileSync(new URL(`../data/nn/${name}`, import.meta.url), 'utf8'))
    );
    const {kinds, misses} = scoreHeteronyms(items);
    const kindMinimums = {
        'default': Number(option('--min-default', minimums.default)),
        'context': Number(option('--min-context', g2pModel.context ? minimums.context : 0)),
        'missing': Number(option('--min-missing', minimums.missing)),
        'names': Number(option('--min-names', minimums.names))
    };
    const {line, failed} = resultLine(kinds, kindMinimums);
    process.stdout.write(`${line}\n`);
    if (args.includes('--show')) {
        for (const miss of misses) {
            const got = String(miss.phones).padEnd(22);
            process.stdout.write(`  ${miss.kind.padEnd(8)} ${miss.word.padEnd(10)} ${got} want ${miss.accept[0]}: ${miss.text}\n`);
        }
    }
    return failed;
}


if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
    const failed = await main();
    process.exitCode = failed ? 1 : 0;
}
