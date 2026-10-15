// Heteronym regression check - the sentences of data/nn/heteronyms-test.json through the app's front end (lib/lexicon.js
// textToPhrases with the G2P network as the pronouncer), each expected reading scored: right when the word's phones are
// one of the reading's variants (or match one without stress and no other reading's). Kinds: "default" (the reading a
// context-free G2P should give), "context" (another reading - needs the neighboring words), "missing" (no dictionary
// variant). Exits 1 when a kind's score is below its minimum.
//
// node tools/heteronyms.js [-m modulePath] [--min-default 0.9] [--min-context 0] [--show]

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


// Main - load the G2P module, score, print - returns true when a kind is below its minimum
async function main() {
    const args = process.argv.slice(2);
    const option = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
    const modulePath = option('-m', null);
    const {g2pModel} = await import(modulePath ? pathToFileURL(resolve(modulePath)).href : '../lib/models/g2pModel.js');
    setPronouncer(await loadG2p(g2pModel));
    const items = JSON.parse(readFileSync(new URL('../data/nn/heteronyms-test.json', import.meta.url), 'utf8'));
    const {kinds, misses} = scoreHeteronyms(items);
    const minimums = {'default': Number(option('--min-default', '0.9')), 'context': Number(option('--min-context', '0')), 'missing': 0};
    let failed = false;
    for (const [kind, {right, total}] of Object.entries(kinds)) {
        const share = right / total;
        const below = share < minimums[kind];
        failed ||= below;
        const note = below ? ` - below the minimum ${minimums[kind]}` : '';
        process.stdout.write(`${kind.padEnd(8)} ${right}/${total} (${(100 * share).toFixed(1)}%)${note}\n`);
    }
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
