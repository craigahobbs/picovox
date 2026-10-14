// Consonant intelligibility test - words in a carrier phrase, transcribed with Whisper, scored per consonant
//
// node tools/words.js [-v male|female|samantha] [-o overridesJson] [--show]

import {execFileSync} from 'node:child_process';
import {mkdirSync} from 'node:fs';
import process from 'node:process';
import {speak} from '../lib/speak.js';
import {transcribe} from './eval.js';
import {writeWav} from '../lib/wav.js';


// Test words by the consonant they test
export const testWords = {
    'P': ['pick', 'pole', 'pain'], 'B': ['bill', 'bowl', 'bake'], 'T': ['tick', 'toe', 'tape'],
    'D': ['dig', 'dough', 'date'], 'K': ['kick', 'cold', 'cake'], 'G': ['give', 'goat', 'gate'],
    'F': ['fill', 'foam', 'fade'], 'V': ['vine', 'vote', 'vase'], 'TH': ['thick', 'thought', 'thin'],
    'S': ['sick', 'soap', 'sale'], 'Z': ['zip', 'zone', 'zero'], 'SH': ['ship', 'show', 'shade'],
    'CH': ['chip', 'chose', 'chain'], 'JH': ['jump', 'joke', 'jail'], 'M': ['mill', 'mole', 'make'],
    'N': ['nip', 'note', 'name'], 'L': ['lip', 'load', 'lake'], 'R': ['rip', 'road', 'rake'],
    'W': ['wit', 'woke', 'wade'], 'Y': ['yes', 'yolk', 'yell'], 'HH': ['hill', 'hope', 'hate'],
    'coda-K': ['back', 'rock', 'week'], 'coda-T': ['bat', 'rot', 'wheat'], 'coda-P': ['cap', 'mop', 'keep'],
    'coda-NG': ['sing', 'long', 'rang'], 'coda-N': ['sin', 'lawn', 'ran'], 'coda-M': ['sum', 'calm', 'ram'],
    'cluster-L': ['glue', 'blue', 'clay', 'play', 'slid', 'flame'], 'cluster-R': ['tree', 'drip', 'green', 'crate', 'brave', 'spring']
};


// Run the test - returns per-consonant accuracy and the misses
export function wordTest(voice, {overrides = {}} = {}) {
    mkdirSync('out/words', {'recursive': true});
    const results = {};
    const misses = [];
    const entries = Object.entries(testWords).flatMap(([key, list]) => list.map((word) => [key, word]));
    entries.forEach(([key, word], ix) => {
        const text = `Now say ${word} again.`;
        const path = `out/words/${voice}-${ix}.wav`;
        const systemVoices = {'samantha': 'Samantha', 'fred': 'Fred', 'kathy': 'Kathy'};
        if (voice in systemVoices) {
            execFileSync('say', ['-v', systemVoices[voice], '-o', path, '--data-format=LEI16@22050', text]);
        } else {
            const {samples, sampleRate} = speak(text, voice, {overrides, 'seed': ix + 1});
            writeWav(path, samples, sampleRate);
        }
        const heard = transcribe(path, 'Now say the word again.').text.toLowerCase().replace(/[^a-z' ]+/g, ' ').trim().split(/\s+/);
        // The tested word is the one after "say" (or the second-to-last word, before "again")
        const sayIndex = heard.findIndex((other) => ['say', 'see', 'said', 'safe', 'sit'].includes(other));
        const fallback = heard.length >= 2 ? heard[heard.length - 2] : '?';
        const got = sayIndex >= 0 && sayIndex + 1 < heard.length ? heard[sayIndex + 1] : fallback;
        results[key] ||= {'correct': 0, 'total': 0};
        results[key].total++;
        if (got === word) {
            results[key].correct++;
        } else {
            misses.push(`${word}->${got} [${heard.join(' ')}]`);
        }
    });
    return {results, misses};
}


if (process.argv[1] && process.argv[1].endsWith('words.js')) {
    const args = process.argv.slice(2);
    let voice = 'male';
    let overrides = {};
    for (let ix = 0; ix < args.length; ix++) {
        if (args[ix] === '-v') {
            voice = args[++ix];
        } else if (args[ix] === '-o') {
            overrides = JSON.parse(args[++ix]);
        }
    }
    const {results, misses} = wordTest(voice, {overrides});
    let correct = 0;
    let total = 0;
    const line = Object.entries(results).map(([key, value]) => {
        correct += value.correct;
        total += value.total;
        return `${key}:${value.correct}/${value.total}`;
    }).join(' ');
    process.stdout.write(`${voice} - ${(100 * correct / total).toFixed(0)}% words correct\n`);
    process.stdout.write(`${line}\n`);
    if (args.includes('--show')) {
        process.stdout.write(`${misses.join('\n')}\n`);
    }
}
