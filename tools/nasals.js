// Nasal, /h/, and utterance-onset test - target words in carrier phrases and sentence-initial "A", transcribed with
// Whisper, scored per group
//
// node tools/nasals.js [-v male|female|samantha|daniel] [-o overridesJson] [-t tag] [--show]

import {execFileSync} from 'node:child_process';
import {mkdirSync} from 'node:fs';
import process from 'node:process';
import {speak} from '../lib/speak.js';
import {transcribeAll} from './eval.js';
import {writeWav} from '../lib/wav.js';


// Test words in the "Now say X again." carrier, by group
export const nasalWords = {
    'initial-M': ['mat', 'mice', 'moon', 'meal', 'mop', 'mug'],
    'initial-N': ['nat', 'nice', 'noon', 'kneel', 'knot', 'nut'],
    'coda-M': ['bomb', 'ram', 'sum', 'team', 'dim', 'comb'],
    'coda-N': ['bond', 'ran', 'sun', 'teen', 'din', 'cone', 'dawn'],
    'coda-NG': ['sang', 'rang', 'sung', 'tongue', 'ding', 'bang'],
    'cluster': ['planks', 'plants', 'bonds', 'bombs', 'sand', 'sank', 'hump', 'hunt'],
    'HH': ['hill', 'hogs', 'hat', 'hope', 'heat', 'hall', 'home', 'hum', 'hen', 'hang']
};


// Accepted spellings of homophones
const spellings = {'mat': ['matt'], 'nat': ['gnat', 'nat'], 'kneel': ['neil', 'neal'], 'sum': ['some'], 'sun': ['son'], 'dawn': ['don'],
    'hen': ['henn'], 'knot': ['not'], 'din': ['dinn']};


// Sentence-initial "A" (and "The" controls) - the first word scored
export const onsetSentences = [
    'A large size in stockings is hard to sell.', 'A rod is used to catch pink salmon.', 'A pot of tea helps to pass the evening.',
    'A king ruled the state in the early days.', 'A green idea sleeps under the loud chair.', 'A sudden river reads the empty shoe.',
    'A man sat down by the door.', 'A dog ran home in the rain.', 'A bird sang in the tall tree.', 'A ship sailed into the bay.',
    'An apple fell from the tree.', 'A small boat drifted past.',
    'The man sat down by the door.', 'The dog ran home in the rain.'
];


// Normalize transcribed text to words
function words(text) {
    return text.toLowerCase().replace(/[^a-z' ]+/g, ' ').replace(/'/g, '').trim().split(/\s+/);
}


// Run the test - returns per-group accuracy and the misses
export async function nasalTest(voice, {overrides = {}, tag = 'nasals'} = {}) {
    mkdirSync('out/nasals', {'recursive': true});
    const systemVoices = {'samantha': 'Samantha', 'daniel': 'Daniel', 'fred': 'Fred'};
    const items = [
        ...Object.entries(nasalWords).flatMap(([group, list]) => list.map((word) => ({group, word, 'text': `Now say ${word} again.`}))),
        ...onsetSentences.map((text) => ({'group': text.startsWith('The') ? 'onset-The' : 'onset-A', 'word': words(text)[0], text}))
    ];
    const paths = items.map((item, ix) => {
        const path = `out/nasals/${tag}-${voice}-${ix}.wav`;
        if (voice in systemVoices) {
            execFileSync('say', ['-v', systemVoices[voice], '-o', path, '--data-format=LEI16@22050', item.text]);
        } else {
            const {samples, sampleRate} = speak(item.text, voice, {overrides, 'seed': ix + 1});
            writeWav(path, samples, sampleRate);
        }
        return path;
    });
    const transcripts = await transcribeAll(paths);
    const results = {};
    const misses = [];
    items.forEach((item, ix) => {
        const heard = words(transcripts[ix].text);
        let got;
        if (item.group.startsWith('onset-')) {
            [got] = heard;
        } else {
            const sayIndex = heard.findIndex((other) => ['say', 'see', 'said', 'safe', 'sit'].includes(other));
            got = sayIndex >= 0 && sayIndex + 1 < heard.length ? heard[sayIndex + 1] : (heard.length >= 2 ? heard[heard.length - 2] : '?');
        }
        results[item.group] ||= {'correct': 0, 'total': 0};
        results[item.group].total++;
        if (got === item.word || (spellings[item.word] || []).includes(got)) {
            results[item.group].correct++;
        } else {
            misses.push(`${item.word}->${got} [${heard.join(' ')}]`);
        }
    });
    return {results, misses};
}


if (process.argv[1] && process.argv[1].endsWith('nasals.js')) {
    const args = process.argv.slice(2);
    let voice = 'male';
    let overrides = {};
    let tag = 'nasals';
    for (let ix = 0; ix < args.length; ix++) {
        if (args[ix] === '-v') {
            voice = args[++ix];
        } else if (args[ix] === '-o') {
            overrides = JSON.parse(args[++ix]);
        } else if (args[ix] === '-t') {
            tag = args[++ix];
        }
    }
    const {results, misses} = await nasalTest(voice, {overrides, tag});
    let correct = 0;
    let total = 0;
    const line = Object.entries(results).map(([key, value]) => {
        correct += value.correct;
        total += value.total;
        return `${key}:${value.correct}/${value.total}`;
    }).join(' ');
    process.stdout.write(`${voice} ${JSON.stringify(overrides)} - ${(100 * correct / total).toFixed(0)}% correct\n`);
    process.stdout.write(`${line}\n`);
    if (args.includes('--show')) {
        process.stdout.write(`${misses.join('\n')}\n`);
    }
}
