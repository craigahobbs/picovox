// Stop consonant test - minimal sets of words differing in a stop's place or voicing, in a carrier phrase, transcribed
// with Whisper and scored by place and voicing of the heard stop
//
// node tools/stops.js [-v male|female|samantha|daniel] [-o overridesJson] [-s seeds] [-t tag] [--show]

import {execFileSync} from 'node:child_process';
import {mkdirSync} from 'node:fs';
import process from 'node:process';
import {speak} from '../lib/speak.js';
import {transcribeAll} from './eval.js';
import {writeWav} from '../lib/wav.js';


// Minimal sets - word-initial stops, then word-final stops
export const initialSets = [
    ['pill', 'bill', 'till', 'dill', 'kill', 'gill'],
    ['pole', 'bowl', 'toll', 'dole', 'coal', 'goal'],
    ['pot', 'tot', 'dot', 'cot', 'got'],
    ['pin', 'bin', 'tin', 'din', 'kin'],
    ['pear', 'bear', 'tear', 'dare', 'care'],
    ['pun', 'bun', 'ton', 'done', 'gun'],
    ['pool', 'tool', 'cool', 'ghoul'],
    ['paw', 'cow', 'gown', 'town', 'down', 'bow'],
    ['peg', 'beg', 'keg', 'deck'],
    ['pane', 'bane', 'cane', 'gain', 'Dane', 'tame']
];
export const finalSets = [
    ['rip', 'rib', 'rid', 'rick', 'rig'],
    ['cap', 'cab', 'cat', 'cad', 'cack'],
    ['mop', 'mob', 'mock', 'mod'],
    ['back', 'bag', 'bat', 'bad'],
    ['tap', 'tab', 'tack', 'tag'],
    ['rot', 'rod', 'rock', 'rob'],
    ['seat', 'seed', 'seek', 'siege'],
    ['coat', 'code', 'coke', 'cope']
];


// The stop of a word (initial or final) by spelling - returns P, B, T, D, K, G or '?'
function stopOf(word, final) {
    const spelled = word.toLowerCase();
    const letters = final ? spelled.replace(/e$/, '').slice(-2) : spelled.slice(0, 2);
    const table = [
        ['th', '?'], ['ph', '?'], ['ck', 'K'], ['gh', 'G'], ['p', 'P'], ['b', 'B'], ['t', 'T'], ['d', 'D'], ['c', 'K'], ['k', 'K'],
        ['q', 'K'], ['g', 'G']
    ];
    if (final) {
        for (const [spelling, stop] of table) {
            if (letters.endsWith(spelling)) {
                return stop;
            }
        }
        return '?';
    }
    for (const [spelling, stop] of table) {
        if (letters.startsWith(spelling)) {
            return stop;
        }
    }
    return '?';
}


// Place and voicing of a stop
const stopPlace = {'P': 'lab', 'B': 'lab', 'T': 'alv', 'D': 'alv', 'K': 'vel', 'G': 'vel'};
const stopVoiced = {'P': false, 'B': true, 'T': false, 'D': true, 'K': false, 'G': true};


// Run the test - returns per-stop results
export async function stopTest(voice, {overrides = {}, seeds = 1, tag = 'stops'} = {}) {
    mkdirSync('out/stops', {'recursive': true});
    const items = [];
    for (let seed = 1; seed <= seeds; seed++) {
        for (const [sets, final] of [[initialSets, false], [finalSets, true]]) {
            for (const set of sets) {
                for (const word of set) {
                    items.push({word, final, seed, 'stop': stopOf(word, final)});
                }
            }
        }
    }
    const paths = items.map((item, ix) => {
        const text = `Now say ${item.word} again.`;
        const path = `out/stops/${tag}-${voice}-${ix}.wav`;
        const systemVoices = {'samantha': 'Samantha', 'daniel': 'Daniel', 'fred': 'Fred'};
        if (voice in systemVoices) {
            execFileSync('say', ['-v', systemVoices[voice], '-o', path, '--data-format=LEI16@22050', text]);
        } else {
            const {samples, sampleRate} = speak(text, voice, {overrides, 'seed': item.seed * 1000 + ix});
            writeWav(path, samples, sampleRate);
        }
        return path;
    });
    const transcripts = await transcribeAll(paths, 'Now say the word again.', 4);
    return items.map((item, ix) => {
        const heard = transcripts[ix].text.toLowerCase().replace(/[^a-z' ]+/g, ' ').trim().split(/\s+/);
        const sayIndex = heard.findIndex((other) => ['say', 'see', 'said', 'safe', 'sit'].includes(other));
        const fallback = heard.length >= 2 ? heard[heard.length - 2] : '?';
        const got = sayIndex >= 0 && sayIndex + 1 < heard.length ? heard[sayIndex + 1] : fallback;
        return {...item, got, 'heardStop': stopOf(got, item.final), 'heard': heard.join(' ')};
    });
}


if (process.argv[1] && process.argv[1].endsWith('stops.js')) {
    const args = process.argv.slice(2);
    let voice = 'male';
    let overrides = {};
    let seeds = 1;
    let tag = 'stops';
    for (let ix = 0; ix < args.length; ix++) {
        if (args[ix] === '-v') {
            voice = args[++ix];
        } else if (args[ix] === '-o') {
            overrides = JSON.parse(args[++ix]);
        } else if (args[ix] === '-s') {
            seeds = Number(args[++ix]);
        } else if (args[ix] === '-t') {
            tag = args[++ix];
        }
    }
    const results = await stopTest(voice, {overrides, seeds, tag});

    // Score - word correct, stop correct, place correct, voicing correct - initial and final, and per stop
    const score = (list) => {
        const total = list.length;
        const pct = (count) => `${(100 * count / Math.max(1, total)).toFixed(0)}%`;
        const voicingCorrect = (result) => result.heardStop in stopVoiced && stopVoiced[result.heardStop] === stopVoiced[result.stop];
        return `word ${pct(list.filter((result) => result.got === result.word).length)} ` +
            `stop ${pct(list.filter((result) => result.heardStop === result.stop).length)} ` +
            `place ${pct(list.filter((result) => stopPlace[result.heardStop] === stopPlace[result.stop]).length)} ` +
            `voicing ${pct(list.filter(voicingCorrect).length)} (n=${total})`;
    };
    process.stdout.write(`${voice} ${JSON.stringify(overrides)}\n`);
    process.stdout.write(`  all     ${score(results)}\n`);
    process.stdout.write(`  initial ${score(results.filter((result) => !result.final))}\n`);
    process.stdout.write(`  final   ${score(results.filter((result) => result.final))}\n`);
    for (const final of [false, true]) {
        const line = ['P', 'B', 'T', 'D', 'K', 'G'].map((stop) => {
            const list = results.filter((result) => result.final === final && result.stop === stop);
            return `${stop}:${list.filter((result) => result.heardStop === stop).length}/${list.length}`;
        }).join(' ');
        // Confusion counts - target>heard
        const confusions = {};
        for (const result of results.filter((other) => other.final === final && other.heardStop !== other.stop)) {
            const key = `${result.stop}>${result.heardStop}`;
            confusions[key] = (confusions[key] || 0) + 1;
        }
        const confusionList = Object.entries(confusions).map(([key, count]) => `${key}:${count}`).join(' ');
        process.stdout.write(`  ${final ? 'final  ' : 'initial'} ${line}  ${confusionList}\n`);
    }
    if (args.includes('--show')) {
        for (const result of results.filter((other) => other.got !== other.word)) {
            process.stdout.write(`    ${result.word}->${result.got} [${result.heard}]\n`);
        }
    }
}
