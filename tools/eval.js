// Intelligibility evaluation - synthesize test sentences, transcribe them with Whisper (whisper.cpp), and report the
// word error rate and Whisper's mean token probability (a rough proxy for how speech-like the audio is)
//
// node tools/eval.js [-v male|female|samantha|daniel] [-o overridesJson] [-n count] [--sus] [--show]

import {execFile, execFileSync} from 'node:child_process';
import {promisify} from 'node:util';
import {mkdirSync, readFileSync} from 'node:fs';
import {speak} from '../lib/speak.js';
import {writeWav} from '../lib/wav.js';


// Harvard sentences (lists 1 and 2) - phonetically balanced
export const harvard = [
    'The birch canoe slid on the smooth planks.',
    'Glue the sheet to the dark blue background.',
    'It\'s easy to tell the depth of a well.',
    'These days a chicken leg is a rare dish.',
    'Rice is often served in round bowls.',
    'The juice of lemons makes fine punch.',
    'The box was thrown beside the parked truck.',
    'The hogs were fed chopped corn and garbage.',
    'Four hours of steady work faced us.',
    'A large size in stockings is hard to sell.',
    'The boy was there when the sun rose.',
    'A rod is used to catch pink salmon.',
    'The source of the huge river is the clear spring.',
    'Kick the ball straight and follow through.',
    'Help the woman get back to her feet.',
    'A pot of tea helps to pass the evening.',
    'Smoky fires lack flame and heat.',
    'The soft cushion broke the man\'s fall.',
    'The salt breeze came across from the sea.',
    'The girl at the booth sold fifty bonds.',
    'The small pup gnawed a hole in the sock.',
    'The fish twisted and turned on the bent hook.',
    'Press the pants and sew a button on the vest.',
    'The swan dive was far short of perfect.',
    'The beauty of the view stunned the young boy.',
    'Two blue fish swam in the tank.',
    'Her purse was full of useless trash.',
    'The colt reared and threw the tall rider.',
    'It snowed, rained, and hailed the same morning.',
    'Read verse out loud for pleasure.',
    'Hoist the load to your left shoulder.',
    'Take the winding path to reach the lake.',
    'Note closely the size of the gas tank.',
    'Wipe the grease off his dirty face.',
    'Mend the coat before you go out.',
    'The wrist was badly strained and hung limp.',
    'The stray cat gave birth to kittens.',
    'The young girl gave no clear response.',
    'The meal was cooked before the bell rang.',
    'What joy there is in living.',
    'A king ruled the state in the early days.',
    'The ship was torn apart on the sharp reef.',
    'Sickness kept him home the third week.',
    'The wide road shimmered in the hot sun.',
    'The lazy cow lay in the cool grass.',
    'Lift the square stone over the fence.',
    'The rope will bind the seven books at once.',
    'Hop over the fence and plunge in.',
    'The friendly gang left the drug store.',
    'Mesh wire keeps chicks inside.'
];


// Semantically unpredictable sentences - no context for the recognizer to lean on
export const unpredictable = [
    'The table walked through the blue truth.',
    'A green idea sleeps under the loud chair.',
    'The strong day drank the narrow fish.',
    'Why does the window eat the silver song?',
    'Never paint the sharp rain across the field.',
    'The cold pencil sang beside a quiet hill.',
    'Did the brave lamp jump over the thin cake?',
    'The black voice found a soft engine.',
    'A sudden river reads the empty shoe.',
    'Push the late garden into the warm cloud.'
];


// Normalize text to words for scoring
function words(text) {
    return text.toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').replace(/'/g, '').split(/\s+/).filter((word) => word !== '');
}


// Word-level edit distance
function editDistance(reference, hypothesis) {
    const rows = reference.length + 1;
    const cols = hypothesis.length + 1;
    const distance = Array.from({'length': rows}, (_, ix) => Array.from({'length': cols}, (__, jx) => (ix === 0 ? jx : (jx === 0 ? ix : 0))));
    for (let ix = 1; ix < rows; ix++) {
        for (let jx = 1; jx < cols; jx++) {
            const cost = reference[ix - 1] === hypothesis[jx - 1] ? 0 : 1;
            distance[ix][jx] = Math.min(distance[ix - 1][jx] + 1, distance[ix][jx - 1] + 1, distance[ix - 1][jx - 1] + cost);
        }
    }
    return distance[rows - 1][cols - 1];
}


// Parse a whisper.cpp JSON result - returns {text, probability (mean token probability)}
function parseResult(base) {
    const result = JSON.parse(readFileSync(`${base}.json`, 'utf8'));
    let text = '';
    let probabilitySum = 0;
    let tokenCount = 0;
    for (const segment of result.transcription) {
        text += segment.text;
        for (const token of segment.tokens) {
            if (!token.text.startsWith('[_')) {
                probabilitySum += token.p;
                tokenCount++;
            }
        }
    }
    return {'text': text.trim(), 'probability': tokenCount === 0 ? 0 : probabilitySum / tokenCount};
}


// Whisper arguments for a WAV file (with an optional context prompt)
function whisperArgs(path, prompt) {
    const promptArgs = prompt === null ? [] : ['--prompt', prompt];
    return ['-m', 'data/whisper/ggml-base.en.bin', '-t', '2', '-f', path, '-np', '-ojf', '-of', path.replace(/\.wav$/, ''), ...promptArgs];
}


// Transcribe a WAV file (with an optional context prompt) - returns {text, probability (mean token probability)}
export function transcribe(path, prompt = null) {
    execFileSync('whisper-cli', whisperArgs(path, prompt), {'stdio': 'ignore'});
    return parseResult(path.replace(/\.wav$/, ''));
}


// Transcribe WAV files in parallel - returns [{text, probability}]
export async function transcribeAll(paths, prompt = null, concurrency = 3) {
    const run = promisify(execFile);
    const results = new Array(paths.length);
    let next = 0;
    const worker = async () => {
        while (next < paths.length) {
            const ix = next++;
            await run('whisper-cli', whisperArgs(paths[ix], prompt));
            results[ix] = parseResult(paths[ix].replace(/\.wav$/, ''));
        }
    };
    await Promise.all(Array.from({'length': concurrency}, worker));
    return results;
}


// Evaluate a voice on sentences - returns {wer, probability, results}
export async function evaluate(voice, sentences, {overrides = {}, tag = 'eval'} = {}) {
    mkdirSync('out/eval', {'recursive': true});
    let errors = 0;
    let total = 0;
    let probabilitySum = 0;
    const paths = sentences.map((sentence, ix) => {
        const path = `out/eval/${tag}-${voice}-${ix}.wav`;
        const systemVoices = {'samantha': 'Samantha', 'daniel': 'Daniel', 'fred': 'Fred', 'kathy': 'Kathy', 'ralph': 'Ralph'};
        if (voice in systemVoices) {
            execFileSync('say', ['-v', systemVoices[voice], '-o', path, '--data-format=LEI16@22050', sentence]);
        } else {
            const {samples, sampleRate} = speak(sentence, voice, {overrides, 'seed': ix + 1});
            writeWav(path, samples, sampleRate);
        }
        return path;
    });
    const transcripts = await transcribeAll(paths);
    const results = sentences.map((sentence, ix) => {
        const {text, probability} = transcripts[ix];
        const reference = words(sentence);
        const sentenceErrors = editDistance(reference, words(text));
        errors += sentenceErrors;
        total += reference.length;
        probabilitySum += probability;
        return {sentence, text, 'errors': sentenceErrors, probability};
    });
    return {'wer': errors / total, 'probability': probabilitySum / sentences.length, results};
}


if (process.argv[1] && process.argv[1].endsWith('eval.js')) {
    const args = process.argv.slice(2);
    let voice = 'male';
    let overrides = {};
    let count = 1000;
    let sentences = harvard;
    let show = false;
    for (let ix = 0; ix < args.length; ix++) {
        if (args[ix] === '-v') {
            voice = args[++ix];
        } else if (args[ix] === '-o') {
            overrides = JSON.parse(args[++ix]);
        } else if (args[ix] === '-n') {
            count = Number(args[++ix]);
        } else if (args[ix] === '--sus') {
            sentences = unpredictable;
        } else if (args[ix] === '--all') {
            sentences = [...harvard, ...unpredictable];
        } else if (args[ix] === '--show') {
            show = true;
        }
    }
    const {wer, probability, results} = await evaluate(voice, sentences.slice(0, count), {overrides});
    if (show) {
        for (const result of results) {
            if (result.errors > 0) {
                console.log(`  ${result.errors} | ${result.sentence} | ${result.text}`);
            }
        }
    }
    console.log(`${voice} ${JSON.stringify(overrides)} - WER ${(100 * wer).toFixed(1)}%, mean token probability ${probability.toFixed(3)}`);
}
