// Sweep voice overrides and report the long-term spectrum of each against a reference
//
// node tools/sweep.js male ref/daniel.wav '[{"rd":0.8}, {"rd":0.8,"emphasis":0.5}]'

import {ltas} from './ltas.js';
import process from 'node:process';
import {speak} from '../lib/speak.js';
import {writeWav} from '../lib/wav.js';

const [voice, reference, overridesJson] = process.argv.slice(2);
const text = 'The quick brown fox jumps over the lazy dog. Where are you going?';
const show = (name, {bands, db}) => {
    const columns = bands.map(([low], ix) => `${low / 1000}k:${db[ix].toFixed(1).padStart(6)}`);
    process.stdout.write(`${name.padEnd(48)} ${columns.join(' ')}\n`);
};
show(reference, ltas(reference));
for (const overrides of JSON.parse(overridesJson)) {
    const {samples, sampleRate} = speak(text, voice, {overrides});
    writeWav('out/sweep.wav', samples, sampleRate);
    show(JSON.stringify(overrides), ltas('out/sweep.wav'));
}
