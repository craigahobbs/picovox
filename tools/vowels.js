// Experiment 1 - sustained vowel glides for a male and a female voice, to check the source and the cascade

import {normalize, synthesize} from '../lib/klatt.js';
import {decimate} from '../lib/analysis.js';
import process from 'node:process';
import {renderSpectrograms} from './spectrogram.js';
import {writeWav} from '../lib/wav.js';


// Hillenbrand et al. (1995) average vowel formants (F1, F2, F3) for men and women
const vowels = {
    'men': {'iy': [342, 2322, 3000], 'aa': [768, 1333, 2522], 'uw': [378, 997, 2343], 'ae': [588, 1952, 2601], 'ow': [497, 910, 2459]},
    'women': {'iy': [437, 2761, 3372], 'aa': [936, 1551, 2815], 'uw': [459, 1105, 2735], 'ae': [669, 2349, 2972], 'ow': [555, 1035, 2828]}
};


function vowelFrames(table, sequence, {f0, rd, f4, f5, f6}) {
    const frames = [];
    const frameMs = 5;
    const vowelMs = 600;
    const framesPerVowel = vowelMs / frameMs;
    const total = framesPerVowel * sequence.length;
    for (let ix = 0; ix <= total; ix++) {
        const position = Math.min(sequence.length - 1, ix / framesPerVowel);
        const index = Math.floor(position);
        const next = Math.min(sequence.length - 1, index + 1);
        // Hold each vowel, gliding to the next over its last third
        const glide = Math.max(0, (position - index - 0.66) / 0.34);
        const smooth = glide * glide * (3 - 2 * glide);
        const [f1a, f2a, f3a] = table[sequence[index]];
        const [f1b, f2b, f3b] = table[sequence[next]];
        const time = ix * frameMs / 1000;
        // A gently falling pitch with onset and offset envelopes
        const envelope = Math.min(1, time / 0.06, (total - ix) * frameMs / 1000 / 0.12);
        frames.push({
            'f0': f0 * (1.1 - 0.15 * ix / total),
            'av': envelope, 'ah': 0.08 * envelope, 'af': 0, rd,
            'f1': f1a + (f1b - f1a) * smooth, 'f2': f2a + (f2b - f2a) * smooth, 'f3': f3a + (f3b - f3a) * smooth,
            f4, f5, f6, 'b1': 80, 'b2': 90, 'b3': 150, 'b4': 250, 'b5': 300, 'b6': 1000,
            'fnp': 250, 'bnp': 100, 'fnz': 250, 'bnz': 100,
            'a2': 0, 'a3': 0, 'a4': 0, 'a5': 0, 'a6': 0, 'ab': 0
        });
    }
    return frames;
}


const sequence = ['aa', 'iy', 'uw', 'ae', 'ow'];
const voices = [
    {'name': 'male', 'table': vowels.men, 'f0': 115, 'rd': 1.0, 'f4': 3500, 'f5': 4500, 'f6': 5500},
    {'name': 'female', 'table': vowels.women, 'f0': 210, 'rd': 1.4, 'f4': 4200, 'f5': 5000, 'f6': 6000}
];
const paths = [];
for (const voice of voices) {
    const samples = synthesize(vowelFrames(voice.table, sequence, voice), {'seed': 7});
    const path = `out/exp1-vowels-${voice.name}.wav`;
    writeWav(path, normalize(decimate(samples, 2)), 22050);
    paths.push(path);
}
renderSpectrograms('out/exp1-vowels.png', paths);
process.stdout.write(`${paths.join('\n')}\n`);
