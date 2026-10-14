// Speak text to a WAV file
//
// node tools/say.js [-v male|female] [-r rate] [-s seed] [--room] -o out.wav "text"

import {speak} from '../lib/speak.js';
import {writeWav} from '../lib/wav.js';
import {convolve, roomImpulse} from '../lib/room.js';
import {normalize} from '../lib/klatt.js';


const args = process.argv.slice(2);
let voice = 'male';
let rate = 1;
let seed = 1;
let output = 'out/say.wav';
let room = false;
const words = [];
for (let ix = 0; ix < args.length; ix++) {
    if (args[ix] === '-v') {
        voice = args[++ix];
    } else if (args[ix] === '-r') {
        rate = Number(args[++ix]);
    } else if (args[ix] === '-s') {
        seed = Number(args[++ix]);
    } else if (args[ix] === '--room') {
        room = true;
    } else if (args[ix] === '-o') {
        output = args[++ix];
    } else {
        words.push(args[ix]);
    }
}
const {samples: dry, sampleRate} = speak(words.join(' '), voice, {seed, rate});
const samples = room ? normalize(convolve(dry, roomImpulse(sampleRate)), 0.89) : dry;
writeWav(output, samples, sampleRate);
console.log(`${output} - ${(samples.length / sampleRate).toFixed(2)} s`);
