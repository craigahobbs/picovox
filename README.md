# voice-synth

A from-scratch formant speech synthesizer in JavaScript, aiming for realistic male and female voices - no neural nets and
no recorded speech. Text goes through a rule-based front end (normalization, the CMU Pronouncing Dictionary, Klatt's
duration rules, intonation) to frames of synthesizer parameters every 5 ms, which drive a Klatt-style cascade
synthesizer with an LF glottal source.

It runs under Node (command-line tools and evaluation) and in the browser (Web Audio, synthesis in a Web Worker). There
are no dependencies.


## The browser page

Serve the repository root statically and open it - the root `index.html` redirects to `web/`:

```sh
python3 -m http.server 8000    # or: node tools/serve.js
open http://localhost:8000/
```

Type text, choose the voice, rate, room ambience, and crowd, and Speak, Stop, or Download WAV. Speak streams: each
block of audio plays as soon as it is synthesized. The player keeps the streamed audio, so Play/Pause, skip back 5 s,
and skip ahead 5 s work during and after synthesis, with the position and length shown. Crowd has six different speakers (the chosen voice leads, centered)
say the text together, each with its own pitch, vocal tract length, rate, timing, and stereo position, synthesized in
parallel by one worker each. The worker fetches
`data/cmudict.dict` (3.6 MB) on startup. Text separated by blank lines is synthesized paragraph by paragraph, and each
paragraph plays as soon as it is ready. The page needs HTTP - it does not work from a `file:` URL, since module
workers and fetch require it.

`node tools/browsercheck.js` checks the browser build under Node: it loads every `lib/` module and `web/worker.js` with
`node:` imports forbidden, synthesizes with the worker's functions from a dictionary string, and checks the audio
matches the Node path sample for sample.


## Command-line tools

Node 22.3 or later (the library reaches `node:fs` with `process.getBuiltinModule`, so it has no static Node imports).

```sh
node tools/say.js -v male -o out/hello.wav "Hello, world."      # -v male|female, -r rate, -s seed, --room
node tools/spectrogram.js out/hello.png out/hello.wav           # spectrogram PNG
node tools/eval.js -v female --all --show                       # intelligibility (word error rate) with Whisper
node tools/words.js -v male --show                              # per-consonant word test
```

`tools/eval.js` and `tools/words.js` transcribe with [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
(`whisper-cli`) and the `ggml-base.en.bin` model in `data/whisper/`. Other analysis tools: `tools/ltas.js` (long-term
average spectrum), `tools/slices.js` (band spectra of time ranges), `tools/pitchstats.js`, and `tools/envelope.js`.


## The library

| Module | |
| --- | --- |
| `lib/glottal.js` | LF glottal pulse, parameterized by Rd |
| `lib/klatt.js` | Klatt-style cascade synthesizer - voicing, aspiration, nasal pole/zero, F1-F6 and higher formants, frication |
| `lib/phonemes.js` | ARPAbet tables - vowel formants, consonant loci and couplings, noise spectra |
| `lib/lexicon.js` | Text normalization, the pronouncing dictionary, function words |
| `lib/speak.js` | Voices, durations, stop phases, formant transitions, smoothing, intonation - `speak(text, voice, options)` |
| `lib/room.js` | Optional room ambience (synthetic impulse response) |
| `lib/wav.js` | WAV encoding (`encodeWav`), and file reading and writing under Node |
| `lib/analysis.js` | FFT, spectrogram, LPC formants, pitch tracking |

Under Node, `lib/lexicon.js` loads `data/cmudict.dict` on first use. Elsewhere, give it the dictionary text first:

```js
import {setDictionary} from './lib/lexicon.js';
import {speak} from './lib/speak.js';

setDictionary(await (await fetch('data/cmudict.dict')).text());
const {samples, sampleRate} = speak('Hello, world.', 'female', {'rate': 1, 'seed': 1});
```


## License

The CMU Pronouncing Dictionary is distributed under its own license, in `data/LICENSE.cmudict`.
