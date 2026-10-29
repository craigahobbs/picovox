# PicoVox

[![npm](https://img.shields.io/npm/v/picovox)](https://www.npmjs.com/package/picovox)
[![GitHub](https://img.shields.io/github/license/craigahobbs/picovox)](https://github.com/craigahobbs/picovox/blob/main/LICENSE)

**PicoVox - realistic speech synthesis in pure JavaScript, under 1 MB.**

**The paper:** [PicoVox: Realistic Neural Speech Synthesis in Under 1 MB of JavaScript](https://craigahobbs.github.io/picovox/paper/) -
how PicoVox's networks are designed, trained, distilled, and evaluated.

PicoVox speaks English text with twelve natural-sounding male and female voices, entirely on your device - in the
browser or in Node. There is no recorded speech, no server, no WebAssembly, and no runtime dependencies: a small
grapheme-to-phoneme network pronounces every word, a small neural network (283K parameters) predicts interpretable
source-filter synthesizer parameters every 5 ms, and a JavaScript synthesizer with an LF glottal source renders them.
The whole package - code and both networks - is under 1 MB.

[PicoShelf](https://craigahobbs.github.io/picoshelf/), every public-domain book read aloud right in your browser, is
built on PicoVox.


## Installation

```sh
npm install picovox
```

PicoVox is ESM-only. It runs in current browsers (Chrome, Edge, Firefox, and Safari 16.4 or later) and in Node 22 or
later.


## Quick start

In the browser - speak text aloud with Web Audio (call it from a click, so the browser lets the audio start):

```html
<button id="speak">Speak</button>
<script type="module">
    import {speak} from './node_modules/picovox/lib/index.js';

    document.getElementById('speak').onclick = () => {
        speak('It was the best of times, it was the worst of times.', {'voice': 'female'});
    };
</script>
```

(With a bundler or an import map, import from `'picovox'`.)

In Node - synthesize speech and write a WAV file:

```js
import {synthesize, toWav} from 'picovox';
import {writeFileSync} from 'node:fs';

const {samples, sampleRate} = await synthesize('Call me Ishmael.', {'voice': 'male'});
writeFileSync('ishmael.wav', toWav(samples, sampleRate));
```

Streaming - a sentence and a fraction of a second at a time, the first speech in about 70 ms (M3 Max):

```js
import {stream} from 'picovox';

for await (const block of stream(longText, {'voice': 'female', 'rate': 1.2})) {
    // block is a Float32Array of 24 kHz samples - play it, or post it from a Web Worker
}
```


## API

### `synthesize(text, options)`

Synthesize speech. Returns a Promise of `{samples, sampleRate}` - `samples` is a `Float32Array` of mono samples (-1 to
1) at `sampleRate` Hz.

### `stream(text, options)`

Synthesize speech as a stream - an async iterable of `Float32Array` blocks of samples, made as they're taken. The text
is spoken a sentence at a time, each sentence a fraction of a second at a time, so the first audio comes almost at once.
The blocks, concatenated, are exactly `synthesize`'s samples. Synthesis runs on the calling thread as each block is
taken - in a browser, take the blocks in a Web Worker to keep the page responsive.

### `speak(text, options)`

Speak text aloud in a browser, with Web Audio. Returns a playback, `{stop(), finished}` - `stop()` stops it, and
`finished` is a Promise that resolves when the playback ends or is stopped (and rejects if synthesis fails). Playback
starts with the first blocks while the rest are synthesized, a few seconds ahead. Synthesis runs on the main thread, a
block at a time between the page's other work; for long texts, use `stream` in a Web Worker and play its blocks.
`options.context` is an `AudioContext` to play in (by default, a new one at the sample rate, closed when the playback
ends). Play blocks in a context at their sample rate (`new AudioContext({'sampleRate': 24000})`): a context at another
rate resamples each block apart, and the blocks' joins click. Under Node,
`speak` throws an error - there is no Web Audio; use `synthesize` and `toWav`.

### `voices()`

The built-in voices - `[{key, label, sex, description}]`, the default first:

| Key | Label | Sex | Character |
| --- | --- | --- | --- |
| `male` (the default) | Male | male | A warm, buoyant male voice over a resonant mid-low core - the default. |
| `female` | Female | female | A modern, confident, and friendly female voice. |
| `sarah` | Sarah | female | A cheerful female voice with real resonance under it, personable and decisive. |
| `helen` | Helen | female | A low, grounded female voice, quick to the point, with a sincere tone. |
| `amy` | Amy | female | A young, playful, and upbeat female voice. |
| `kate` | Kate | female | A settled, even-toned, and kind female voice with crisp consonants. |
| `grace` | Grace | female | A rounded, soft-edged female voice with an easy, relaxed delivery. |
| `james` | James | male | A resonant, decisive male voice with a wide dynamic range. |
| `robert` | Robert | male | A trim, focused male voice with a steady, low-set core. |
| `michael` | Michael | male | A buoyant, energetic male voice with a playful lift at the ends of phrases. |
| `david` | David | male | A punchy, close male voice with crisp consonants and a pitch that lifts across each line. |
| `thomas` | Thomas | male | A high-energy, encouraging male voice. |

Each voice is one row of the voice model's speaker table, learned from one voice of the local TTS model that speaks
make train's training speech (`train/data/train.json`'s `tts` voices - see [the training speech](#the-training-speech));
`test/voices.json` is their design record. The voices are the model's speakers as it speaks them - no blends, and no
pitch, tract, or rate adjustments of their own - except that every voice's source is 5 dB darker than the network
predicts it (`lib/synthesis.js` `sourceShelf`): in blind A/B listening (1.6.6) the network's source sounded tinny, and
darker it was less tinny and never less natural. On the listener panel's texts the voices score TODO(results).

### `styles()`

The speaking styles - `[{key, label, kind, description}]`, neutral (the default) first. Any style goes with any voice:

| Key | Kind | |
| --- | --- | --- |
| `neutral` (the default) | emotion | The voice as designed - plain narration |
| `angry` | emotion | A pressed, raised, louder voice |
| `happy` | emotion | A brighter, much higher voice |
| `sad` | emotion | A quieter, lower, flatter, and slower voice |
| `fearful` | emotion | A breathy, quieter, much higher, and quicker voice |
| `disgusted` | emotion | A pressed, quieter, lower, and much slower voice |
| `whisper` | mode | No voicing - the breath only |
| `soft` | mode | A quieter voice, a little slower |
| `shouting` | mode | A pressed, bright, raised voice, a little louder |
| `dialogue` | mode | Neutral dialogue - a little higher, brighter, and quicker than the narration |
| `exclaim` | mode | An exclamation - the dialogue style livelier: louder, wider in pitch, and quicker |

Every style transforms the synthesizer's parameters (`lib/transforms.js` - a pressed or relaxed glottal pulse, pitch and
pitch range, whisper, level, frication) and the pace; the voice model has no style input. The emotions' transforms are
measured: before 1.7 the voice model had emotion embeddings learned from CREMA-D's actors, and happy, sad, fearful, and
disgusted are those embeddings' effects on the parameter frames (pitch, pitch range, pulse, voicing, frication, level,
and duration, over the panel texts in every voice), less the breathiness they added; angry was transforms alone already.
The styles' listener-panel measurements (Whisper's word error rate, UTMOS, and loudness against neutral) were of the
former model and its embeddings; they will be re-measured for the new voices - TODO(results).

### Expressive reading: `style: 'auto'`

Style `'auto'` reads a text as a narrator does - its delivery follows the text's structure:

- **The dialogue beat.** The quotations (between double quotation marks) are read in the `dialogue` style - a little
  higher, brighter, and quicker than the narration (pitch +7%, pace +5%) - and the narration neutral.
- **The exclamation beat.** A quoted sentence ending in "!" is read in the `exclaim` style - livelier than the
  dialogue beat: 3 dB louder, its pitch range 20% wider, a little more pressed and quicker - excited, not shouted.
- **The paragraph beat.** Each paragraph is read as one arc: its first sentence lifted (pitch +8%, 2 dB louder), the
  pitch settling to 4% below the voice's by its last sentence, which is 7% slower.

Style `'auto'` needs no model - it reads the text's quotation marks, exclamation marks, and paragraphs. (A small network that would read a
quotation angry, shouting, or fearful was tried and dropped: even a classifier on a 4-billion-parameter embedding model
couldn't choose those styles reliably enough - see the paper's Section 12.7.)

On the listener panel (the voices before 1.7) the beats kept the voices intelligible and natural: the dialogue style's
UTMOS was within 0.04 of neutral's in four voices, its word error rate unchanged; the exclaim style's, read over whole
texts in all twelve voices, within 0.13 of neutral's, its word error rate at most 2.1% - for the new voices,
TODO(results). The output is deterministic, as every style's; without `'auto'` the audio is the same as before, bit for
bit.

### `toWav(samples, sampleRate)`

Encode samples (a `Float32Array`, or an array of numbers, -1 to 1 - values beyond are clipped) as a 16-bit PCM mono WAV
file - a `Uint8Array` of the file's bytes.

### Options

`synthesize`, `stream`, and `speak` take the same options:

| Option | Default | |
| --- | --- | --- |
| `voice` | `'male'` | The voice's key - see `voices()` |
| `style` | `'neutral'` | The speaking style's key - see `styles()` - or `'auto'`, expressive reading (below) |
| `rate` | `1` | The speaking speed, from 0.5 to 2 (1.2 is 20% faster) |
| `seed` | `1000` | The noise seed (an integer) - sentence *i* of the text is rendered with seed + *i* |
| `sampleRate` | `24000` | The output sample rate (Hz), from 8000 to 96000 - 24 kHz is the voices' own rate |

The text is split into sentences (and paragraphs, at blank lines); each sentence starts with a short pause - 0.24
seconds, or 0.4 before a paragraph (so the text's first sentence starts with one). Numbers (four-digit numbers as years,
1972 as "nineteen seventy-two"; ordinals, 1st; decades, 1970s; percentages; money, $1.50 and $5 million; times, 10:30
a.m.), common abbreviations, dashes, ellipses, and Roman numerals (Chapter IV, World War II) are spoken as words, and
an aside's parentheses pause as a comma does. An invalid text or option throws a `TypeError` or a
`RangeError` with a message saying what's wrong (`stream` and `speak` throw at once; `synthesize` rejects).

The output is bit-for-bit deterministic: the same text and options give the same samples, in the browser and in Node.

### The modules

The package's modules are importable too (`picovox/lib/<module>.js`) - the pieces the API is built from, for an app
that runs its own synthesis loop (PicoShelf's worker does):

| Module | |
| --- | --- |
| `lib/index.js` | The API |
| `lib/synthesis.js` | The pipeline - the voices, a sentence's blocks (`sentenceBlocks`), a text's (`textBlocks`), the gain and limiter |
| `lib/sentences.js` | Text to sentences (`splitSentences`), their quotations (`quoteSpans`), and their speakable text (`speechText`) |
| `lib/style.js` | Style `'auto'` - a text's quotations in the dialogue style (`autoStyles`) and its paragraph beats (`paragraphBeats`) |
| `lib/lexicon.js` | Normalization and phrasing (`textToPhrases`), every word pronounced by the pronouncer (`setPronouncer`) |
| `lib/g2p.js` | The grapheme-to-phoneme network's inference (`loadG2p`) - heteronyms read from their context |
| `lib/tokens.js` | Phrases to the voice network's tokens |
| `lib/model.js` | The voice network - text to parameter frames (`speakFrames`, `streamFrames`) in a speaker's voice |
| `lib/render.js` | The source-filter synthesizer - parameter frames to samples, streaming (`createRenderer`, `render`) |
| `lib/lf.js`, `lib/rng.js`, `lib/fft.js`, `lib/params.js` | The LF glottal pulse, the counter-based noise, the FFT, the parameter set |
| `lib/transforms.js` | Voice and style transforms of parameter frames (pitch, tract length, bandwidths, whisper, effort, pulse softness, source shelf) |
| `lib/package.js` | The generated model modules' loader (`loadPackagedModel`) |
| `lib/wav.js` | WAV encoding (`encodeWav`) |
| `lib/models/g2pModel.js`, `lib/models/voiceModel.js` | The generated models (`make train`) |


## Size and speed

| | |
| --- | --- |
| The package (`npm pack`) | TODO(results) packed, TODO(results) unpacked - the G2P network 419 KB and the voice network TODO(results) of it, as base64 |
| The voice network | 282,627 parameters, 8-bit weights - TODO(results) gzipped |
| The G2P network | 373,256 parameters (and the heteronym context network's 47,779), 6-bit weights |
| Speed, Apple M3 Max (Node) | The voice network about 127x real time, the renderer about 97x; synthesis end to end (24 kHz) about 55x |
| Speed, iPhone without a JIT (Safari over plain http, or in Lockdown Mode) | About 2.6x real time in one worker; about 3.1-3.4x with the network and the renderer in two workers (`sentenceFrames` and `createSentenceRenderer`) |
| Startup | About 0.1 s to load both networks (Node, M3 Max) |

The voices score a mean UTMOS (a neural predictor of listeners' naturalness ratings, 1-5) of TODO(results) on the
listener panel's texts, with a Whisper word error rate of TODO(results) - see [the listener panel](#the-listener-panel).


## How the models are trained

Both networks are trained from scratch by `make train`, which rebuilds `lib/models/g2pModel.js` and
`lib/models/voiceModel.js` from a clean checkout. The voice network learns from training speech that make train
generates: Project Gutenberg sentences spoken by a local TTS model (Gradium's Phonon, through Ollama - one of its voices
for each of the package's twelve), aligned and checked with our own phoneme recognizer, then a large teacher model
trained through a differentiable copy of the synthesizer, distilled to the small student that ships. The G2P network
learns from the CMU Pronouncing Dictionary (downloaded, pinned; it is not used at runtime), its heteronyms' readings
measured in the training speech. The whole run takes TODO(results) on an M3 Max (the training speech generated once,
then cached). See [make train](#make-train) below, and the paper,
[PicoVox: Realistic Neural Speech Synthesis in Under 1 MB of JavaScript](paper/paper.md), for the methods and results.


## PicoShelf

[PicoShelf](https://craigahobbs.github.io/picoshelf/) is the showcase app - an audiobook player for every public-domain
book (Project Gutenberg and Standard Ebooks), read aloud by PicoVox in the browser, in a Web Worker
([source](https://github.com/craigahobbs/picoshelf)).


## Development

This repository is the package (`lib/`, `test/`) and the training pipeline that makes its models (`train/`,
`tools/`, the paper in `paper/`). The package is developed with
[javascript-build](https://github.com/craigahobbs/javascript-build#readme), and the training's Python code with
[python-build](https://github.com/craigahobbs/python-build#readme):

```sh
make commit      # the package's unit tests (100% coverage), lint, and API documentation (jsdoc) - and the training's gate
make test        # the package's unit tests (TEST='name' runs one)
make -C train commit      # the training's gate alone - its unit tests, pylint, and 100% branch coverage
make -C train test TEST=tests.test_params    # one test module of the training's
make -j train             # train both models - the G2P and the voice - each gated, in build/train/, then a report (see below)
make train-check          # the package's tests with the trained models, then the blind A/B of the trained voice
make train-install        # the trained models that differ into lib/models/ (train-install-voice, -g2p: one model)
make -j train-voice       # the voice alone; make -j train-g2p - the G2P alone, on every core
```

The training pipeline is a Python package, `picovox_train` (`train/src/picovox_train/` - `picovox_train/<module>.py`
below), and Node scripts (`tools/`); they use the package's modules but aren't part of it. `train/` is a python-build
package of its own: its unit tests (`train/src/tests/`) run on the CPU in a few minutes, with tiny synthetic data and no
network, in python-build's environment (`train/build/venv/`), which installs the package with the training's pinned
PyTorch and numpy (`train/pyproject.toml`, the same versions as `train/requirements.txt`). `make train` runs the package
from its source (`PYTHONPATH=train/src`) in its environment, `build/venv-train` from `train/requirements.txt` - the
training (PyTorch, soxr, Praat, and the phoneme recognizer's transformers), the listener models, and spaCy
(`picovox_train/corpora.py`'s downloads need only the standard library). A training module runs the same way by hand:

```sh
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.smoke   # every training entry point, a few steps on the CPU
```


## Command-line tools

```sh
node tools/unpack.js lib/models/voiceModel.js build/voice                 # a packaged model back to its export
node tools/speak.js -m build/voice "Hello, world." -o build/hello.wav     # an exported model (<voice>.json and .bin)
node tools/packagecheck.js lib/models/voiceModel.js build/voice           # check a packaged model against its export
node tools/heteronyms.js                                                # the heteronym and names regression sets
```

The performance lab, `tools/perf/`, times the playback pipeline on a phone: `node tools/perf/server.js` serves the
repository on the local network (open the printed URL on the phone, which then polls for commands), and `node
tools/perf/send.js COMMAND '{options}'` runs one there and prints its result - `timing` (the network, the renderer, and
the streaming pipeline, stage by stage, in a worker as an app runs them), `parallel` and `pipeline` (several workers, or
network and render workers in pairs, at once), `bench` (fixed workloads - whether the JavaScript engine compiles them;
an iPhone's Safari interprets JavaScript on a page served over plain http, as the lab's is - so the lab measures that
case; over https it compiles), and `reload`. A variant of `lib/` to try a change in is
`tools/perf/variant.sh NAME` (`build/perf/variants/NAME/lib`, which the lab's commands take as `lib`);
`tools/perf/compare.js NAME` compares its output with `lib/`'s (identical, else the SNR), and `tools/perf/speed.js lib
NAME` times each under Node without a JIT (`--jitless` - the phone's speed within about 20%). `--root DIR` serves
another app (PicoShelf, with its `?timing&log=URL` timeline posted to `build/perf/devlog.jsonl`).

`tools/speak.js` speaks with an exported model (`picovox_train.export CHECKPOINT --voice NAME` writes
`train/data/NAME.json` and `.bin`; `tools/unpack.js` makes one of the package's multi-speaker model), every word pronounced by the
G2P network as the package pronounces (`-r` rate, `-s` seed, `--params` the parameter frames, `--rate` the duration
multiplier, `--batch` a list of items); `-m` also takes a packaged module (`lib/models/voiceModel.js`). The
multi-speaker model takes a speaker (`--speaker`, an id of its manifest's speakers - one per voice, in `voices()`'s
order: 0 is male, the default); the voices' source shelf and the styles are `lib/synthesis.js`'s, not the tool's.

The training dashboard, `tools/dashboard/`, follows `make train` as it runs: serve the repository (`python3 -m
http.server`) and open http://localhost:8000/tools/dashboard/ (with `?train=DIR` for a run in another `TRAIN_DIR` -
another checkout's or worktree's; default `build/train`). It has three tabs (each linkable, e.g. `#listen`): Train,
Listen, and Logs. Train is the `make train` run (the default tab while it runs): the run's state, its ETA and critical
path; the CPU and GPU utilization over the run with each stage as a band (the GPU's idle time shaded,
the forecast faded); every stage of `train/data/train-stages.json` in collapsible groups - its state (waiting, queued for
the GPU, running, done, failed, stopped, or not in the run's make target, from `TRAIN_DIR/stages.jsonl`), its start,
end, and time against the expected time, and its mean CPU and GPU use; the training runs (`TRAIN_DIR/runs/`, their
`log.jsonl` and `train.log`) with charts of all their measurements - losses, validation, pitch error, and step times;
and the gates (the heteronym and G2P checks, the listener panel's scores, and the pick). Listen plays the `make train`
run's listening samples (each training stage's final model beside the shipped model - see make train, below; the
stages not rendered yet listed as waiting), and marks new samples until they're heard. Logs has the tails of `make
train`'s 12 most recently updated logs (all of them with "Show older logs"). It refreshes every second, read-only,
re-fetching only what changed, and reloads itself when its own code changes.


## The listener panel

Neural listener models score the synthesizer the way a panel of listeners would - a repeatable check of naturalness,
speaker identity, and speaking style alongside Whisper's intelligibility. They run in a Python environment of their own
(PyTorch), used only by the tools - the package has no dependencies.

| Listener | Model | Measures |
| --- | --- | --- |
| Intelligibility | Whisper `base.en` (whisper.cpp's `whisper-cli` - installed separately, e.g. `brew install whisper-cpp`; the model downloaded, pinned, into `downloads/whisper/` - `picovox_train/corpora.py`) | Word error rate against the text |
| Naturalness | [UTMOS22 strong](https://github.com/sarulab-speech/UTMOS22) (through torch.hub [tarepan/SpeechMOS](https://github.com/tarepan/SpeechMOS)) | Predicted mean opinion score, 1-5 |
| Speaker identity | [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (SpeechBrain, VoxCeleb) | A 192-dimension speaker embedding |

`picovox_train/panel.py` is the panel's scoring, shared by every evaluation (`picovox_train.distill eval`, make train's
gate): an exported model speaks a fixed evaluation set, `tools/panel-texts.json` - six Harvard sentences, six narrative
passages from public-domain books with dialogue (Austen, Melville, Dickens, Doyle, Potter, Grahame), two questions, and
two exclamations - through `tools/speak.js` (every word pronounced by the G2P network, as the package pronounces), and
the renderings are scored with UTMOS, Whisper's WER, and the ECAPA speaker similarity.

`picovox_train/listeners.py` is the scoring tool itself - WAV files (16-bit PCM) of any sample rate in, one JSON line per
file out:

```sh
make build/venv-train.build                                 # build/venv-train from train/requirements.txt
build/venv-train/bin/python -I train/src/picovox_train/listeners.py [--no-mos] [--no-embedding] [--device auto|cpu|mps] \
    [--files-from list.txt|-] file.wav ...
```

```json
{"file": "a.wav", "seconds": 2.63, "mos": 3.33, "embedding": [31.48, ...]}
```

The models load once per run (4-7 s) and then score about 35 seconds of speech a second (6 sentences) on an M3 Max's
GPU (MPS, the default on Apple silicon - about 3x the CPU, with the same scores). Files longer than 20 s are scored in
chunks. The models download on first use - about 1.5 GB, into `downloads/listeners/` (torch.hub, Hugging Face - the
training speech's phoneme recognizer, `picovox_train/ctc.py`'s, downloads there too; like the training speech, `make
clean` keeps them - `make superclean` removes them) - and later runs need no network.

The environment is built with `python3` (3.14 here; 3.12 also works - `make build/venv-train.build
PYTHON_VENV=python3.12`). Its packages are pinned in `train/requirements.txt`: PyTorch 2.14, torchaudio 2.11 (UTMOS's),
SpeechBrain 1.1, transformers 5.20 (the phoneme recognizer's wav2vec2 - `picovox_train/ctc.py`), and soxr for resampling
audio (WAV files are read with the standard library - `picovox_train/wav.py`). (The emotion2vec+ emotion recognizer,
FunASR's - about 40 packages more - was the style experiments'; it's in git history.)

Calibration (the evaluation set, or 16 recordings of each human speaker - measured in October 2026, on the human
corpora the voice model trained on before 1.7):

| Speech | MOS mean | MOS min | WER |
| --- | --- | --- | --- |
| Human - CMU ARCTIC bdl, rms, aew (men), slt, clb (women) | 4.28-4.39 | 4.01-4.27 | 2.7-4.7% |
| Human - CMU ARCTIC eey (woman) | 3.84 | 3.00 | 4.0% |
| Human - LibriTTS-R dev-clean audiobook readers 174, 251, 1272 (men), 1462, 1673, 2035 (women) | 3.77-4.40 | 3.09-4.17 | 2.6-8.2% |
| Apple Samantha (unit selection) | 3.94 | 3.58 | 3.9% |
| Apple Daniel (unit selection, British) | 3.41 | 3.07 | 5.7% |
| Apple Fred (1980s formant synthesis) | 2.81 | 2.52 | 10.6% |
| The training speech - the local TTS model's twelve voices (their held-out clips) | TODO(results) | TODO(results) | TODO(results) |

Speaker similarity: every human pair 0.0-0.48; halves of one human speaker's recordings about 0.9.


## make train

`make train` recreates the generated models - `lib/models/g2pModel.js` and `lib/models/voiceModel.js` - from a clean
checkout, by one recipe, `train/data/train.json` (every stage's settings and seeds): `make train-g2p` and `make
train-voice` at once. Each model's training ends in its gate, with its module in `build/train/` (`g2pModel.js`,
`voiceModel.js`) - `lib/models/` unchanged. The G2P is deterministic, so its gate is that it reproduces the committed
module (identical below the comment header, else within the recipe's tolerances); the voice's gate is the listener
panel against the shipped model. The voice data is built with the committed G2P (a G2P installed that differs makes the
next `make train` rebuild the voice).

The prerequisites: [Ollama](https://ollama.com/) and its TTS model, the first time - make train's training speech is
spoken by a local TTS model (`train/data/train.json`'s `tts.engine` - `phonon`, pinned by digest; see
[the training speech](#the-training-speech)), so the first run needs Ollama running at the recipe's URL
(`http://localhost:11434` - the app, or `ollama serve`); the model is pulled if it's missing, and a model of another
digest fails the stage (a new model is a new corpus). Once the corpus is generated (`downloads/tts/`), later runs -
`make train-clean` too - need no Ollama. The listener panel needs whisper.cpp's `whisper-cli` (`brew install
whisper-cpp`); everything else is downloaded, pinned - the CMU Pronouncing Dictionary and the panel's Whisper model by
their own rules (`picovox_train/corpora.py`), the listener models and the phoneme recognizer on first use.

The flow:

1. `make -j train` - both models, then a report (`picovox_train.trained report`): each model's size, the voice's
   listener-panel scores against the shipped model's, and whether each module differs from the committed one.
2. The training dashboard (below) - the gates and the listening samples.
3. `make train-check` - the package's tests with the trained models in place of `lib/models/`'s
   (`tools/trainmodels.js`, a module hook), then the blind A/B of the trained voice against the shipped one, in your
   browser (`tools/trainab.js` serves the voice lab locally, with the trained voice in `build/voicelab/trained/`; both
   sides speak with the shipped G2P, so the A/B is the voice's alone): each of the package's voices speaks the one of
   its candidate texts (the panel's Harvard sentences and the recipe's listening samples texts) where the two voices
   differ most - the distance of their parameter frames (pitch, voicing, pulse shape, tilt, the noises, the lower
   formants and bandwidths, each scaled to its range), aligned by dynamic time warping - so an audible difference is
   heard if there is one. The trials are shuffled, with 3 repeats - 15 forced choices (or Same), about 4 minutes; after
   6 Same answers in a row the page offers to finish early (no audible difference). The answers are saved as you go
   (`build/train/listen/` - a rerun resumes; a rerun after a finished round starts another, and the report counts every
   round). At the end it prints the report (`tools/trainabreport.js`): the trained voice's win rate over the decided
   trials with its 95% interval and sign test, the repeats' agreement, each voice's counts, the panel's scores, and a
   recommendation - no audible difference when at most a fifth of the answers chose a voice (keep the shipped voice:
   installing changes nothing you can hear), keep the shipped voice when it's significantly preferred, install the
   trained voice when it's preferred at least as often, else too close to call. A retrain of an unchanged recipe is
   usually no audible difference - the A/B is its guardrail against a bad run; a recipe change is where it decides.
4. `make train-install` - the trained modules that differ into `lib/models/` (the G2P when it differs below its
   comment header; `picovox_train.trained install`) - or one model, `make train-install-voice` or `make
   train-install-g2p`; then `make commit` and commit them.

The `train-*` targets run their commands - none decides on another's results. `make train-check` and `make
train-install` fail until `make train` has trained both models (`train-install-voice` and `-g2p`, their
model). Each stage is a
stamp in `build/train/` (`TRAIN_DIR`), so `make -j train` runs independent stages at once; the GPU stages run one at a
time (`picovox_train/gpulock.py` - two at once each run at about half speed) while the CPU stages run beside them:

| Stage | Runs on | Needs |
| --- | --- | --- |
| tts-corpus - the training speech: the local TTS model's clips (`picovox_train.tts generate` - Ollama) and their CTC alignment (`tts align`), in `downloads/tts/` | GPU - about 45 and 15 minutes the first time, a minute cached | `lib/models/g2pModel.js`, the dictionary |
| tts-readings - the heteronym tokens' readings measured (`picovox_train.measure`), and the clips whose tokens they change aligned again | GPU - a minute cached | tts-corpus |
| data-tts - the voices' training data (`picovox_train.tts build`, `build/train/data/tts/` - the clips' checks) | CPU (every core), ~5 min | tts-readings |
| teacher-a, teacher-b - the teacher (`picovox_train.trainmulti --recipe teacher.a`, `teacher.b`) | GPU | data-tts |
| teacher-eval - the teacher through the listener panel | CPU | teacher-b |
| targets, student, polish - the distilled student (`picovox_train.distill`) | GPU | teacher-b |
| baseline-eval - the shipped model through the panel, beside the student's GPU stages | CPU | teacher-eval |
| student-eval - the panel and the gate (`picovox_train.distill pick --against baseline`: no worse than the shipped model) | CPU | polish, baseline-eval |
| package - the voice's module, `build/train/voiceModel.js`, with its provenance | CPU | student-eval |
| heteronyms - the relabeled words from the measured readings, checked to reproduce `train/data/g2p-relabel.json` (the G2P network's input) | CPU, ~1 min | tts-readings (after data-tts) |
| g2p-network, context-labels, context-network, g2p - the G2P (the network; the heteronym labels from the pinned books; the context network and its cross-validation; the reading stage, the export, and the gate) | CPU - the network one core, ~30 min | the labels: tts-readings (after data-tts) |
| samples-reference, samples-teacher-a, -teacher-b, -student, -polish - the listening samples | CPU, under a minute each | the shipped model (after data-tts); each stage's model |

The critical path - the training speech (generated and aligned once - about an hour the first time, a minute or two
cached), the voices' data (TODO(results)), teacher A (TODO(results)) and B (TODO(results)), the targets (TODO(results)),
the student (TODO(results)) and its polish (TODO(results)), the gate, the package - took TODO(results) from scratch on
an M3 Max on full power (94 W; the training speech cached), with the G2P chain beside it. (Before 1.7, on the human
corpora, it took 71 minutes.) The package's manifest records its provenance: the commit, the recipe's and the training
data's hashes, and the gate's scores. `python -m picovox_train.smoke` runs every stage's entry point for a few steps on
the CPU.

Each training stage's final model - teacher A's last checkpoint (which teacher B starts from), teacher B's best (the
distillation's teacher), the student's best, and its polish's best (the model the package stage ships) - is rendered
for listening as the stage finishes, in its own CPU stage beside the next GPU stage (`tools/trainsamples.js`): exported
as the package stage exports it, it speaks `train/data/train.json`'s `samples` texts (Moby-Dick's opening, a dialogue,
and heteronyms) in six of the package's voices (three female, three male) exactly as the package speaks a
sentence (`lib/synthesis.js`, every word pronounced by the G2P network), and each sample is scored with UTMOS on the
CPU. The shipped `lib/models/voiceModel.js` is rendered the same way, as the reference. The WAV files are
`TRAIN_DIR/samples/STAGE-SPEAKER-TEXT.wav`, listed in `TRAIN_DIR/samples/index.json`; the dashboard's Listen tab plays
them, each beside the shipped model's.

`make train` keeps a record for the training dashboard (`tools/dashboard/`): each stage's start, end, and failure in
`TRAIN_DIR/stages.jsonl` and each command's output also in `TRAIN_DIR/logs/STAGE.log` (`tools/trainstage.sh` - the
teacher's also in its run's `train.log`), and every 20 seconds, for the life of `make train`, the CPU busy %, the GPU
utilization %, the memory used %, and the running stages in `TRAIN_DIR/util.log` (`tools/trainutil.sh`). The stages,
their dependencies, and their expected times are `train/data/train-stages.json` (`test/testTrainStages.js` keeps it in
step with the Makefile). `make train` never removes anything; `make train-clean` removes `TRAIN_DIR` for a run from
scratch.

The books make train reads - the heteronym labels' 132 Project Gutenberg books and the distillation's 22, which the
training speech's sentences come from too - are pinned:
`train/data/books.json` records each text (by its name in `build/nn/books/`, the recipe's `texts`) with its Gutenberg id
and source URL, the SHA-256 and size of its bytes as fetched, the date it was pinned, its Gutenberg revision, and a
[Wayback Machine](https://web.archive.org/) capture of the source URL with the same bytes, where one exists. Project
Gutenberg revises its books in place and keeps no old revisions (pg2554 changed on 2026-10-16 and moved the context
labels by one example), so the pins are the texts the shipped models were trained on - the G2P's labels' copies (fetched
2026-10-15) and the first full make train's distillation copies (2026-10-16, whose pg2554 is already the revision:
`pg2554-2026-10-16`). A text is fetched once (`picovox_train/books.py` - its Gutenberg URL, else its archived copy: the
first with the pinned hash) and verified on every use. A text whose hash doesn't match is never used: the stage fails
with the book, the expected and actual hashes, and how to proceed - restore the pinned copy (an earlier run's
`build/nn/books/NAME.txt`; a mismatched download is kept beside it as `NAME.txt.unpinned`), or re-pin deliberately and
retrain (`python -m picovox_train.books pin NAME` records the new text's hash and looks up an archived copy; the labels
or the distillation set change with it, so make train no longer reproduces the committed models byte for byte). The
fetch alone, for every pinned text: `python -m picovox_train.books fetch`.

`make -j train-g2p` runs the G2P alone - its stages (with the training speech's readings, tts-corpus and tts-readings),
without the voice stages or make train's ordering after the voices' data build - for G2P work. The chain's independent
parts run in worker processes - the context labels' part-of-speech tags (by book) in `G2P_JOBS`, and the context
network's trainings and cross-validate's ten folds (each one thread) in `CONTEXT_JOBS` - and the context network and its
cross-validation run beside the G2P network; every output is the same for any count. Inside make train the chain runs
beside the GPU stages, which CPU work slows (the Makefile's `BESIDE`), so `G2P_JOBS` is 2 and `CONTEXT_JOBS` 6 there;
`make train-g2p` runs `G2P_STANDALONE_JOBS` (10) for both at the default QoS. Measured at the background QoS (the
efficiency cores only, beside another job's load - October 2026): the context labels took 36 minutes (65 before: the
tags are made by book in worker processes, not by spaCy's worker pool for each book), and the ten folds 54 minutes at
once (one fold alone, two threads, took 134). On a quiet M3 Max the chain was 59 minutes - the G2P network 30, then the
reading stage, the context network, and the folds one after another (23 minutes) - and is now about the network's 30
plus the reading stage and the check (the labels, the context network, and the folds finish beside the network - with
the context network's four teachers, each fold's too, they took 26 minutes standalone).

`make train` first checks the power (`tools/trainpower.sh`, macOS laptops): the run keeps the GPU busy for hours, and
an M3 Max under it draws about 85 W, so on battery or an adapter under 90 W (`TRAIN_MIN_WATTS`) it says so and, at a
terminal, waits for Enter - plug in a bigger adapter first (Apple's 96 W or 140 W - a 14-inch MacBook Pro negotiates
about 94 W from either). `TRAIN_POWER_CHECK=0` skips it. Even on full power, a 14-inch M3 Max throttles: about 2.5
minutes into a GPU stage at about 70 W, macOS's thermal pressure reaches moderate, then heavy, the system is held near
40 W, and a teacher step takes about 0.12 s instead of 0.078 (October 2026 - every teacher A, both seeds, from step
about 2,000). `TRAIN_DIR/util.log` records the system's power and the thermal pressure, and the dashboard shows the time
throttled. `make train` and its targets (`train-voice`, `train-g2p`, and their `-check`s) keep the Mac awake
until they exit (`caffeinate -i -s` - on AC power, with the lid closed too): idle sleep once paused a run's teacher B
for 28 minutes.


## Distilling the voice model

Each retrain of the multi-speaker acoustic model ends the same way: train a big teacher, then distill small students
from it and ship the smallest one that keeps the teacher's quality. `picovox_train/distill.py` runs that pipeline, with
its settings in `train/data/train.json`'s `distill` (the teacher, the distillation set, the training settings, the student
ladder, the polish, the evaluation voices, and the tolerance):

```sh
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.distill ladder                  # everything below, in order
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.distill targets                 # 1. the teacher's targets (about 10 minutes on the GPU)
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.distill train --student s64     # 2. each student of the ladder
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.distill polish --student s64    # 3. optional: a short fine-tune on the training speech
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.distill eval teacher s96 s64 s48 s32   # 4. each model through the browser path
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.distill pick                    # 5. speed, samples, and the smallest within tolerance
```

1. **Targets** - sequence-level distillation in parameter space. The teacher speaks (tokens, speaker) to its durations
   (pauses included) and its 32 synthesizer parameters every 5 ms, as inference does, with no audio rendered. The
   inputs are the training utterances with their own voices, and every sentence of the recipe's 22 Project Gutenberg
   books (`distill.targets` `texts` - pinned texts, `train/data/books.json` - in `build/nn/books/`; tokenized by
   `picovox_train/frontend.py`, split into sentences as inference splits text) spoken twice, each time by a random one
   of the twelve voices. The set goes to `build/nn/distill/targets/` (frames as float16).
2. **Students** - the same `MultiModel` with smaller widths and depths. They share the teacher's conditioning: its
   speaker and channel-equalizer tables (a row per voice) are copied and frozen, so a voice is the same voice in a
   student. A student runs with the teacher's durations (teacher forcing on its length regulator) and learns L1 on the
   normalized parameter frames plus the duration losses against the teacher's duration outputs. The training
   (`train/data/train.json`'s `distill.train`) is 12,000 steps of Muon on the hidden matrices (AdamW for the rest) with
   a warmup-stable-decay schedule - constant, then the last 25% cooled down by 1 - sqrt (`picovox_train/optim.py`). Runs
   go to `build/nn/runs/distill-<student>/`.
3. **Polish** - `picovox_train/trainmulti.py`'s real-audio losses (the spectral loss through the differentiable
   synthesizer on the training speech's clips, the adversarial critic - started from the teacher's - and feature
   matching) for a short fine-tune.
4. **Evaluation** - each model is exported (`picovox_train/export.py`), packaged (`picovox_train/package.py`, the
   generated module's bytes and gzip bytes), and spoken by `tools/speak.js` on `tools/panel-texts.json` in the twelve
   voices (`distill.eval` `speakers`); the renderings are scored with UTMOS, Whisper's WER, and the ECAPA speaker
   similarity to the teacher's rendering of the same voice. `pick` measures `tools/speak.js`'s real-time factor, renders
   listening samples to `build/nn/distill-samples/`, and chooses the smallest student within the tolerance - mean UTMOS
   within 0.05 of the teacher's, WER at most 1 point worse, and speaker similarity at least 0.85 - writing
   `build/nn/distill/summary.json`. With `--against baseline` (make train's gate) the basis is instead the shipped
   model - `eval baseline=lib/models/voiceModel.js` speaks the packaged module, as the package loads it
   (`tools/unpack.js`) - so a new student must be no worse than what ships, and a better teacher never fails it.

Progress lines go to `build/nn/distill/` (`targets.log`, `students.log`, `eval.log`).

The distillation of the voices trained on the training speech (1.7): TODO(results) - the targets' size and time, and the
teacher and the chosen student's parameters, module size, UTMOS, WER, and similarity (the teacher 931,619 parameters,
the d64 student 282,627, by their twelve-row speaker tables).

The measurements below are of the models trained on human recordings before 1.7 (CMU ARCTIC and LibriTTS-R readers, and
CREMA-D's actors - their speaker and style tables larger), kept as the ladder's record. The first distillation (October
2026), from stage B's best checkpoint (`multi-b` step 19,000): 294,678 sentence-speaker pairs - 40,624 training
utterances and 254,054 book sentences - 397 hours of teacher speech, 18.5 GB, made in 9 minutes on an M3 Max's GPU. Each
student trained 20,000 steps (16-35 minutes); polish is 2,000 steps (28-46 minutes, the GPU shared). Six speakers by 16
panel texts; the real-time factor is `tools/speak.js`'s at 44.1 kHz (the best of three, about +/-15%):

| Model | Parameters | Module (gzip) | UTMOS | WER | Similarity | Real time (network / total) |
| --- | --- | --- | --- | --- | --- | --- |
| Teacher (dim 128, 3 + 6 layers) | 965,803 | 1,447 KB (1,062 KB) | 4.144 | 2.37% | 1 | 18x / 8.4x |
| s96 (dim 96, 3 + 6) | 570,987 | 925 KB (666 KB) | 4.147 | 1.86% | 0.978 | 24x / 8.8x |
| **s80 + polish (dim 80, 3 + 4) - chosen** | 349,547 | 614 KB (431 KB) | 4.130 | 1.71% | 0.969 | 50x / 12.1x |
| s80 | 349,547 | 613 KB (431 KB) | 4.096 | 1.91% | 0.966 | 48x / 11.5x |
| s64 + feature matching + polish | 238,251 | 459 KB (313 KB) | 4.076 | 1.76% | 0.959 | 73x / 13.6x |
| s64 + polish | 238,251 | 459 KB (313 KB) | 4.070 | 1.86% | 0.956 | 67x / 12.9x |
| s64 + feature matching | 238,251 | 457 KB (312 KB) | 4.065 | 1.61% | 0.952 | 60x / 11.5x |
| s64 (dim 64, 3 + 4) | 238,251 | 457 KB (312 KB) | 4.045 | 2.17% | 0.949 | 73x / 13.0x |
| s48 (dim 48, 2 + 4) | 127,035 | 302 KB (196 KB) | 3.935 | 2.32% | 0.924 | 147x / 17.6x |
| s32 (dim 32, 2 + 3) | 64,683 | 209 KB (125 KB) | 3.593 | 2.72% | 0.831 | 220x / 16.8x |

The polish added 0.025-0.034 UTMOS, most of it to the package's two voices of the time (CMU ARCTIC's bdl and slt), which
pure distillation lost most; intermediate-feature matching (weight 0.05) added about 0.02, and the two barely stacked.

The decoder rung (October 2026, picovox 1.2.0): on a phone without a JIT the decoder, per frame, was over half the voice
network's time, so students that shrink only it - s80's encoder - were trained on one teacher, polished, and scored:

| Student | Parameters | Module (gzip) | UTMOS | WER | Similarity |
| --- | --- | --- | --- | --- | --- |
| s80 - decoder 80 wide, 4 layers (two seeds) | 349,547 | 432 KB | 4.144, 4.165 | 1.46%, 1.26% | 0.977, 0.973 |
| **d64 - decoder 64 wide, 4 layers (three runs) - chosen** | 305,035 | 385 KB | 4.147, 4.126, 4.135 | 1.06%, 1.41%, 1.26% | 0.972, 0.968, 0.972 |
| L3 - decoder 80 wide, 3 layers | 317,947 | 397 KB | 4.136 | 1.56% | 0.974 |
| d64-L3 - decoder 64 wide, 3 layers | 283,851 | 361 KB | 4.095 | 1.76% | 0.964 |

d64 is about 0.02 UTMOS below s80 on average (the seed spread) with a lower WER, and its network is 1.24x faster on an
iPhone without a JIT. The voice's training on the GPU isn't bit-reproducible - the recipe's d64, retrained, matched its losses but not
its weights bitwise - though its export and packaging from a checkpoint are.


## Training the G2P model

PicoVox pronounces every word with one small network, `lib/models/g2pModel.js` (`lib/g2p.js` runs it;
`picovox_train/g2pmodel.py` is its bit-exact Python port, which pronounces the voice model's training data).
`picovox_train/g2p.py` trains it, with the recipe in `train/data/train.json`'s `g2p` settings, and `picovox_train.context
export` writes the module, with its context network (below):

```sh
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.heteronyms --out build/nn/heteronyms --keep train/data/g2p-relabel.json  # its words (optional)
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.g2p train --out build/nn/g2p-train      # four stages, about 30 minutes on one CPU core
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.g2p evaluate                            # the book-token error of lib/models/g2pModel.js
```

The network is an autoregressive encoder-decoder (embedding 32, a bidirectional GRU of 96 over the letters, a GRU
decoder of 192 with attention; 369,767 parameters). It trains on every word of the CMU Pronouncing Dictionary (its
first pronunciation) but a 2,000-word monitoring set, sampled half uniformly and half by the square root of the word's
count in 128 Project Gutenberg books, plus - 30% of the samples - 100,812 words the dictionary lacks, labeled by a
large teacher ensemble or by the spelling and `-s` rules where they apply (`train/data/g2p-words.json`). A base stage
(20,000 steps of 512 at learning rate 3e-3) is polished twice with more of the samples on the most frequent words
(5,000 steps at 1e-3 over the top 20,000 words, then 5,000 at 5e-4 over the top 12,000). The last stage (focus, 3,000
steps) teaches the spoken pronunciations of 56 words whose dictionary-first pronunciation isn't the usual reading
(lead L IY1 D, live L IH1 V, wind W IH1 N D, ... - `train/data/g2p-relabel.json`; a stage's `relabel` replaces those
words' targets, the rule-labeled words derived from them follow, and `focusBoost` gives them 2% of the samples). The
export quantizes to 6 bits per weight. Training on the CPU with one thread is deterministic: the recipe reproduces the
shipped model's weights bit for bit.

`picovox_train/heteronyms.py` chooses the relabeled words from the training speech's measured readings
(`picovox_train/measure.py` - below): for each word with several dictionary readings, the reading the voices used; a
word is relabeled when the voices' majority isn't the first (significant - the Wilson 95% lower bound above one half,
from 3 or more books, the independent unit: the voices share one front end), the difference is lexical (a stressed
vowel's quality, or the primary stress moving with it - not vowel reduction or stress alone), and it isn't a function
word. The committed words were chosen from the LibriTTS-R readers' usage before 1.7 (their evidence is in the file);
with `--keep` (make train's heteronyms stage), a committed word the training speech can't judge - its readings
unmeasured, or no significant majority either way - is kept, and one whose first reading the voices significantly
prefer is dropped, and the stage checks the result reproduces `train/data/g2p-relabel.json`.

The measured readings (`python -m picovox_train.measure`, make train's tts-readings stage, on the GPU - in the corpus's
`readings/`): each heteronym token in the training speech (a measured word, by its clip's alignment) is measured by the
kind of difference between its readings - where they differ in a full vowel's quality or a consonant, the phoneme
recognizer's vote (`picovox_train/ctc.py` - each reading's CTC likelihood, with the neighboring words' phones, on the
word's segment; S against Z with each only its own token); where the recognizer can't tell them apart (a reduced vowel
against a full one, the vowels it merges), the vowel's formants (F1 and F2, normalized per voice, nearest of the
voice's own reference vowels); the noun/verb stress pairs aren't measured. Each method is gated by controls - words of
one pronunciation with a phone swapped, whose votes are the method's false votes: a method's readings are used only
when at most 3% of 100 or more control votes are wrong. The measured readings are the G2P's heteronym evidence - the
relabeled words, the context labels (below), and the dictionary audit - and the voice's training tokens take the
measured reading where the recognizer's vote differs from the front end's.

### The dictionary audit

The network learns the dictionary's errors too (HOBBES HH AA1 B IY0 Z - "hobbies" - and, from it, "mobbes" M AA1 B
IY2 Z), so `train/data/g2p-fixes.json` corrects the dictionary's labels where the evidence says they're wrong; `g2p.py`
applies it in every stage, before the relabeled words (the rule-labeled words derived from a fixed word follow - HOBBES'S
HH AA1 B Z IH0 Z). `picovox_train/g2paudit.py` finds them:

```sh
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.g2paudit folds --out build/nn/g2paudit -j 5     # 10 screening models (resumable)
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.g2paudit suspects --out build/nn/g2paudit \
    --books build/nn/books --catalog ../picoshelf/data/catalog.json --heteronyms build/nn/context/labels.json
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.g2paudit audio --out build/nn/g2paudit [--speak]
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.g2paudit adjudicate --out build/nn/g2paudit
```

Two partitions of the dictionary into five folds (a word's possessives in its fold) train ten screening models (the
recipe's network and sampling, 6,000 steps, no rule- or teacher-labeled words), so every word is held out by two of
them (held-out accuracy 62% - any pronunciation, stress included; the dictionary is mostly names). A word is a suspect
when both its models' phones differ from its label (not by stress alone) by a log-probability margin of 3, or both prefer
another of its variants by that margin: 12,750 of 124,101 words - mostly legitimate irregulars (of, colonel) and
foreign names. Where the training speech's voices spoke a suspect (with `--speak`, a word of too few tokens has
sentences of the books with it spoken by the TTS model, in turn by the voices, and aligned), the CTC phoneme recognizer
(`picovox_train/ctc.py`) scores the label and each alternative (the models' predictions, the dictionary's variants -
those that would be a lexical correction: not stress or reduction alone, not a consonant dropped, not a merger's vowels,
not a function word or a one- or two-letter word) on up to 12 of its aligned tokens with their neighbors; a token votes
for a candidate 2 (log) more likely than the rest, and a confirmed error needs the votes' Wilson 95% lower bound above
one half from 3 or more books. Control words (both models agree with the label, against a model's second guess)
measure the method. The TTS model's pronunciation of a rare word or a name is its own G2P's guess, so a verdict is the
engine and the fold models together. (The committed fixes were found before 1.7 with the LibriTTS-R readers'
recordings: 140 control words, no false errors and 1.5% of the tokens' votes for the wrong candidate; the audio
confirmed 13 errors and 143 labels, the rest without audio evidence.)

The names regression set (`train/data/names-test.json`, run by `tools/heteronyms.js` as kind `names`): surnames, first
names, and place names with verified pronunciations - HOBBES first (the dictionary's own evidence: its second variant,
HOBBS, and every other -bes word), the audio's confirmed errors and labels, and well-established names. Its minimum is
the shipped module's score when it was added (63/73), so it can only go up.

### Heteronyms

Some words have readings their neighbors decide - "I read it yesterday" and "I will read it", "their lives" and "she
lives", "a lead pipe", the noun/verb stress pairs. The G2P module also holds two small pieces for them (no word list or
rule at runtime): the G2P takes a reading index (a learned vector added to its decoder's first state; reading 0 is the
default) and has an ambiguity head (whether the word's reading depends on its context); for an ambiguous word, a context
network (`picovox_train/context.py`: a character GRU over the three words and phrase-end marks each side, a
bidirectional GRU, and a layer gated by the word itself - 47,779 parameters at 6 bits) chooses the reading.

The context network is distilled from an ensemble, for steadiness: one network's choices where its labels are thin
depend on its seed (October 2026: five seeds of the same labels differed on 14% of unlabeled book windows and 9 of the
regression set's cases a pair, and adding a few seeds for one word flipped others). So four teacher networks (their
own seeds) are trained first, and the shipped network - the same size - is trained toward their mean probabilities on
its examples and on unlabeled transfer windows (the ambiguous words' other windows in the books), with context words
dropped at random (`itemDropout`).

```sh
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.contextlabels books                                # the recipe's books (pinned texts)
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.contextlabels aligned --out build/nn/context      # the training speech's measured readings
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.contextlabels tag --out build/nn/context         # part-of-speech tags (spaCy)
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.contextlabels labels --out build/nn/context       # the ambiguous words and examples
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.context train-g2p --labels build/nn/context/labels.json --init build/nn/g2p-train/focus.pt \
    --out build/nn/context/g2p.pt                                                   # the reading input and ambiguity head
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.context train-context --labels build/nn/context/labels.json --out build/nn/context/context.pt -j 4
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.context export --g2p build/nn/context/g2p.pt --context build/nn/context/context.pt
node tools/heteronyms.js                                                            # the regression set (exit 1 below the gate)
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.context cross-validate --labels build/nn/context/labels.json --out build/nn/context/folds -j 10
```

(`tag`, `train-context` - its teachers - and `cross-validate` take `-j`, their worker processes - the results
are the same for any count. `cross-validate` keeps a fold's results only when they are from the same labels and
settings.)

The settings are `train/data/train.json`'s `context` section. `make train-g2p` runs all of it (the voice stages use the
committed `lib/models/g2pModel.js`) and gates its module with `picovox_train.context check`: identical to the
committed module below the comment header (the stages are deterministic on the same inputs), else within the recipe's
tolerances (the regression set, the book error, the top-10K, the monitoring WER) - then `make train-install` replaces
the committed module only when it differs. The books are pinned (`train/data/books.json` - see make train), so a clean run reproduces the committed module.

The labels (`picovox_train/contextlabels.py`, with the curated part in `train/data/context-labels.json`) come from three
sources: the training speech's measured readings (each candidate word's token whose reading the recognizer's vote or
the formants measured - the voice's reading, heard in the audio); part-of-speech tags over the Project Gutenberg books
(spaCy, training time only - `train/requirements.txt`), mapped to readings only for words whose mapping agrees with the
measured readings on 95% of 10 or more tokens (a stress-pair or other family word of too few, by its family's mapping
on 200 or more); and hand-written seed sentences and reviewed collocations for the readings the tags can't separate
(lead the metal, bass, bow, row, does the deer, house the verb) and the pairs the audio doesn't validate - a
collocation's token is its example, not a measured one, and the corpus's disagreements with the collocations (the TTS
model's own reading errors) are reported. Two books and the regression set's sentences are held out - every source's
examples of them, the book the independent unit - and the held-out accuracy gate (`picovox_train.context gate`,
`folds`) is cross-validated by book.


## The training speech

The voice network's training speech isn't downloaded - make train makes it: `picovox_train/tts.py` has a local TTS
model speak sentences of the pinned Project Gutenberg books through Ollama's speech endpoint (`POST /v1/audio/speech` -
24 kHz WAV), one of the model's voices for each of the package's voices, then measures the clips with our own tools. Its
settings are `train/data/train.json`'s `tts` section: the engine (its URL, model, and pinned digest), the voices, the
corpus's size, the coverage quotas, the split, and the checks.

```sh
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.tts generate             # the selection and the clips (Ollama - resumable)
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.tts align                # the clips' CTC alignment (GPU - the changed clips only)
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.measure                  # the heteronym tokens' readings (GPU)
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.tts build --out DIR      # the voices' training data
```

The engine is Gradium's [Phonon](https://ollama.com/library/phonon) (`phonon`, digest `58f58601e4e8`), and the
package's voices are its Trevor (`male`), Harper (`female`), Juniper (`sarah`), Delia (`helen`), Zoey (`amy`), Hadley
(`kate`), Everly (`grace`), Miles (`james`), Preston (`robert`), Micah (`michael`), Easton (`david`), and Russell
(`thomas`).

- **The selection** - book sentences through our front end, each sentence's text its normalized words and phrase marks,
  so the engine speaks exactly the words the voice network's tokens spell (the panel's and the listening samples' texts
  excluded). First the heteronym coverage - the context gate's words, the context labels' candidate words, and the
  other measured words, each in up to its quota of sentences (200, 24, and 12) from the context labels' books, the
  rarest word first - then sentences at random from the distillation's books, to 4 hours of speech a voice (about 48
  hours). Each clip's voice and split (5% test, 5% validation) are by a hash of its text.
- **The alignment** - a wav2vec2 phoneme recognizer
  ([facebook/wav2vec2-lv-60-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft) - espeak IPA,
  20 ms frames; `picovox_train/ctc.py`) force-aligns our phones to each clip (each phone the sum of the IPA tokens it
  may be heard as, a Viterbi pass over the CTC topology). Checked against CMU ARCTIC's EHMM phone labels (300
  utterances of slt and bdl - the check is in git history), its phone onsets were within 20 ms 68% of the time and
  within 40 ms 93%, a mean absolute error of 18 ms.
- **The checks** - the recognizer's greedy phones against the sentence's: a clip cut short (its last phones missing) or
  garbled (a phone error above 35% - a misread or hallucinated clip) is left out of the training data, as is one
  clipped at full scale or with a phone of no frames.
- **The readings** - each heteronym token's reading, measured (`picovox_train/measure.py` - see
  [Training the G2P model](#training-the-g2p-model)).

The corpus is cached in `downloads/tts/<model>-<digest>/` - the selection, the clips (`wav/<voice>/<id>.wav`), their
alignments, and the measured readings - so it is generated once: with it there, make train needs no Ollama, and the
tts-corpus and tts-readings stages take a minute. A clip's id is a hash of its engine voice and text, so a bigger corpus
or another selection only adds clips. `make clean` keeps it; `make superclean` removes it with the rest of
`downloads/` (about 10 GB in all) - and since the engine isn't deterministic, a corpus generated again is new speech, and make
train no longer reproduces the committed voice model.

Another local TTS model is a change of the recipe alone: set `tts.engine` (the model, its digest - the first 12 hex
digits of its digest in `ollama list` - and the URL) and `tts.voices` (an engine voice and its sex for each of the
package's voice keys, the default first), and the next make train generates a new corpus (a new model or digest is a new
directory) and retrains the voice from it; the voices' descriptions (`lib/synthesis.js`, `test/voices.json`) follow
what they sound like. The engine is reached through Ollama's API alone - its model list and pull, and the speech
endpoint (a model, a voice, and text in; a WAV out).

Phonon's English checkpoint and its voices are under the Apache License 2.0
([ollama.com/library/phonon](https://ollama.com/library/phonon),
[huggingface.co/gradium/phonon-voices](https://huggingface.co/gradium/phonon-voices)); its training data is
undocumented. The training speech is used locally, at train time - it isn't distributed; the package holds only the
voice network trained on it.

make train's other downloads are pinned files, fetched by `picovox_train/corpora.py` - idempotently and sequentially,
resuming interrupted downloads, each checked against its SHA-256 (`--list` shows them): the CMU Pronouncing Dictionary
(`downloads/cmudict/`) and the listener panel's Whisper model (`downloads/whisper/`):

```sh
make build/venv-train.build                                         # build/venv-train (corpora.py needs only the standard library)
build/venv-train/bin/python -I train/src/picovox_train/corpora.py   # the dictionary and the Whisper model into downloads/
```

(The human speech corpora the voice model was trained on before 1.7 - CMU ARCTIC, LibriTTS-R with LibriTTSLabel's
alignments, and CREMA-D - and the speaker pool, `train/data/speakers.json`, are in git history.)


## License

PicoVox is distributed under the [MIT License](LICENSE). The CMU Pronouncing Dictionary, the G2P network's training
labels, isn't distributed with PicoVox: make train downloads it, pinned, from
[cmusphinx/cmudict](https://github.com/cmusphinx/cmudict) (its own BSD-style license) into `downloads/cmudict/`. Nor is
the voice network's training speech: make train generates it with a local TTS model, Gradium's Phonon (its checkpoint
and voices under the Apache License 2.0 - see [the training speech](#the-training-speech)).
