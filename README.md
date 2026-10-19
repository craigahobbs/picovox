# PicoVox

[![npm](https://img.shields.io/npm/v/picovox)](https://www.npmjs.com/package/picovox)
[![GitHub](https://img.shields.io/github/license/craigahobbs/picovox)](https://github.com/craigahobbs/picovox/blob/main/LICENSE)

**PicoVox - realistic speech synthesis in pure JavaScript, under 1 MB.**

**The paper:** [PicoVox: Realistic Neural Speech Synthesis in Under 1 MB of JavaScript](https://craigahobbs.github.io/picovox/paper/) -
how PicoVox's networks are designed, trained, distilled, and evaluated.

PicoVox speaks English text with natural-sounding male and female voices, entirely on your device - in the browser or
in Node. There is no recorded speech, no server, no WebAssembly, and no runtime dependencies: a small grapheme-to-phoneme
network pronounces every word, a small neural network (350K parameters) predicts interpretable source-filter
synthesizer parameters every 5 ms, and a JavaScript synthesizer with an LF glottal source renders them. The whole
package - code and both networks - is under 1 MB.

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
`options.context` is an `AudioContext` to play in (by default, a new one, closed when the playback ends). Under Node,
`speak` throws an error - there is no Web Audio; use `synthesize` and `toWav`.

### `voices()`

The built-in voices - `[{key, label, sex}]`:

| Key | Label | Sex |
| --- | --- | --- |
| `male` (the default) | Male | male |
| `female` | Female | female |

### `toWav(samples, sampleRate)`

Encode samples (a `Float32Array`, or an array of numbers, -1 to 1 - values beyond are clipped) as a 16-bit PCM mono WAV
file - a `Uint8Array` of the file's bytes.

### Options

`synthesize`, `stream`, and `speak` take the same options:

| Option | Default | |
| --- | --- | --- |
| `voice` | `'male'` | The voice's key - see `voices()` |
| `rate` | `1` | The speaking speed, from 0.5 to 2 (1.2 is 20% faster) |
| `seed` | `1000` | The noise seed (an integer) - sentence *i* of the text is rendered with seed + *i* |
| `sampleRate` | `24000` | The output sample rate (Hz), from 8000 to 96000 - 24 kHz is the voices' own rate |

The text is split into sentences (and paragraphs, at blank lines); each sentence starts with a short pause - 0.24
seconds, or 0.4 before a paragraph (so the text's first sentence starts with one). Numbers, common abbreviations, dashes,
ellipses, and Roman-numeral chapter headings are spoken as words. An invalid text or option throws a `TypeError` or a
`RangeError` with a message saying what's wrong (`stream` and `speak` throw at once; `synthesize` rejects).

The output is bit-for-bit deterministic: the same text and options give the same samples, in the browser and in Node.

### The modules

The package's modules are importable too (`picovox/lib/<module>.js`) - the pieces the API is built from, for an app
that runs its own synthesis loop (PicoShelf's worker does):

| Module | |
| --- | --- |
| `lib/index.js` | The API |
| `lib/synthesis.js` | The pipeline - the voices, a sentence's blocks (`sentenceBlocks`), a text's (`textBlocks`), the gain and limiter |
| `lib/sentences.js` | Text to sentences (`splitSentences`) and their speakable text (`speechText`) |
| `lib/lexicon.js` | Normalization and phrasing (`textToPhrases`), every word pronounced by the pronouncer (`setPronouncer`) |
| `lib/g2p.js` | The grapheme-to-phoneme network's inference (`loadG2p`) - heteronyms read from their context |
| `lib/tokens.js` | Phrases to the voice network's tokens |
| `lib/model.js` | The voice network - text to parameter frames (`speakFrames`, `streamFrames`), any speaker blend and style |
| `lib/render.js` | The source-filter synthesizer - parameter frames to samples, streaming (`createRenderer`, `render`) |
| `lib/lf.js`, `lib/rng.js`, `lib/fft.js`, `lib/params.js` | The LF glottal pulse, the counter-based noise, the FFT, the parameter set |
| `lib/transforms.js` | Voice and style transforms of parameter frames (pitch, tract length, bandwidths, whisper, effort) |
| `lib/package.js` | The generated model modules' loader (`loadPackagedModel`) |
| `lib/wav.js` | WAV encoding (`encodeWav`) |
| `lib/models/g2pModel.js`, `lib/models/voiceModel.js` | The generated models (`make train`) |


## Size and speed

| | |
| --- | --- |
| The package (`npm pack`) | 803 KB packed, 1.2 MB unpacked - the G2P network 419 KB and the voice network 616 KB of it, as base64 |
| The voice network | 349,547 parameters, 8-bit weights - 432 KB gzipped |
| The G2P network | 373,256 parameters (and the heteronym context network's 47,779), 6-bit weights |
| Speed, Apple M3 Max (Node) | The voice network about 94x real time; synthesis end to end (network and rendering, 24 kHz) about 40x |
| Speed, iPhone (Safari) | About 1.18x real time - fast enough for continuous playback |
| Startup | About 0.1 s to load both networks (Node, M3 Max) |

The voices score a mean UTMOS (a neural predictor of listeners' naturalness ratings, 1-5) of about 4.1 on the listener
panel's texts, with a Whisper word error rate under 2% - see [the listener panel](#the-listener-panel).


## How the models are trained

Both networks are trained from scratch from public corpora - CMU ARCTIC, LibriTTS-R, CREMA-D, and the CMU Pronouncing
Dictionary (the G2P network's training labels; it is not used at runtime) - by `make train`, which rebuilds
`lib/models/g2pModel.js` and `lib/models/voiceModel.js` from a clean checkout in about 65 minutes on an M3 Max: a large
teacher model trained through a differentiable copy of the synthesizer, distilled to the small student that ships. See
[make train](#make-train) below, and the paper,
[PicoVox: Realistic Neural Speech Synthesis in Under 1 MB of JavaScript](paper/paper.md), for the methods and results.


## PicoShelf

[PicoShelf](https://craigahobbs.github.io/picoshelf/) is the showcase app - an audiobook player for every public-domain
book (Project Gutenberg and Standard Ebooks), read aloud by PicoVox in the browser, in a Web Worker
([source](https://github.com/craigahobbs/picoshelf)).


## Development

This repository is the package (`lib/`, `test/`) and the training pipeline that makes its models (`train/`, `nn/js/`,
`tools/`, `data/`, the paper in `paper/`). The package is developed with
[javascript-build](https://github.com/craigahobbs/javascript-build#readme), and the training's Python code with
[python-build](https://github.com/craigahobbs/python-build#readme):

```sh
make commit      # the package's unit tests (100% coverage), lint, and API documentation (jsdoc) - and the training's gate
make test        # the package's unit tests (TEST='name' runs one)
make -C train commit      # the training's gate alone - its unit tests, pylint, and 100% branch coverage
make -C train test TEST=tests.test_params    # one test module of the training's
make train       # retrain the voice model (see below)
make -j train-check # check that the committed G2P and heteronym relabels reproduce (standalone)
make -j train-g2p   # the G2P chain alone, for G2P work
```

The training pipeline is a Python package, `picovox_train` (`train/src/picovox_train/` - `picovox_train/<module>.py`
below), and Node scripts (`nn/js/`, `tools/`); they use the package's modules but aren't part of it. `train/` is a
python-build package of its own: its unit tests (`train/src/tests/`) run on the CPU in a few minutes, with tiny
synthetic data and no network, in python-build's environment (`train/build/venv/`), which installs the package with the
training's pinned PyTorch and numpy (`train/pyproject.toml`, the same versions as `requirements/nn.txt`). `make train`
runs the package from its source (`PYTHONPATH=train/src`) in the training environments, `build/venv-<name>` from
`requirements/<name>.txt` - `venv-nn` (PyTorch, Praat), `venv-listeners` (the listener models), `venv-pos` (spaCy), and
`venv-analysis` (the corpus download). A training module runs the same way by hand:

```sh
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.smoke   # every training entry point, a few steps on the CPU
```


## Command-line tools

```sh
node nn/js/speak.js -m data/nn/slt "Hello, world." -o out/hello.wav    # an exported model (data/nn/<voice>.json and .bin)
node nn/js/speak.js -m data/nn/multi "Hello." -o out/hello.wav --voice narrator-f   # a designed voice (data/nn/voices.json)
node nn/js/unpack.js lib/models/voiceModel.js out/voice                 # a packaged model back to its export
node nn/js/packagecheck.js lib/models/voiceModel.js out/voice           # check a packaged model against its export
node nn/js/voicescheck.js [data/nn/multi]                               # check the designed voices
node tools/heteronyms.js                                                # the heteronym and names regression sets
```

`nn/js/speak.js` speaks with an exported model (`picovox_train.export CHECKPOINT --voice NAME` writes
`data/nn/NAME.json` and `.bin`; `data/nn/multi` is an export of the multi-speaker model), every word pronounced by the
G2P network as the package pronounces (`-r` rate, `-s` seed, `--params` the parameter frames, `--batch` a list of
items); the multi-speaker model takes a speaker (`--speaker`, `--blend`), a style (`--style`, `--intensity`), or a
designed voice and style (`--voice`, `--voice-style` - `nn/js/voices.js`, the designed voices of `data/nn/voices.json`,
each a blend of the model's speakers with a style and transforms).

The training dashboard, `tools/dashboard/`, follows `make train` and the neural voices' training as they run: serve the
repository (`python3 -m http.server`), start its monitor (`node tools/dashboard/monitor.js`), and open
http://localhost:8000/tools/dashboard/. It has five tabs (each linkable, e.g. `#listen`): Train, Listen, Runs, Logs,
and Archive. Train is a `make train` run (`tools/dashboard/pipeline.json`'s `train`: its `TRAIN_DIR`, default
`build/train`; the default tab while it runs): the run's state, its ETA and critical path; the CPU and GPU utilization
over the run with each stage as a band (the GPU's idle time shaded, the forecast faded); every stage of
`data/nn/train-stages.json` in collapsible groups - its state (waiting, queued for the GPU, running, done, failed,
stopped, or not in the run's make target), its start, end, and time against the expected time, and its mean CPU and
GPU use; the training runs' curves (the teacher's compared with the earlier teacher runs); and the gates (the
heteronym and G2P checks, the listener panel's scores, and the pick). A run without the stage events (from before the
bookkeeping) is inferred from its stamps and make's log (`pipeline.json`'s `train.log`). Listen plays the `make train` run's
listening samples (each training stage's final model beside the shipped model - see make train, below; the stages not
rendered yet listed as waiting), the monitor's checkpoint samples (below), and other samples in `build/nn/`, and marks
new samples until they're heard. Runs has the runs in
`build/nn/runs/` (their `log.jsonl` and `train.log`) with charts of their losses, validation, pitch error, and step
times. Logs has the tails
of `make train`'s logs and the logs in `build/nn/` updated in the last hour (all of them with "Show older logs").
Archive shows the earlier experiment pipelines (`tools/dashboard/pipelines/`): each phase is a group with its progress,
what's happening now, and how many of its steps are done, and it opens to its rows - the workflow's agents, the planned
steps (each tracked by a log, the newest training run, an output file, and its process), and the work that appeared
during it. It refreshes every second, read-only, re-fetching only what changed, and reloads itself when its own code
changes.

The monitor writes `build/nn/activity.json` every second: the Python and Node processes working in the repository for
10 seconds or more (with their multiprocessing workers folded in), running now or before, and the workflow's agents
(`pipeline.json`'s workflow, else the newest workflow in `~/.claude/projects/`) - so the dashboard shows the steps that
write no log. It also renders each new checkpoint of the pipeline's runs (`build/nn/runs/<run>/step-*.pt`) with
`pipeline.json`'s sample speakers and text, scores the samples with UTMOS on the CPU (leaving the GPU to training), and
lists them in `build/nn/samples-progress/index.json` with the package's voices as a reference; the dashboard's Checkpoint
samples section plays them, newest first.


## The listener panel

Neural listener models score the synthesizer the way a panel of listeners would - a repeatable check of naturalness,
speaker identity, and speaking style alongside Whisper's intelligibility. They run in a Python environment of their own
(PyTorch), used only by the tools - the package has no dependencies.

| Listener | Model | Measures |
| --- | --- | --- |
| Intelligibility | Whisper `base.en` (whisper.cpp, `whisper-cli`, the model in `data/whisper/`) | Word error rate against the text |
| Naturalness | [UTMOS22 strong](https://github.com/sarulab-speech/UTMOS22) (through torch.hub [tarepan/SpeechMOS](https://github.com/tarepan/SpeechMOS)) | Predicted mean opinion score, 1-5 |
| Speaker identity | [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (SpeechBrain, VoxCeleb) | A 192-dimension speaker embedding |
| Style | [emotion2vec+ base](https://huggingface.co/emotion2vec/emotion2vec_plus_base) (FunASR) | Probabilities of angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown |

`picovox_train/panel.py` is the panel's scoring, shared by every evaluation (`picovox_train.distill eval`, make train's
gate): an exported model speaks a fixed evaluation set, `tools/panel-texts.json` - six Harvard sentences, six narrative
passages from public-domain books with dialogue (Austen, Melville, Dickens, Doyle, Potter, Grahame), two questions, and
two exclamations - through `nn/js/speak.js` (every word pronounced by the G2P network, as the package pronounces), and
the renderings are scored with UTMOS, Whisper's WER, and the ECAPA speaker similarity.

`picovox_train/listeners.py` is the scoring tool itself - WAV files of any sample rate in, one JSON line per file out:

```sh
make listeners                                              # build/venv-listeners from requirements/listeners.txt
build/venv-listeners/bin/python -I picovox_train/listeners.py [--no-mos] [--no-embedding] [--no-emotion] [--device auto|cpu|mps] \
    [--files-from list.txt|-] file.wav ...
```

```json
{"file": "a.wav", "seconds": 2.63, "mos": 3.33, "embedding": [31.48, ...], "emotion": {"angry": 0.0, ..., "neutral": 1.0}}
```

The models load once per run (4-7 s) and then score about 35 seconds of speech a second (6 sentences) on an M3 Max's
GPU (MPS, the default on Apple silicon - about 3x the CPU, with the same scores). Files longer than 20 s are scored in
chunks. The models download on first use - about 1.5 GB, into `build/listeners/` (torch.hub, Hugging Face) - and
later runs need no network.

The environment is built with `python3` (3.14 here; 3.12 also works - `make listeners PYTHON_VENV=python3.12`). Its
packages are pinned in `requirements/listeners.txt`: PyTorch 2.14, torchaudio 2.11 (resampling, for UTMOS),
SpeechBrain 1.1, FunASR 1.4 (emotion2vec), and soundfile and soxr for reading and resampling audio.

Calibration (the evaluation set, or 16 recordings of each human speaker; October 2026):

| Speech | MOS mean | MOS min | WER |
| --- | --- | --- | --- |
| Human - CMU ARCTIC bdl, rms, aew (men), slt, clb (women) | 4.28-4.39 | 4.01-4.27 | 2.7-4.7% |
| Human - CMU ARCTIC eey (woman) | 3.84 | 3.00 | 4.0% |
| Human - LibriTTS-R dev-clean audiobook readers 174, 251, 1272 (men), 1462, 1673, 2035 (women) | 3.77-4.40 | 3.09-4.17 | 2.6-8.2% |
| Apple Samantha (unit selection) | 3.94 | 3.58 | 3.9% |
| Apple Daniel (unit selection, British) | 3.41 | 3.07 | 5.7% |
| Apple Fred (1980s formant synthesis) | 2.81 | 2.52 | 10.6% |

Speaker similarity: every human pair 0.0-0.48; halves of one human speaker's recordings about 0.9.


## make train

`make train` recreates the generated models - `lib/models/g2pModel.js` and `lib/models/voiceModel.js` - from a clean
checkout, by one recipe, `data/nn/train.json` (every stage's settings and seeds). Each stage is a stamp in
`build/train/` (`TRAIN_DIR`), so `make -j train` runs independent stages at once; the GPU stages run one at a time
(`picovox_train/gpulock.py` - two at once each run at about half speed) while the CPU stages run beside them:

| Stage | Runs on | Needs |
| --- | --- | --- |
| corpora - CMU ARCTIC, LibriTTS-R and its alignments, CREMA-D | network | |
| g2p-network, context-labels, context-network, g2p - the G2P retrain (the network; the heteronym labels from the pinned books; the context network and its cross-validation; the reading stage and export), checked to reproduce `lib/models/g2pModel.js` | CPU - the network one core, ~30 min | corpora (the labels) |
| heteronyms - the relabeled words, checked to reproduce `data/nn/g2p-relabel.json` | CPU, ~6 min | corpora |
| data-multi - the readers' training data (the pool: `data/nn/speakers.json`) | CPU (every core) | corpora, `lib/models/g2pModel.js` |
| cremad-align, data-cremad - the emotional speech | GPU 2 min, CPU 1 min | corpora |
| teacher-a, teacher-b - the teacher (`picovox_train.trainmulti --recipe teacher.a`, `teacher.b`) | GPU | the data |
| teacher-eval - the teacher through the listener panel | CPU | teacher-b |
| targets, student, polish - the distilled student (`picovox_train.distill`) | GPU | teacher-b |
| student-eval - the panel and the gate (`picovox_train.distill pick --against baseline`: no worse than the shipped model) | CPU | polish, teacher-eval |
| package - `lib/models/voiceModel.js`, with its provenance | CPU | student-eval (after samples-reference) |
| samples-reference, samples-teacher-a, -teacher-b, -student, -polish - the listening samples | CPU, under a minute each | the shipped model (after data-multi); each stage's model |

The critical path - the readers' data (7.3 minutes), teacher A (28.6) and B (20.4), the targets (3.6), the student
(4.9) and its polish (4.4), the gate, the package (2.0) - took 71 minutes from scratch on an M3 Max on full power (94
W; the corpora downloaded once before), with the G2P chain beside it; without it - make train-check now runs it
standalone - teacher A takes about 19, so about 62. The voice data is built with the committed G2P; make train-check
reproduces the G2P and checks its module against the committed `lib/models/g2pModel.js` (identical, else within the
recipe's tolerances - it never replaces it), and the heteronym relabeling against `data/nn/g2p-relabel.json`. The
package's manifest records its provenance: the commit, the recipe's and the training data's hashes, and the gate's
scores. `python -m picovox_train.smoke` runs every stage's entry point for a few steps on the CPU.

Each training stage's final model - teacher A's last checkpoint (which teacher B starts from), teacher B's best (the
distillation's teacher), the student's best, and its polish's best (the model the package stage ships) - is rendered
for listening as the stage finishes, in its own CPU stage beside the next GPU stage (`tools/trainsamples.js`): exported
as the package stage exports it, it speaks `data/nn/train.json`'s `samples` texts (Moby-Dick's opening, a dialogue,
and heteronyms) in six of the training speakers' voices (three female, three male) exactly as the package speaks a
sentence (`lib/synthesis.js`, every word pronounced by the G2P network), and each sample is scored with UTMOS on the
CPU. The shipped `lib/models/voiceModel.js` is rendered the same way, as the reference, before the package stage
replaces it. The WAV files are `TRAIN_DIR/samples/STAGE-SPEAKER-TEXT.wav`, listed in `TRAIN_DIR/samples/index.json`;
the dashboard's Listen tab plays them, each beside the shipped model's.

`make train` keeps a record for the training dashboard (`tools/dashboard/`): each stage's start, end, and failure in
`TRAIN_DIR/stages.jsonl` and each command's output also in `TRAIN_DIR/logs/STAGE.log` (`tools/trainstage.sh` - the
teacher's also in its run's `train.log`), and every 20 seconds, for the life of `make train`, the CPU busy %, the GPU
utilization %, the memory used %, and the running stages in `TRAIN_DIR/util.log` (`tools/trainutil.sh`). The stages,
their dependencies, and their expected times are `data/nn/train-stages.json` (`test/testTrainStages.js` keeps it in
step with the Makefile). `make train` never removes anything; `make train-clean` removes `TRAIN_DIR` for a run from
scratch.

The books make train reads - the heteronym labels' 132 Project Gutenberg books and the distillation's 22 - are pinned:
`data/nn/books.json` records each text (by its name in `build/nn/books/`, the recipe's `texts`) with its Gutenberg id
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

`make -j train-g2p` runs the G2P chain alone - its four stages, without the voice stages or make train's ordering after
the readers' data build - for G2P work. The chain's independent parts run in `G2P_JOBS` worker processes - the context
labels' aligned readings (by utterance) and part-of-speech tags (by book), and cross-validate's ten folds (each one
thread) - and the context network and its cross-validation run beside the G2P network; every output is the same for any
count. Inside make train the chain runs beside the GPU stages, which CPU work slows (the Makefile's `BESIDE`), so
`G2P_JOBS` is 2 there; `make train-g2p` runs `G2P_STANDALONE_JOBS` (10) at the default QoS. Measured at the background
QoS (the efficiency cores only, beside another job's load - October 2026): the context labels took 36 minutes (65
before: the tags are made by book in worker processes, not by spaCy's worker pool for each book), and the ten folds 54
minutes at once (one fold alone, two threads, took 134). On a quiet M3 Max the chain was 59 minutes - the G2P network
30, then the reading stage, the context network, and the folds one after another (23 minutes) - and is now about the
network's 30 plus the reading stage and the check (the labels, the context network, and the folds finish beside the
network - with the context network's four teachers, each fold's too, they took 26 minutes standalone).

`make train` first checks the power (`tools/trainpower.sh`, macOS laptops): the run keeps the GPU busy for hours, and
an M3 Max under it draws about 85 W, so on battery or an adapter under 90 W (`TRAIN_MIN_WATTS`) it says so and, at a
terminal, waits for Enter - plug in a bigger adapter first (Apple's 96 W or 140 W - a 14-inch MacBook Pro negotiates
about 94 W from either). `TRAIN_POWER_CHECK=0` skips it.


## Distilling the voice model

Each retrain of the multi-speaker acoustic model ends the same way: train a big teacher, then distill small students
from it and ship the smallest one that keeps the teacher's quality. `picovox_train/distill.py` runs that pipeline, with
its settings in `data/nn/train.json`'s `distill` (the teacher, the distillation set, the training settings, the student
ladder, the polish, the evaluation speakers, and the tolerance):

```sh
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.distill ladder                  # everything below, in order
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.distill targets                 # 1. the teacher's targets (about 10 minutes on the GPU)
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.distill train --student s64     # 2. each student of the ladder
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.distill polish --student s64    # 3. optional: a short fine-tune on the recordings
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.distill eval teacher s96 s64 s48 s32   # 4. each model through the browser path
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.distill pick                    # 5. speed, samples, and the smallest within tolerance
```

1. **Targets** - sequence-level distillation in parameter space. The teacher speaks (tokens, speaker, style) to its
   durations (pauses included) and its 32 synthesizer parameters every 5 ms, as inference does, with no audio rendered.
   The inputs are the training utterances with their own speakers and styles, and every sentence of the recipe's 22
   Project Gutenberg books (`distill.targets` `texts` - pinned texts, `data/nn/books.json` - in `build/nn/books/`;
   tokenized by `picovox_train/frontend.py`, split into sentences as inference splits text) spoken twice, each time by a random reader (8% an emotional-corpus actor, 10% a blend of two readers) in
   a neutral or, 30% of the time, an emotional style at a random intensity. The set goes to `build/nn/distill/targets/`
   (frames as float16).
2. **Students** - the same `MultiModel` with smaller widths and depths. They share the teacher's conditioning: its
   speaker, style, and channel-equalizer tables are copied and frozen, so a voice designed as a blend of the teacher's
   speakers is the same voice in a student. A student runs with the teacher's durations (teacher forcing on its length
   regulator) and learns L1 on the normalized parameter frames plus the duration losses against the teacher's duration
   outputs. `--init-teacher` copies every teacher tensor of the same shape; `--feature-weight` adds intermediate-feature
   matching. The training (`data/nn/train.json`'s `distill.train`) is 12,000 steps of Muon on the hidden matrices
   (AdamW for the rest) with a warmup-stable-decay schedule - constant, then the last 25% cooled down by 1 - sqrt
   (`picovox_train/optim.py`). Runs go to `build/nn/runs/distill-<student>/`.
3. **Polish** - `picovox_train/trainmulti.py`'s real-audio losses (the spectral loss through the differentiable
   synthesizer on the recordings, the adversarial critic - started from the teacher's - and feature matching) for a
   short fine-tune.
4. **Evaluation** - each model is exported (`picovox_train/export.py`), packaged (`picovox_train/package.py`, the
   generated module's bytes and gzip bytes), and spoken by `nn/js/speak.js` on `tools/panel-texts.json` for six
   speakers; the renderings are scored with UTMOS, Whisper's WER, and the ECAPA speaker similarity to the teacher's
   rendering of the same speaker. `pick` measures `nn/js/speak.js`'s real-time factor, renders listening samples to
   `build/nn/distill-samples/`, and chooses the smallest student within the tolerance - mean UTMOS within 0.05 of the
   teacher's, WER at most 1 point worse, and speaker similarity at least 0.85 - writing `build/nn/distill/summary.json`.
   With `--against baseline` (make train's gate) the basis is instead the shipped model - `eval
   baseline=lib/models/voiceModel.js` speaks the packaged module, as the package loads it (`nn/js/unpack.js`) - so a new
   student must be no worse than what ships, and a better teacher never fails it.

Progress lines go to `build/nn/distill/` (`targets.log`, `students.log`, `eval.log`), which the training dashboard
follows.

The first distillation (October 2026), from stage B's best checkpoint (`multi-b` step 19,000): 294,678 sentence-speaker
pairs - 40,624 training utterances and 254,054 book sentences - 397 hours of teacher speech, 18.5 GB, made in 9
minutes on an M3 Max's GPU. Each student trained 20,000 steps (16-35 minutes); polish is 2,000 steps (28-46 minutes,
the GPU shared). Six speakers by 16 panel texts; the real-time factor is `nn/js/speak.js`'s at 44.1 kHz (the best of
three, about +/-15%):

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

The polish adds 0.025-0.034 UTMOS, most of it to the package's two ARCTIC voices (bdl and slt), which pure
distillation loses most; intermediate-feature matching (weight 0.05) adds about 0.02, and the two barely stack.


## Training the G2P model

PicoVox pronounces every word with one small network, `lib/models/g2pModel.js` (`lib/g2p.js` runs it;
`picovox_train/g2pmodel.py` is its bit-exact Python port, which pronounces the voice model's training data).
`picovox_train/g2p.py` trains and exports it, with the recipe in `data/nn/train.json`'s `g2p` settings:

```sh
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.heteronyms --out build/nn/heteronyms  # data/nn/g2p-relabel.json's words (optional)
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.g2p train --out build/nn/g2p-train      # four stages, about 30 minutes on one CPU core
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.g2p export build/nn/g2p-train --out lib/models/g2pModel.js
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.g2p evaluate                            # the book-token error of lib/models/g2pModel.js
```

The network is an autoregressive encoder-decoder (embedding 32, a bidirectional GRU of 96 over the letters, a GRU
decoder of 192 with attention; 369,767 parameters). It trains on every word of the CMU Pronouncing Dictionary (its
first pronunciation) but a 2,000-word monitoring set, sampled half uniformly and half by the square root of the word's
count in 128 Project Gutenberg books, plus - 30% of the samples - 100,812 words the dictionary lacks, labeled by a
large teacher ensemble or by the spelling and `-s` rules where they apply (`data/nn/g2p-words.json`). A base stage
(20,000 steps of 512 at learning rate 3e-3) is polished twice with more of the samples on the most frequent words
(5,000 steps at 1e-3 over the top 20,000 words, then 5,000 at 5e-4 over the top 12,000). The last stage (focus, 3,000
steps) teaches the readers' pronunciations of 56 words whose dictionary-first pronunciation isn't the usual reading
(lead L IY1 D, live L IH1 V, wind W IH1 N D, ... - `data/nn/g2p-relabel.json`; a stage's `relabel` replaces those
words' targets, the rule-labeled words derived from them follow, and `focusBoost` gives them 2% of the samples). The
export quantizes to 6 bits per weight. Training on the CPU with one thread is deterministic: the recipe reproduces the
shipped model's weights bit for bit.

`picovox_train/heteronyms.py` chooses the relabeled words from the LibriTTS-R readers' usage: for each word with several
dictionary pronunciations, the variant the readers used by the forced alignments; a word is relabeled when the
readers' majority isn't the first (significant, 3+ speakers), the difference is lexical (not vowel reduction or stress
alone, not a function word), and the vowel's formants (or the fricative's voicing) agree where measured.

### The dictionary audit

The network learns the dictionary's errors too (HOBBES HH AA1 B IY0 Z - "hobbies" - and, from it, "mobbes" M AA1 B
IY2 Z), so `data/nn/g2p-fixes.json` corrects the dictionary's labels where the evidence says they're wrong; `g2p.py`
applies it in every stage, before the relabeled words (the rule-labeled words derived from a fixed word follow - HOBBES'S
HH AA1 B Z IH0 Z). `picovox_train/g2paudit.py` finds them:

```sh
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.g2paudit folds --out build/nn/g2paudit -j 5     # 10 screening models (resumable)
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.g2paudit suspects --out build/nn/g2paudit \
    --books build/nn/books --catalog ../picoshelf/data/catalog.json --heteronyms build/nn/context/labels.json
HF_HOME=build/nn/cremad/hf PYTHONPATH=train/src build/venv-listeners/bin/python -B -m picovox_train.g2paudit audio --out build/nn/g2paudit
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.g2paudit adjudicate --out build/nn/g2paudit [--extra EXTRA.json]
```

Two partitions of the dictionary into five folds (a word's possessives in its fold) train ten screening models (the
recipe's network and sampling, 6,000 steps, no rule- or teacher-labeled words), so every word is held out by two of
them (held-out accuracy 62% - any pronunciation, stress included; the dictionary is mostly names). A word is a suspect
when both its models' phones differ from its label (not by stress alone) by a log-probability margin of 3, or both prefer
another of its variants by that margin: 12,750 of 124,101 words - mostly legitimate irregulars (of, colonel) and
foreign names. Where 3+ LibriTTS-R readers spoke a suspect, the CTC phoneme recognizer (`cremad.py`'s) scores the label
and each alternative (the models' predictions, the dictionary's variants - those that would be a lexical correction: not
stress or reduction alone, not a consonant dropped, not a merger's vowels, not a function word or a one- or two-letter
word) on up to 12 of its aligned tokens with their neighbors; a token votes for a candidate 2 (log) more likely than the
rest, and a confirmed error needs the votes' Wilson 95% lower bound above one half from 3+ speakers. 140 control words
(both models agree with the label, against a model's second guess) measure the method: no false errors, 1.5% of the
tokens' votes for the wrong candidate. The audio confirmed 13 errors and 143 labels; the rest have no audio evidence.

The names regression set (`data/nn/names-test.json`, run by `tools/heteronyms.js` as kind `names`): surnames, first
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
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.contextlabels books                                # the recipe's books (pinned texts)
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.contextlabels aligned --out build/nn/context      # readers' readings (MFA alignments)
PYTHONPATH=train/src build/venv-pos/bin/python -B -m picovox_train.contextlabels tag --out build/nn/context         # part-of-speech tags (spaCy)
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.contextlabels labels --out build/nn/context       # the ambiguous words and examples
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.context train-g2p --labels build/nn/context/labels.json --init build/nn/g2p-train/focus.pt \
    --out build/nn/context/g2p.pt                                                   # the reading input and ambiguity head
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.context train-context --labels build/nn/context/labels.json --out build/nn/context/context.pt -j 4
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.context export --g2p build/nn/context/g2p.pt --context build/nn/context/context.pt
node tools/heteronyms.js                                                            # the regression set (exit 1 below the gate)
PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.context cross-validate --labels build/nn/context/labels.json --out build/nn/context/folds -j 10
```

(`aligned`, `tag`, `train-context` - its teachers - and `cross-validate` take `-j`, their worker processes - the results
are the same for any count. `cross-validate` keeps a fold's results only when they are from the same labels and
settings.)

The settings are `data/nn/train.json`'s `context` section. `make train-check` runs all of it (the voice stages use the
committed `lib/models/g2pModel.js`) and checks its module with `picovox_train.context check`: identical to the
committed module below the comment header (the stages are deterministic on the same inputs), else within the recipe's
tolerances (the regression set, the book error, the top-10K, the monitoring WER) - it never replaces the committed
module. The books are pinned (`data/nn/books.json` - see make train), so a clean run reproduces the committed module.

The labels (`picovox_train/contextlabels.py`, with the curated part in `data/nn/context-labels.json`) come from three
sources: the LibriTTS-R readers' readings by the forced alignments; part-of-speech tags over the Project Gutenberg books
(spaCy, training time only - `requirements/pos.txt`), mapped to readings only for words whose mapping agrees with the
aligned audio on 95% of 10 or more tokens; and hand-written seed sentences and reviewed collocations for the readings
the tags can't separate (lead the metal, bass, bow, row, does the deer, house the verb) and the pairs the audio doesn't
validate. The voice training data (`picovox_train.multi build`) takes a heteronym token's reading from the alignment
where the aligner chose another dictionary reading than the front end's (`picovox_train/readings.py`; `python -m
picovox_train.readings check`) - except where the two differ only by S against Z (use, close, excuse): the aligner can't
judge word-final voicing, so those keep the front end's reading, and the held-out accuracy gate (`picovox_train.context
gate`, `folds`) leaves them to the regression set.


## The corpora

The training data comes from public corpora, downloaded by `picovox_train/corpora.py` (make train's first stage) -
idempotently and sequentially, resuming interrupted downloads; `--list` shows them; `data/corpora/` is gitignored and
its `README.md` records each corpus's source, license, and size:

```sh
make build/venv-analysis.build                                  # build/venv-analysis (picovox_train/corpora.py's environment)
build/venv-analysis/bin/python -I picovox_train/corpora.py              # download the corpora into data/corpora/
```

| Corpus | Speakers | Download | License |
| --- | --- | --- | --- |
| [CMU ARCTIC](http://festvox.org/cmu_arctic/) 0.95 releases | bdl, rms (male); slt, clb (female) - 1,132 sentences each, 16 kHz, with phone labels | 460 MB | CMU ARCTIC free license (any use, keep the notice) |
| CMU ARCTIC packed | aew (male); eey, ljm, lnh (female) - WAV and text only | 260 MB | CMU ARCTIC free license |
| [LibriTTS-R](https://www.openslr.org/141/) | LibriVox audiobook readers, 24 kHz | | CC BY 4.0 |
| [LibriTTSLabel](https://github.com/kan-bayashi/LibriTTSLabel) | Montreal Forced Aligner phone alignments of LibriTTS | | none stated - used locally, not redistributed |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | 91 actors' emotional speech (`picovox_train.cremad fetch`) | | ODbL |

The four labeled ARCTIC speakers' analysis settings (pitch range and formant ceiling, `data/targets/arctic-*.json`) were
measured by the rule-based synthesizer's calibration tool, `tools/targets.py` (removed in commit 33fbfad; see git
history); `picovox_train/data.py` featurizes their recordings with them.


## License

PicoVox is distributed under the [MIT License](LICENSE). The CMU Pronouncing Dictionary, the G2P network's training
labels (in the repository, not the package), is distributed under its own license, in `data/LICENSE.cmudict`.
