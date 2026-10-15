# voice-synth

A from-scratch speech synthesizer in JavaScript, aiming for realistic male and female voices - no recorded speech. Text
goes through a front end (normalization, the CMU Pronouncing Dictionary) to frames of synthesizer parameters every 5 ms,
which drive a source-filter synthesizer with an LF glottal source. The parameters come either from rules (Klatt's
duration rules, intonation, formant targets - `lib/`) or from a small neural network trained through a differentiable
copy of the synthesizer (`nn/`); the audiobook player uses the neural voices.

It runs under Node (command-line tools and evaluation) and in the browser (Web Audio, synthesis in a Web Worker). There
are no dependencies.

The paper [Small Voices: Browser Speech Synthesis for Audiobooks](paper/paper.md) describes the methods and results -
the rule-based synthesizer and its calibration, the neural voices that drive it, and the work in progress. On the
published site it reads at `paper/` (rendered with [MarkdownUp](https://craigahobbs.github.io/markdown-up/)).


## The audiobook player

A public-domain audiobook player, read aloud by the synthesizer - [open the audiobook player](static/audiobook/). Serve
the repository root statically and open `static/audiobook/` (the root `index.html` shows this README):

```sh
python3 -m http.server 8000
open http://localhost:8000/static/audiobook/
```

To listen on a phone, open `http://<your computer's LAN address>:8000/` on it (same network).

- **Books** come live from two sources, straight from the browser (both send CORS headers, so no server or proxy is
  needed): [Standard Ebooks](https://standardebooks.org) - about 1,300 carefully produced classics, downloaded as EPUBs
  and unzipped in the browser - and [Project Gutenberg](https://www.gutenberg.org) - its catalog through
  [Gutendex](https://gutendex.com) and its texts through the [GITenberg](https://github.com/GITenberg) mirror on GitHub
  (gutenberg.org's own text files don't send CORS headers; GITenberg is a 2016 snapshot, so newer books are missing).
- **The player** has play/pause, back and forward 30 seconds, the book's contents, bookmarks (add, name, go to,
  delete), and the narrator's voice and speed. It plays on from the library (with a mini player), and works with the
  lock screen and media keys.
- **Your place is kept** - every book's position, its bookmarks, and the open book are saved in browser storage as you
  listen, so reloading, closing the tab, or coming back days later picks up at the same sentence. Downloaded books are
  cached in IndexedDB, so reopening one is instant (and works offline).
- **Long books are cheap**: the player's timeline is the book's text, split into sentences (long ones at semicolons and
  colons) that are synthesized on demand a few minutes ahead of the position, and only the audio near the position is
  kept - so playback starts in about 0.1 s anywhere in a book, and Moby Dick costs no more memory than a chapter.

The page needs HTTP - it does not work from a `file:` URL, since module workers and fetch require it. The worker fetches
`data/cmudict.dict` (3.6 MB) on startup.

| Module | |
| --- | --- |
| `static/audiobook/app.js` | The library and player interface |
| `static/audiobook/engine.js` | Playback - the text timeline, on-demand synthesis, scheduling, skipping, speech normalization |
| `static/audiobook/worker.js` | The synthesis Web Worker - speaks the player's units (sentences) with the neural voices |
| `static/audiobook/sources/standardebooks.js` | Standard Ebooks - browse, search, and EPUB loading |
| `static/audiobook/sources/gutenberg.js` | Project Gutenberg - Gutendex browse and search, GITenberg texts, boilerplate and chapters |
| `static/audiobook/store.js` | The listener's state in browser storage - places, bookmarks, settings |
| `static/audiobook/bookcache.js` | Downloaded books in IndexedDB |

`node tools/browsercheck.js` checks the browser build under Node: it loads the app-time modules (`lib/`, `static/audiobook/worker.js`,
`nn/js/`, and the generated model module) with `node:` imports forbidden, speaks with the worker's functions from a
dictionary string, and checks the audio matches the same neural pipeline run directly, sample for sample.


## The catalog and the published site

The app is published as static files (GitHub Pages). The site holds only a catalog of books - their metadata, in
`data/catalog.json` - and each book's full text is fetched from its source at play time, with CORS.

```sh
make catalog                          # rebuild data/catalog.json (GUTENBERG_COUNT=1000 by default)
make catalog GUTENBERG_COUNT=2000     # consider more of Project Gutenberg's most-downloaded books
make site                             # assemble the published site in build/site/
make gh-pages                         # test, lint, and cover, then sync build/site/ to ../voice-synth.gh-pages
```

`make catalog` runs `node tools/catalog.js`, which reads every [Standard Ebooks](https://standardebooks.org) book (its
popularity listing and each book's page) and the top Project Gutenberg books by downloads from
[Gutendex](https://gutendex.com). Gutenberg books that duplicate a Standard Ebooks book are left out, and each
remaining book's plain text is found in the [GITenberg](https://github.com/GITenberg) mirror on GitHub (whose
`raw.githubusercontent.com` files send CORS headers) with the [gitberg](https://github.com/gitenberg-dev/gitberg)
project's repository list and HEAD requests; books not in GITenberg are left out. Requests are throttled, and every
fetched page is cached under `out/catalog/`, so a rerun takes seconds and an interrupted build resumes (`--max-age
<days>`, 30 by default, refetches older pages). For books missing from the gitberg list, the GitHub search API is used -
10 searches a minute, or many more with a `GITHUB_TOKEN` in the environment.

A full build from an empty cache takes about 45 minutes: Standard Ebooks' ~1,500 book pages at 2 a second (about 12
minutes), and Gutendex, which can take a minute per page of 32 books (about 30 minutes for 1,000). To finish a partial
catalog, rerun `make catalog` (with `GITHUB_TOKEN=<token>` set to speed up any GitHub searches) - it picks up from the
cache and resolves only the books it hasn't yet. `make catalog CATALOG_ARGS=--cache-only` rebuilds the catalog from the
cache alone, without network requests (books not yet resolved are left out).

The catalog format (`version` 1):

```js
{
    'version': 1, 'generated': '<ISO time>',
    'sources': {'standardebooks': {'name', 'url', 'description'}, 'gutenberg': {'name', 'url', 'description'}},
    'books': [{
        'source': 'standardebooks' | 'gutenberg',
        'id': '/ebooks/herman-melville/moby-dick',  // Standard Ebooks: the book's path; Gutenberg: its number, as a string
        'title': 'Moby Dick', 'subtitle': 'Or, The Whale',  // the subtitle is optional
        'author': 'Herman Melville',                        // display names, several joined with ", "
        'description': '<plain text, at most about 600 characters>',
        'subjects': ['Fiction'],                            // at most 8
        'language': 'en',
        'cover': '<absolute image URL, or null>',
        'rank': 1,                                          // 1-based popularity rank within its source
        'words': 211202,                                    // approximate, or null
        'text': {'url': '<absolute CORS URL of the full text>', 'format': 'epub' | 'gutenberg-text',
                 'encoding': 'utf-8' | 'windows-1252'}      // the encoding is for gutenberg-text only
    }]
}
```

`lib/catalog.js` browses it in the app: `listBooks(catalog, source)` (by rank; `'all'` interleaves the sources),
`searchBooks(catalog, query, source)` (every query word must start a title, subtitle, author, or subject word; title and
author matches first), `findBook(catalog, source, id)`, `pageItems(items, page, pageSize)`, and `normalizeText(text)`.


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

The training dashboard, `tools/dashboard/`, follows the neural voices' training as it runs: serve the repository
(`python3 -m http.server`), start its monitor (`node tools/dashboard/monitor.js`), and open
http://localhost:8000/tools/dashboard/. It has four tabs (each linkable, e.g. `#listen`): Progress, Listen, Training,
and Logs. Progress is the pipeline in `tools/dashboard/pipeline.json`: each phase is a
group with its progress, what's happening now, and how many of its steps are done, and it opens to its rows - the
workflow's agents with their latest actions, the planned steps (each tracked by a log, the newest training run, an
output file, and its process), and unplanned work since the pipeline started (other logs, runs, and processes), each
with its progress bar and latest line. Rows are never removed. The overall progress is weighted by each step's expected
time. Listen plays the checkpoint samples (below) and other samples in `build/nn/`, and marks a new checkpoint until
it's heard. Training has the runs in `build/nn/runs/` (their `log.jsonl` and `train.log`) with charts of their losses,
validation, pitch error, and step times. Logs has the tails of the logs in `build/nn/`. It refreshes every second,
read-only, re-fetching only what changed, and reloads itself when its own code changes.

The monitor writes `build/nn/activity.json` every second: the Python and Node processes working in the repository for
10 seconds or more (with their multiprocessing workers folded in), running now or before, and the workflow's agents
(`pipeline.json`'s workflow, else the newest workflow in `~/.claude/projects/`) - so the dashboard shows the steps that
write no log. It also renders each new checkpoint of the pipeline's runs (`build/nn/runs/<run>/step-*.pt`) with
`pipeline.json`'s sample speakers and text, scores the samples with UTMOS on the CPU (leaving the GPU to training), and
lists them in `build/nn/samples-progress/index.json` with today's app voices as a reference; the dashboard's Checkpoint
samples section plays them, newest first.


## The listener panel

Neural listener models score the synthesizer the way a panel of listeners would - a repeatable check of naturalness,
speaker identity, and speaking style alongside Whisper's intelligibility. They run in a Python environment of their own
(PyTorch), used only by the tools - the synthesizer and the app have no dependencies.

```sh
make listeners                                              # build/venv-listeners from requirements/listeners.txt
node tools/panel.js -v male -v female -v samantha           # MOS, WER, speaker consistency, emotion, similarity
node tools/panel.js -v male -o '{"f0": 95}' -v male --show  # a variant against the voice, per-text details
node tools/panel.js -v female --style '{"scale": {"f0": 1.2, "rate": 0.9}}'
node tools/panel.js -v male --ref data/corpora/arctic/cmu_us_bdl_arctic/wav \
    --ref data/corpora/libritts_r/LibriTTS_R/dev-clean/174     # against human recordings
node tools/panel.js -v male --json > out/male-panel.json   # machine-readable, every file's scores
```

| Listener | Model | Measures |
| --- | --- | --- |
| Intelligibility | Whisper `base.en` (whisper.cpp, as `tools/eval.js`) | Word error rate against the text |
| Naturalness | [UTMOS22 strong](https://github.com/sarulab-speech/UTMOS22) (through torch.hub [tarepan/SpeechMOS](https://github.com/tarepan/SpeechMOS)) | Predicted mean opinion score, 1-5 |
| Speaker identity | [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (SpeechBrain, VoxCeleb) | A 192-dimension speaker embedding |
| Style | [emotion2vec+ base](https://huggingface.co/emotion2vec/emotion2vec_plus_base) (FunASR) | Probabilities of angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown |

`tools/panel.js` synthesizes a fixed evaluation set, `tools/panel-texts.json` - six Harvard sentences (none from
`tools/eval.js`'s tuning set), six narrative passages from public-domain books with dialogue (Austen, Melville,
Dickens, Doyle, Potter, Grahame), two questions, and two exclamations - with each voice (`-v`: a voice name, a voice
spec JSON file, or an Apple system voice - samantha, daniel, fred, ...), and reports for each:

- **MOS** - the mean, median, and minimum UTMOS, and the mean by kind of text
- **WER** - Whisper's word error rate
- **spkr** - speaker consistency, the mean cosine similarity of each utterance's embedding to the voice's mean (a
  stable voice is above about 0.8)
- **emotion** - the top mean emotion probabilities - the check that a style reads as the intended emotion
- **similarity** - with several voices, the cosine similarity matrix of their mean speaker embeddings: distinctiveness.
  Halves of one human speaker's recordings score about 0.9; different human speakers 0.0-0.5.

`-o` (overrides JSON) and `--style` (a style JSON object or file - see `applyStyle` in `lib/voices.js`) apply to the
`-v` before them, or to every voice when given first. `--ref <wav or directory ...>` adds reference recordings - human
speech - to the table and the similarity matrix (a directory gives its first `--ref-count` WAV files, 16 by default;
ARCTIC and LibriTTS transcripts are found for their WER; `--ref-label` names the group). `-n`, `--kind`, and `-s`
choose the texts and the seed; `--no-wer` and `--no-emotion` skip listeners.

Everything is cached under `out/panel/`: synthesized audio by a hash of the voice, the options, the text, and the
library code and dictionary (`out/panel/audio/`), and the Whisper transcripts and listener scores by a hash of the
audio itself (`out/panel/scores/`) - so a rerun is instant, and a refactor that leaves the audio bit-identical costs
only the synthesis.

`tools/listeners.py` is the scoring tool itself - WAV files of any sample rate in, one JSON line per file out:

```sh
build/venv-listeners/bin/python -I tools/listeners.py [--no-mos] [--no-embedding] [--no-emotion] [--device auto|cpu|mps] \
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
| voice-synth female | 3.27 | 2.69 | 8.5% |
| voice-synth male | 3.18 | 2.88 | 11.5% |
| Apple Fred (1980s formant synthesis) | 2.81 | 2.52 | 10.6% |

Speaker similarity: voice-synth male and female 0.66 (one speaker with a pitch and formant shift, to ECAPA), male and
Fred 0.57 (a shared formant-synthesis signature), every human pair 0.0-0.48, and male against male at `f0` 95 Hz 0.99
(F0 alone does not change the perceived speaker).


## Distilling the voice model

Each retrain of the multi-speaker acoustic model ends the same way: train a big teacher, then distill small students
from it and ship the smallest one that keeps the teacher's quality. `nn/distill.py` runs that pipeline, with its
settings in `data/nn/distill.json` (the teacher, the distillation set, the training settings, the student ladder, the
polish, the evaluation speakers, and the tolerance):

```sh
build/venv-nn/bin/python -B -m nn.distill ladder                  # everything below, in order
build/venv-nn/bin/python -B -m nn.distill targets                 # 1. the teacher's targets (about 10 minutes on the GPU)
build/venv-nn/bin/python -B -m nn.distill train --student s64     # 2. each student of the ladder
build/venv-nn/bin/python -B -m nn.distill polish --student s64    # 3. optional: a short fine-tune on the recordings
build/venv-nn/bin/python -B -m nn.distill eval teacher s96 s64 s48 s32   # 4. each model through the browser path
build/venv-nn/bin/python -B -m nn.distill pick                    # 5. speed, samples, and the smallest within tolerance
```

1. **Targets** - sequence-level distillation in parameter space. The teacher speaks (tokens, speaker, style) to its
   durations (pauses included) and its 32 synthesizer parameters every 5 ms, as inference does, with no audio
   rendered. The inputs are the training utterances with their own speakers and styles, and every sentence of the
   Project Gutenberg books in `build/nn/g2p/books/` (tokenized by `nn/frontend.py`, split into sentences as inference
   splits text) spoken twice, each time by a random reader (8% an emotional-corpus actor, 10% a blend of two readers)
   in a neutral or, 30% of the time, an emotional style at a random intensity. The set goes to
   `build/nn/distill/targets/` (frames as float16).
2. **Students** - the same `MultiModel` with smaller widths and depths. They share the teacher's conditioning: its
   speaker, style, and channel-equalizer tables are copied and frozen, so a voice designed as a blend of the teacher's
   speakers is the same voice in a student. A student runs with the teacher's durations (teacher forcing on its length
   regulator) and learns L1 on the normalized parameter frames plus the duration losses against the teacher's
   duration outputs. `--init-teacher` copies every teacher tensor of the same shape; `--feature-weight` adds
   intermediate-feature matching. Runs go to `build/nn/runs/distill-<student>/`.
3. **Polish** - `nn/trainmulti.py`'s real-audio losses (the spectral loss through the differentiable synthesizer on
   the recordings, the adversarial critic - started from the teacher's - and feature matching) for a short fine-tune.
4. **Evaluation** - each model is exported (`nn/export.py`), packaged (`nn/package.py`, the generated module's bytes
   and gzip bytes), and spoken by `nn/js/speak.js` on `tools/panel-texts.json` for six speakers; the renderings are
   scored with UTMOS, Whisper's WER, and the ECAPA speaker similarity to the teacher's rendering of the same speaker.
   `pick` measures `nn/js/speak.js`'s real-time factor, renders listening samples to `build/nn/distill-samples/`, and
   chooses the smallest student within the tolerance - mean UTMOS within 0.05 of the teacher's, WER at most 1 point
   worse, and speaker similarity at least 0.85 - writing `build/nn/distill/summary.json`.

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

The polish adds 0.025-0.034 UTMOS, most of it to the two ARCTIC voices the app uses (bdl and slt), which pure
distillation loses most; intermediate-feature matching (weight 0.05) adds about 0.02, and the two barely stack.


## The library

| Module | |
| --- | --- |
| `lib/glottal.js` | LF glottal pulse, parameterized by Rd |
| `lib/klatt.js` | Klatt-style cascade synthesizer - voicing, aspiration, nasal pole/zero, F1-F6 and higher formants, frication |
| `lib/phonemes.js` | ARPAbet tables - vowel formants, consonant loci and couplings, noise spectra |
| `lib/lexicon.js` | Text normalization, the pronouncing dictionary, function words |
| `lib/voices.js` | Voice specs - the documented voice parameters, the voices, inheritance, and speaking styles |
| `lib/speak.js` | Durations, stop phases, formant transitions, smoothing, intonation - `speak(text, voice, options)` |
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


## Voice specs

A voice is data - a spec in `lib/voices.js`. `voiceParameters` documents every synthesis parameter (its group, units,
typical range, meaning, and default), and the defaults are the root voice, `baseVoice` (the reference adult male
speaker of the phoneme tables). A spec is metadata - `name`, `label`, `description`, and `base` (the name or spec it
inherits from; the root voice if none) - and parameter overrides. So a new voice is a few overrides of a base:

```js
import {speak} from './lib/speak.js';

const baritone = {
    'name': 'baritone', 'label': 'Baritone', 'description': 'A deep, unhurried storyteller',
    'base': 'male', 'f0': 96, 'tractScale': 0.95, 'accent': 0.3, 'rate': 1.05, 'reduction': 0.8, 'sentencePause': 420
};
const {samples, sampleRate} = speak('Call me Ishmael.', baritone);
```

`speak(text, voice, options)` takes a voice name (`'male'`, `'female'`) or a spec. The parameter groups:

| Group | Parameters |
| --- | --- |
| pitch | The F0 baseline (`f0`, `pitchScale`) and range (`accent`), and the intonation style - accent placement (deaccented given words, accent spacing), phrase onset, question and exclamation, final fall, declination and its reset at commas, downstep, continuation, and the slow, aperiodic pitch drift |
| source | Voice quality - the LF pulse shape (`rd`), breathiness (`breathDb`, and `noiseShelfDb`, the breath noise's high-frequency turbulence), jitter, shimmer, F0 flutter, creak, spectral tilt (`shelfDb`), the louder accents (`f0Loudness`), and `voicing` (0 whispers) |
| tract | The vocal tract - its length (`tractScale` scales every resonance), the vowel space (`vowelBlend` - 0 the men's average formants, 1 the women's), the consonant loci and noise scales, the fixed higher formants, and the nasal and lateral resonances |
| articulation | Precision - vowel `reduction` and `transitionScale` (coarticulation) - and the burst, aspiration, frication, and liquid levels |
| timing | The speaking `rate`, the pauses (comma - growing with the phrases' lengths - semicolon and colon, sentence, intermediate phrase, leading, and final), and phrase-final and unstressed-vowel durations |

`resolveVoice(voice, {overrides, style, rate})` merges the root voice, the spec's base chain, the spec, the overrides,
and the style into the complete parameters (`speak` returns them as `voice`); `validateVoice(spec)` lists a spec's
unknown parameters, wrong types, and values outside the typical ranges.

The speak options are `seed`, `rate` (a multiplier of the voice's), `overrides` (parameters, unchecked - for
experiments and the evaluation tools' `-o`), `style`, and `onBlock` (streaming).

### Speaking styles

A style transforms a voice's parameters - `{'intensity', 'scale', 'add', 'set'}`, each of `scale`, `add`, and `set` an
object of parameters. At full intensity a style multiplies, adds to, or replaces the parameters; at a lower intensity
(0 to 1, default 1) it is blended: `scale` geometrically (`value * factor ** intensity`), `add` and numeric `set`
linearly, and boolean `set` switches at one half. An array of styles applies in order. For example, a hypothetical
whisper:

```js
const whisper = {'scale': {'accent': 0.6}, 'add': {'breathDb': 28}, 'set': {'voicing': 0, 'creak': 0, 'glottalAttack': false}};
speak('Do not wake the baby.', 'female', {'style': {...whisper, 'intensity': 0.8}});
```


## Human speech targets

The voices are calibrated against statistics of real American English speech, measured identically for a human
speaker and for the synthesizer - so a voice can be fit to a target speaker, and checked against the range of human
speakers. The analysis runs in a Python environment of its own (Praat through
[parselmouth](https://parselmouth.readthedocs.io), numpy, scipy, soundfile), used only by the tools:

```sh
make build/venv-analysis.build                    # build/venv-analysis from requirements/analysis.txt
# or directly: python3 -m venv build/venv-analysis && build/venv-analysis/bin/pip install -r requirements/analysis.txt
build/venv-analysis/bin/python -I tools/corpora.py              # download the corpora (about 2.1 GB) into data/corpora/
build/venv-analysis/bin/python -I tools/targets.py all          # every corpus speaker -> data/targets/<speaker>.json
node tools/synthstats.js -v male --target data/targets/arctic-bdl.json   # our voice on the same prompts, measured and compared
build/venv-analysis/bin/python -I tools/targets.py population out/synthstats/synth-male.json  # among the human speakers
```

**The corpora** (`tools/corpora.py` downloads and unpacks them idempotently and sequentially, resuming interrupted
downloads; `--list` shows them; `data/corpora/` is gitignored and its `README.md` records each corpus's source, license,
and size):

| Corpus | Speakers | Download | License |
| --- | --- | --- | --- |
| [CMU ARCTIC](http://festvox.org/cmu_arctic/) 0.95 releases | bdl, rms (male); slt, clb (female) - 1,132 sentences each, 16 kHz, with phone labels | 460 MB | CMU ARCTIC free license (any use, keep the notice) |
| CMU ARCTIC packed | aew (male); eey, ljm, lnh (female) - WAV and text only | 260 MB | CMU ARCTIC free license |
| [LibriTTS-R](https://www.openslr.org/141/) dev-clean | 40 LibriVox audiobook readers (20 female, 20 male), 24 kHz | 1.36 GB | CC BY 4.0 |
| [LibriTTSLabel](https://github.com/kan-bayashi/LibriTTSLabel) | Montreal Forced Aligner phone alignments of LibriTTS (dev-clean unpacked) | 178 MB | none stated - used locally, not redistributed |

The ARCTIC speakers are American; LibriTTS-R has no accent metadata (its readers are mostly, not all, American). The
speakers' sex comes from the corpus metadata (LibriTTS-R's `speakers.tsv`) - one LibriTTS reader (7976) is listed as
male with a 200 Hz median F0, and `population` leaves out such conflicts.

**The measures** (`tools/targets.py extract`; all audio is first resampled to 16 kHz, ARCTIC's rate):

- **F0** - Praat autocorrelation pitch with a two-pass speaker-adaptive range (floor 0.75 x the first quartile, ceiling
  1.5 x the third): the median, mean, and 5th and 95th percentiles in Hz; the spread and 5-95% range in semitones; and
  per utterance, the range, declination (the slope of a line fit, semitones per second), the onset and the
  statement-final fall relative to the utterance median, and the F0 velocity (median absolute rate of change).
- **Vowel formants** - F1-F4 per vowel (Burg, 5 formants) at 50% (the median of 5 frames) and F1-F2 at 20% and 80%
  (the onset and offset - diphthongs and vowel-inherent spectral change), with the formant ceiling chosen per speaker
  as the one minimizing within-vowel variance (Escudero et al., 2009); the vowel quadrilateral area (IY, AE, AA, UW),
  the formant dispersion, and the apparent vocal tract length.
- **Timing** - per-phone and per-class mean durations, the articulation rate (syllables and phones per second, less
  pauses), and phrase-final lengthening (final vowels' durations relative to their vowel's mean).
- **Pauses** - found acoustically (runs of at least 100 ms more than 25 dB below the utterance's 95th-percentile
  intensity, inside the speech), since aligners often fold pauses into neighboring phones: per minute, per punctuation
  mark, the pause fraction, and the mean, median, and 90th-percentile durations.
- **Spectrum** - the long-term average spectrum of the speech (and of the vowels) in 15 bands from 63 Hz to 8 kHz,
  relative to the total; the alpha ratio, Hammarberg index, and slope (dB per octave, 0.5-6 kHz).
- **Voice quality** at vowel middles - H1-H2 and H1*-H2* (corrected for F1 and F2, Iseli and Alwan 2004, with Hawks
  and Miller bandwidths), H1-A1, H1-A3, H1*-A3*, HNR, jitter and shimmer (local), and cepstral peak prominence.
- **Intensity** - the speech frames' spread and 5-95% range, the declination of the vowels' levels, and each phone
  class's level relative to the full vowels (and the reduced vowel's).
- **Fricatives** - the spectral center of gravity of each fricative's middle 20 ms.

Phone-based measures need phone labels - ARCTIC's own (xwaves, 10 ms resolution), LibriTTSLabel's alignments (stress
removed, AH0 as the reduced vowel AX), or the synthesizer's exact segments; unlabeled speakers (aew, eey, ljm, lnh) get
the utterance-level measures only. Speakers with fewer than 40 utterances are skipped. A target file is about 12 KB.

**Our speech, measured the same way** - `node tools/synthstats.js` speaks the ARCTIC prompts (`-v` a voice name or a
JSON voice spec, `--overrides`, `--style`, `-r`, `-s`, `-n` count, `-j` worker threads) into `out/synthstats/<name>/`
in the ARCTIC layout - 44.1 kHz WAVs, and labels written from `speak()`'s segments (pauses as `pau`, unstressed AH as
`ax`) - then runs `tools/targets.py extract` on it, writing `out/synthstats/<name>.json`. It also checks the
measurement: Praat's vowel formants against the synthesizer's own formant targets at the vowel middles (the median
absolute error is about 8 Hz for F1 and F2 and 10-25 Hz for F3), and with `--target` it prints the comparison.

**Comparing** - `tools/targets.py compare ours.json target.json` prints every measure (Hz differences in semitones,
formants in Bark, durations as log ratios after removing the overall tempo difference, levels in dB), each divided by
a scale of a clearly audible difference, and the weighted root-mean-square of those - a single distance for fitting
(`--brief` prints only the distances and the largest discrepancies, `--json` everything). `tools/targets.py population
ours.json` places a voice among the human speakers of its sex instead - each measure's z-score against their mean and
spread - which separates what is merely another speaker's voice (F0 level, vowel space) from what no human speaker does.

Runtimes: the corpora download in about 10 minutes; `targets.py all` measures the 43 speakers in about 13 minutes on
16 cores (about 17 s for an ARCTIC speaker's 1,132 sentences); `synthstats.js` synthesizes the 1,131 prompts (56 minutes
of speech, 290 MB of WAVs) in about 35 s with 8 threads, and measures them in about 20 s.


## License

The CMU Pronouncing Dictionary is distributed under its own license, in `data/LICENSE.cmudict`.
