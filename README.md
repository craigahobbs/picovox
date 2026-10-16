# voice-synth

A from-scratch speech synthesizer in JavaScript, aiming for realistic male and female voices - no recorded speech at
runtime. Text goes through a front end (normalization, and a small grapheme-to-phoneme network for every word) to
tokens; a small neural network (`nn/js/model.js`, trained through a differentiable copy of the synthesizer - `nn/`)
predicts frames of interpretable synthesizer parameters every 5 ms, which drive a source-filter synthesizer with an LF
glottal source (`nn/js/render.js`). The generated model modules are in `lib/models/`.

It runs under Node (command-line tools and evaluation) and in the browser (Web Audio, synthesis in a Web Worker). There
are no runtime dependencies. (The first, rule-based formant synthesizer - Klatt's duration rules, intonation, formant
targets, the "classic voices" - and its analysis tools were removed in commit 33fbfad; see git history.)

The paper [Small Voices: Browser Speech Synthesis for Audiobooks](paper/paper.md) describes the methods and results -
the rule-based synthesizer and its calibration, the neural voices that replaced it, and the work in progress. On the
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

The page needs HTTP - it does not work from a `file:` URL, since module workers and fetch require it. The player loads
the generated G2P and voice model modules (`lib/models/`) and passes them to the worker.

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
node nn/js/speak.js -m data/nn/slt "Hello, world." -o out/hello.wav    # an exported model (data/nn/<voice>.json and .bin)
node nn/js/speak.js -m data/nn/multi "Hello." -o out/hello.wav --voice narrator-f   # a designed voice (data/nn/voices.json)
node nn/js/voicescheck.js [data/nn/multi]                               # check the designed voices
node tools/browsercheck.js                                              # the browser build under Node
node tools/heteronyms.js                                                # the heteronym regression set
```

`nn/js/speak.js` speaks with an exported model (`nn.export CHECKPOINT --voice NAME` writes `data/nn/NAME.json` and
`.bin`; `data/nn/multi` is an export of the multi-speaker model; `-r` rate, `-s` seed, `--params` the parameter frames,
`--batch` a list of items); the multi-speaker model takes a speaker (`--speaker`, `--blend`), a style (`--style`,
`--intensity`), or a designed voice and style (`--voice`, `--voice-style` - `nn/js/voices.js`).

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
bookkeeping) is inferred from its stamps and make's log (`pipeline.json`'s `train.log`). Listen plays the checkpoint
samples (below) and other samples in `build/nn/`, and marks a new checkpoint until it's heard. Runs has the runs in
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
lists them in `build/nn/samples-progress/index.json` with today's app voices as a reference; the dashboard's Checkpoint
samples section plays them, newest first.


## The listener panel

Neural listener models score the synthesizer the way a panel of listeners would - a repeatable check of naturalness,
speaker identity, and speaking style alongside Whisper's intelligibility. They run in a Python environment of their own
(PyTorch), used only by the tools - the synthesizer and the app have no dependencies.

| Listener | Model | Measures |
| --- | --- | --- |
| Intelligibility | Whisper `base.en` (whisper.cpp, `whisper-cli`, the model in `data/whisper/`) | Word error rate against the text |
| Naturalness | [UTMOS22 strong](https://github.com/sarulab-speech/UTMOS22) (through torch.hub [tarepan/SpeechMOS](https://github.com/tarepan/SpeechMOS)) | Predicted mean opinion score, 1-5 |
| Speaker identity | [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (SpeechBrain, VoxCeleb) | A 192-dimension speaker embedding |
| Style | [emotion2vec+ base](https://huggingface.co/emotion2vec/emotion2vec_plus_base) (FunASR) | Probabilities of angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown |

`nn/panel.py` is the panel's scoring, shared by every evaluation (`nn.distill eval`, make train's gate): an exported
model speaks a fixed evaluation set, `tools/panel-texts.json` - six Harvard sentences, six narrative passages from
public-domain books with dialogue (Austen, Melville, Dickens, Doyle, Potter, Grahame), two questions, and two
exclamations - through `nn/js/speak.js`, and the renderings are scored with UTMOS, Whisper's WER, and the ECAPA
speaker similarity.

`tools/listeners.py` is the scoring tool itself - WAV files of any sample rate in, one JSON line per file out:

```sh
make listeners                                              # build/venv-listeners from requirements/listeners.txt
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
| Apple Fred (1980s formant synthesis) | 2.81 | 2.52 | 10.6% |

Speaker similarity: every human pair 0.0-0.48; halves of one human speaker's recordings about 0.9.


## make train

`make train` recreates the generated models - `lib/models/g2pModel.js` and `lib/models/voiceModel.js` - from a clean
checkout, by one recipe, `data/nn/train.json` (every stage's settings and seeds). Each stage is a stamp in
`build/train/` (`TRAIN_DIR`), so `make -j train` runs independent stages at once; the GPU stages run one at a time
(`nn/gpulock.py` - two at once each run at about half speed) while the CPU stages run beside them:

| Stage | Runs on | Needs |
| --- | --- | --- |
| corpora - CMU ARCTIC, LibriTTS-R and its alignments, CREMA-D | network | |
| g2p - the G2P retrain, checked to reproduce `lib/models/g2pModel.js` | 1 CPU core, ~30 min | |
| heteronyms - the relabeled words, checked to reproduce `data/nn/g2p-relabel.json` | CPU, ~6 min | corpora |
| data-multi - the readers' training data (the pool: `data/nn/speakers.json`) | CPU (every core) | corpora, `lib/models/g2pModel.js` |
| cremad-align, data-cremad - the emotional speech | GPU 2 min, CPU 1 min | corpora |
| teacher-a, teacher-b - the teacher (`nn.trainmulti --recipe teacher.a`, `teacher.b`) | GPU | the data |
| teacher-eval - the teacher through the listener panel | CPU | teacher-b |
| targets, student, polish - the distilled student (`nn.distill`) | GPU | teacher-b |
| student-eval - the panel and the gate (`nn.distill pick`: within the tolerance) | CPU | polish, teacher-eval |
| package - `lib/models/voiceModel.js`, with its provenance | CPU | student-eval |

The critical path - the readers' data, teacher A (~22 minutes) and B (~60), the targets (~9), the student (~11) and
its polish (~9), the gate, the package - takes about 2 hours on an M3 Max (the corpora downloaded once before). The
voice data is built with the committed G2P, so the G2P retrain is off the critical path; if the retrained G2P
differs, its stage updates `lib/models/g2pModel.js` and fails, and `make train` again rebuilds the data with it. The
package's manifest records its provenance: the commit, the recipe's and the training data's hashes, and the gate's
scores. `python -m nn.smoke` runs every stage's entry point for a few steps on the CPU.

`make train` keeps a record for the training dashboard (`tools/dashboard/`): each stage's start, end, and failure in
`TRAIN_DIR/stages.jsonl` and each command's output also in `TRAIN_DIR/logs/STAGE.log` (`tools/trainstage.sh` - the
teacher's also in its run's `train.log`), and every 20 seconds, for the life of `make train`, the CPU busy %, the GPU
utilization %, the memory used %, and the running stages in `TRAIN_DIR/util.log` (`tools/trainutil.sh`). The stages,
their dependencies, and their expected times are `data/nn/train-stages.json` (`test/testTrainStages.js` keeps it in
step with the Makefile). `make train` never removes anything; `make train-clean` removes `TRAIN_DIR` for a run from
scratch.

`make train` first checks the power (`tools/trainpower.sh`, macOS laptops): the run keeps the GPU busy for hours, and
an M3 Max under it draws about 85 W, so on battery or an adapter under 96 W (`TRAIN_MIN_WATTS`) it says so and, at a
terminal, waits for Enter - plug in a bigger adapter first (Apple's 96 W or 140 W). `TRAIN_POWER_CHECK=0` skips it.


## Distilling the voice model

Each retrain of the multi-speaker acoustic model ends the same way: train a big teacher, then distill small students
from it and ship the smallest one that keeps the teacher's quality. `nn/distill.py` runs that pipeline, with its
settings in `data/nn/train.json`'s `distill` (the teacher, the distillation set, the training settings, the student
ladder, the polish, the evaluation speakers, and the tolerance):

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


## Training the G2P model

The app pronounces every word with one small network, `lib/models/g2pModel.js` (`lib/g2p.js` runs it;
`nn/g2pmodel.py` is its bit-exact Python port, which pronounces the voice model's training data). `nn/g2p.py` trains
and exports it, with the recipe in `data/nn/train.json`'s `g2p` settings:

```sh
build/venv-nn/bin/python -B -m nn.heteronyms --out build/nn/heteronyms  # data/nn/g2p-relabel.json's words (optional)
build/venv-nn/bin/python -B -m nn.g2p train --out build/nn/g2p-train      # four stages, about 30 minutes on one CPU core
build/venv-nn/bin/python -B -m nn.g2p export build/nn/g2p-train --out lib/models/g2pModel.js
build/venv-nn/bin/python -B -m nn.g2p evaluate                            # the book-token error of lib/models/g2pModel.js
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

`nn/heteronyms.py` chooses the relabeled words from the LibriTTS-R readers' usage: for each word with several
dictionary pronunciations, the variant the readers used by the forced alignments; a word is relabeled when the
readers' majority isn't the first (significant, 3+ speakers), the difference is lexical (not vowel reduction or stress
alone, not a function word), and the vowel's formants (or the fricative's voicing) agree where measured.

### Heteronyms

Some words have readings their neighbors decide - "I read it yesterday" and "I will read it", "their lives" and "she
lives", "a lead pipe", the noun/verb stress pairs. The G2P module also holds two small pieces for them (no word list or
rule at runtime): the G2P takes a reading index (a learned vector added to its decoder's first state; reading 0 is the
default) and has an ambiguity head (whether the word's reading depends on its context); for an ambiguous word, a context
network (`nn/context.py`: a character GRU over the three words and phrase-end marks each side, a bidirectional GRU, and a
layer gated by the word itself - 47,779 parameters at 6 bits) chooses the reading.

```sh
build/venv-nn/bin/python -B -m nn.contextlabels books                                # the recipe's books (by Gutenberg id)
build/venv-nn/bin/python -B -m nn.contextlabels aligned --out build/nn/context      # readers' readings (MFA alignments)
build/venv-pos/bin/python -B -m nn.contextlabels tag --out build/nn/context         # part-of-speech tags (spaCy)
build/venv-nn/bin/python -B -m nn.contextlabels labels --out build/nn/context       # the ambiguous words and examples
build/venv-nn/bin/python -B -m nn.context train-g2p --labels build/nn/context/labels.json --init build/nn/g2p-train/focus.pt \
    --out build/nn/context/g2p.pt                                                   # the reading input and ambiguity head
build/venv-nn/bin/python -B -m nn.context train-context --labels build/nn/context/labels.json --out build/nn/context/context.pt
build/venv-nn/bin/python -B -m nn.context export --g2p build/nn/context/g2p.pt --context build/nn/context/context.pt
node tools/heteronyms.js                                                            # the regression set (exit 1 below the gate)
build/venv-nn/bin/python -B -m nn.context cross-validate --labels build/nn/context/labels.json --out build/nn/context/folds
```

The settings are `data/nn/train.json`'s `context` section. `make train` runs all of it beside the voice stages (which
use the committed `lib/models/g2pModel.js`) and checks its module with `nn.context check`: identical to the committed
module below the comment header (the stages are deterministic on the same inputs), else within the recipe's tolerances
(the regression set, the book error, the top-10K, the monitoring WER) - it never replaces the committed module. A clean
run reproduces the committed module only while the books are unchanged (Project Gutenberg revises its files).

The labels (`nn/contextlabels.py`, with the curated part in `data/nn/context-labels.json`) come from three sources: the
LibriTTS-R readers' readings by the forced alignments; part-of-speech tags over the Project Gutenberg books (spaCy,
training time only - `requirements/pos.txt`), mapped to readings only for words whose mapping agrees with the aligned
audio on 95% of 10 or more tokens; and hand-written seed sentences and reviewed collocations for the readings the tags
can't separate (lead the metal, bass, bow, row, does the deer, house the verb) and the pairs the audio doesn't
validate. The voice training data (`nn.multi build`) takes a heteronym token's reading from the alignment where the
aligner chose another dictionary reading than the front end's (`nn/readings.py`; `python -m nn.readings check`) - except
where the two differ only by S against Z (use, close, excuse): the aligner can't judge word-final voicing, so those keep
the front end's reading, and the held-out accuracy gate (`nn.context gate`, `folds`) leaves them to the regression set.


## The library

| Module | |
| --- | --- |
| `lib/lexicon.js` | Text normalization, phrasing, and function words - `textToPhrases(text)`, every word pronounced by the pronouncer set with `setPronouncer` (the app's: the G2P network) |
| `lib/g2p.js` | The grapheme-to-phoneme network's inference (`loadG2p(model)`), run on `lib/models/g2pModel.js` - heteronyms by context |
| `lib/phonemes.js` | The ARPAbet vowels (`isVowel`) |
| `lib/wav.js` | WAV encoding (`encodeWav`), and file writing under Node (`writeWav`) |
| `lib/catalog.js` | The book catalog's browsing and search |
| `lib/gutenberg.js` | Project Gutenberg texts - decoding, boilerplate, chapters, and speech preparation |
| `lib/epub.js`, `lib/zip.js` | EPUB books (Standard Ebooks) - ZIP reading in the browser |
| `lib/models/g2pModel.js`, `lib/models/voiceModel.js` | The generated models (`make train`) |

The voice model runs in `nn/js/`: `model.js` (the network - text to parameter frames, a speaker blend and a style),
`render.js` (the synthesizer, streaming), `package.js` (the generated module's loader), `transforms.js` (voice and
style transforms of the frames), and `voices.js` (the designed voices of `data/nn/voices.json` - each a blend of the
model's speakers with a style and transforms).


## The corpora

The training data comes from public corpora, downloaded by `tools/corpora.py` (make train's first stage) - idempotently
and sequentially, resuming interrupted downloads; `--list` shows them; `data/corpora/` is gitignored and its `README.md`
records each corpus's source, license, and size:

```sh
make build/venv-analysis.build                                  # build/venv-analysis (tools/corpora.py's environment)
build/venv-analysis/bin/python -I tools/corpora.py              # download the corpora into data/corpora/
```

| Corpus | Speakers | Download | License |
| --- | --- | --- | --- |
| [CMU ARCTIC](http://festvox.org/cmu_arctic/) 0.95 releases | bdl, rms (male); slt, clb (female) - 1,132 sentences each, 16 kHz, with phone labels | 460 MB | CMU ARCTIC free license (any use, keep the notice) |
| CMU ARCTIC packed | aew (male); eey, ljm, lnh (female) - WAV and text only | 260 MB | CMU ARCTIC free license |
| [LibriTTS-R](https://www.openslr.org/141/) | LibriVox audiobook readers, 24 kHz | | CC BY 4.0 |
| [LibriTTSLabel](https://github.com/kan-bayashi/LibriTTSLabel) | Montreal Forced Aligner phone alignments of LibriTTS | | none stated - used locally, not redistributed |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | 91 actors' emotional speech (`nn.cremad fetch`) | | ODbL |

The four labeled ARCTIC speakers' analysis settings (pitch range and formant ceiling, `data/targets/arctic-*.json`) were
measured by the rule-based synthesizer's calibration tool, `tools/targets.py` (removed in commit 33fbfad; see git
history); `nn/data.py` featurizes their recordings with them.


## License

The CMU Pronouncing Dictionary is distributed under its own license, in `data/LICENSE.cmudict`.
