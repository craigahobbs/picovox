# Changelog

## 2026-10-19

- [2dc76cf](https://github.com/craigahobbs/picovox/commit/2dc76cf) - picovox 1.1.0: playback for JavaScript engines without a JIT - on an iPhone whose Safari interprets JavaScript, synthesis 1.4x -> 2.0x real time in one worker \(the renderer 1.8x faster - voicing from per-frame tables, a radix-4 FFT; the network 1.1x, bit-identical\), and the pipeline in two parts \(sentenceFrames, createSentenceRenderer\) for the network and the renderer in two workers - 2.7x; the M3 Max 40x -> 52x

- [370e03c](https://github.com/craigahobbs/picovox/commit/370e03c) - The paper brought up to picovox 1.1.0: Section 12.4, without a JIT - the performance lab, the iPhone's Safari interpreting JavaScript \(a fixed loop 93 ms against 1 ms, never faster; WebAssembly interpreted too\), node --jitless as its stand-in, the changes \(voicing tables, the radix-4 FFT, the noise gains, the network's loops, two workers\) and their measures \(one worker 1.4x -> 2.0x real time, a network/render pipeline 2.7x\), what didn't make it; the speeds in Node \(52x end to end\), the abstract, and the phones limitation

- [1c5e2c4](https://github.com/craigahobbs/picovox/commit/1c5e2c4) - The performance lab \(tools/perf\): a backend serving the repository to a phone on the local network, which polls it for commands - timings of the playback pipeline stage by stage in a worker, workers and network/render pipelines at once, fixed workloads \(whether the engine compiles them - an iPhone's Safari ran without a JIT\), reloads, and audio route tones - their results in build/perf/results.jsonl; variants of lib/ to try changes in \(variant.sh\), their fidelity \(compare.js - identical, else the SNR\) and speed under node --jitless \(speed.js - within 20% of the phone\); it serves PicoShelf too, with its timeline log. The README's speeds: M3 Max 52x real time end to end \(was 40x\); an iPhone without a JIT 2x in one worker, 2.7x with the network and renderer in two

- [e5e66f9](https://github.com/craigahobbs/picovox/commit/e5e66f9) - The synthesis pipeline in two parts - sentenceFrames \(the network's frame chunks\) and createSentenceRenderer \(the pause, the renderer, the gain and the limiter\) - so an app can run the network and the renderer apart \(in two workers\); sentenceBlocks is the two together, its audio unchanged

- [24a8fc9](https://github.com/craigahobbs/picovox/commit/24a8fc9) - Playback for JavaScript engines without a JIT \(an iPhone's Safari can run without one - every operation interpreted\): the renderer 1.8x and the voice network 1.1x faster on that iPhone \(render 2.09x -> 3.8x real time, network 3.57x -> 4.0x; node --jitless agrees within 20%\), with the network bit-identical and the noise sample for sample

- [427bab1](https://github.com/craigahobbs/picovox/commit/427bab1) - The paper brought up to picovox 1.0.2: Section 9.9's stability result - the seed noise measured \(11.0% of the held-out books' windows and 12.8 regression cases a pair of seeds\), the remedies that didn't steady it, the distillation from four teachers on unlabeled transfer windows that did \(5.0%, 3.8\), the variants, the seeds the steadier network showed were missing \(read, lead, bow\), and 1.0.2's module against 1.0.1's; the stale-folds note, the training-time cost, and the limitation revised

- [3895fcf](https://github.com/craigahobbs/picovox/commit/3895fcf) - The paper: the MLX result stated plainly - MLX 1.78x fresh / 1.60x warm faster in a fair A/B, dropped for one trainer

- [d0cd562](https://github.com/craigahobbs/picovox/commit/d0cd562) - The paper brought up to picovox 1.0.1: the distilled student \(the ladder, the s80 pick, distillation against training from scratch\), the G2P without a dictionary \(one network for every word, the readers' relabeling, heteronyms by a context network and its cross-validated gate\), the dictionary audit \(the Hobbes story - cross-validated screening, adjudication by the readers' audio with controls, the 14 fixes and the listener's 5, the month and day abbreviations, the retrained G2P's measures\), the lead-story fix and the context network's instability \(a placeholder for the stability work\), make train \(the stage graph, the gates, provenance, the pinned texts, the two from-scratch runs\), training speed \(114 -> 71 min: profiling, the power and memory-bandwidth findings, the NanoGPT-speedrun screens - Muon + WSD for the student, a 10K WSD teacher B, the 16-channel critic, Muon rejected for teacher A\), the package and PicoShelf \(the API, the sizes, streaming, the iPhone\), and limitations and future work; the pipeline figure without the dictionary; 9 references added \(Kim & Rush, Muon, modded-nanogpt, Hagele et al., MLX, CTC, Wilson, spaCy, the Wayback Machine\), each link fetched

- [1de36e7](https://github.com/craigahobbs/picovox/commit/1de36e7) - picovox 1.0.2: a steadier heteronym context network \(distilled from four teachers - the same size\) - fresh produce, the sow, the muscles contract, and a bass in the lake now right, none broken; the regression set default 52/52, context 50/54

- [d728c1b](https://github.com/craigahobbs/picovox/commit/d728c1b) - The app's G2P with the steady context network \(make train-g2p, the G2P network and reading stage unchanged\): the regression set default 50/52 -> 52/52 and context 48/54 -> 50/54 \(fresh produce, the sow, the muscles contract, a bass in the lake right; none broken\), missing 3/3 and names 67/73 the same; the 10-fold gate 89.3% -> 90.7%, held out 763 -> 769 of 805; the book measures unchanged; 419,575 bytes \(316,376 gzipped, +8\)

- [6ea6bcd](https://github.com/craigahobbs/picovox/commit/6ea6bcd) - A steady context network: distilled from four teachers \(their own seeds\) toward their mean probabilities, on its examples and on 27,997 unlabeled transfer windows of the ambiguous words in the books \(transferShare 0.5, distill 0.8\), with context items dropped at random \(itemDropout 0.15\) - the same size; five seeds of today's training differed on 11.0% of the held-out books' windows and 12.8 regression cases a pair, the distilled network on 5.6% and 3.2. Seeds for produce \(fresh produce, the farm's produce; the verb\), sow the animal, read \(please read it aloud\), lead's following sense \(follow my lead\), and a ship's bow - each now right in all five seeds. cross-validate keeps a fold's results only from the same labels and settings \(a digest - the shipped module's 88.8% came from stale folds; fresh, 89.3%\); train-context and cross-validate train the teachers in worker processes \(-j\), the same results for any count

- [f19c0cb](https://github.com/craigahobbs/picovox/commit/f19c0cb) - picovox 1.0.1: a better voice \(UTMOS 4.124 -> 4.167, WER 1.61% -> 1.11%, the same size and speed - from a reproducible from-scratch make train\), the G2P with 19 dictionary fixes \(hobbes, welsh, gough, monday, picturesque, ...\) and the lead fix \(lead story, role, singer\), and the month and day abbreviations spelled out \(Nov. -> November, Mon. -> Monday\)

- [1b6bb51](https://github.com/craigahobbs/picovox/commit/1b6bb51) - The app's G2P with the lead fix: its context network retrained \(make train-g2p, 5 min\) with the leading-sense seeds - lead story, role, singer, article now L IY1 D \(the metal - pipe, pencil - stays L EH1 D\); the regression set default 48/52 -> 50/52 \(bow, followed his lead, separate ways\), context 47/54 -> 48/54 \(the four new lead cases right; fresh produce and the sow flipped - more seeds for those destabilized the small context network, so they wait for a steadier context training\); the book measures unchanged

- [301350f](https://github.com/craigahobbs/picovox/commit/301350f) - lead's leading sense in its context: 20 hand-labeled seeds of lead before a noun meaning main \(lead story, role, singer, actor, article, ...\) as L IY1 D, and 4 more of the metal \(lead pencil, paint, poisoning, crystal\) as L EH1 D; regression cases for both \(the listener heard 'They read the lead story' as led\)

- [000709f](https://github.com/craigahobbs/picovox/commit/000709f) - The training dashboard follows any make train target: make train, train-check, and train-g2p record their run \(target and start\) in stages.jsonl; the manifest names each target's stages, the Train view shows only those \(its header names the target\), and an earlier run's unfinished attempts are stale \(waiting, not failed\)

- [be48004](https://github.com/craigahobbs/picovox/commit/be48004) - make train-check: the G2P chain and the heteronym relabeling - checks that the committed G2P and relabels reproduce - run standalone, not in make train, where beside teacher A they cost it about 9 minutes of memory bandwidth \(the run: 71 min, about 62 without them\)

- [8d11f7c](https://github.com/craigahobbs/picovox/commit/8d11f7c) - The app's voice from the second full from-scratch make train \(dddc189, 71 min on 94 W\): teacher B 10K WSD with the 16-channel critic, the 12K Muon + WSD student, the voice data built with the 19-fix G2P - UTMOS 4.167, WER 1.11%, similarity 0.978 vs the shipped 4.124 / 1.61% / 0.937 \(the teacher 4.189 / 1.56%\); the same size \(432 KB gzipped\) and speed; its provenance names a clean dddc189. The G2P chain reproduced lib/models/g2pModel.js byte for byte from the pinned texts. \(The dashboard follows that run.\)

- [dddc189](https://github.com/craigahobbs/picovox/commit/dddc189) - The app's G2P with the 19 dictionary fixes \(the audit's 14 - hobbes, picturesque, reverend, danglars, sancho, leant, ... - and the listener's 5 - welsh, basil, auguste, gough, monday\), retrained by make train-g2p from the pinned texts \(29 min\): book error 0.081% -> 0.062%, top-10K wrong 21 -> 9, names 63/73 -> 67/73, context 44/50 -> 45/50, default 48/52 \(=\); distinct wrong 622 -> 636 and held-out 42.15% -> 42.80% \(about the seed spread\) - adopted by the user over make train's reproduction tolerance; the names gate's minimum 0.86 -> 0.91

- [5f593c9](https://github.com/craigahobbs/picovox/commit/5f593c9) - The listener's shortlist answers: the month and day abbreviations are spelled out with their period \(Jan. ... Dec., Mon. Tue. Thu. Fri. - not mar/sat/sun/wed, which are words, and mon only with its period, as it's French too; lib/lexicon.js and the Python front end alike\), and five more dictionary fixes heard by the listener - welsh W EH1 L SH, basil B EY1 S AH0 L, auguste AH0 G AH1 S T, gough G OW1, monday M AH1 N D EY2 \(data/nn/g2p-fixes.json, with the evidence\); the rest of the 58 kept the dictionary's

- [02f6228](https://github.com/craigahobbs/picovox/commit/02f6228) - make train ships the 16-channel critic \(teacher B and the polish\) and the lean clip \(teachers A and B\): quiet interleaved timings at 94 W - B -20% per step with the narrower critic \(its quality inside the seed spread on two full chains: students 4.166 / 1.11 and 4.147 / 1.21\), the lean clip A -4.4% and B -2.3%; the critical path estimate about 65 minutes

- [f34ff1a](https://github.com/craigahobbs/picovox/commit/f34ff1a) - Tests of threads wait with helpers.wait_until \(its own test covers its branches\) - a wait loop in a test left its body uncovered when the condition held at once \(the prefetcher tests at background QoS\); make train's power check wants 90 W \(a 14-inch MacBook Pro negotiates about 94 W from Apple's 140 W adapter\)

- [1e86148](https://github.com/craigahobbs/picovox/commit/1e86148) - The dictionary audit \(picovox_train/g2paudit.py\): the CMU Pronouncing Dictionary's label errors the G2P learns \(HOBBES HH AA1 B IY0 Z - hobbies\), found by cross-validated disagreement \(two partitions of five folds, ten screening models; 12,750 of 124,101 words suspect\) and adjudicated with the LibriTTS-R readers' audio \(the CTC phoneme recognizer's likelihood of each candidate on the aligned word with its neighbors; 140 control words, no false errors\): 13 confirmed errors and HOBBES \(the dictionary's own evidence\) in data/nn/g2p-fixes.json, which g2p.py's data applies in every stage \(the derived words follow, their -s suffix chosen again\); the names regression set \(data/nn/names-test.json, kind names in tools/heteronyms.js, minimum 0.86 - the shipped module's 63/73\)

- [eeb6d23](https://github.com/craigahobbs/picovox/commit/eeb6d23) - Teacher B ships at 10,000 steps of the warmup-stable-decay schedule \(data/nn/train.json teacher.b: steps 10000, schedule wsd, cooldown 0.25, cooldown-shape 1-sqrt - constant 5e-4 after the warmup, the last 2,500 steps cooled down by 1 - sqrt -, val-from 7500; 20,000 steps of the cosine before\) - B ~27 minutes against ~60, make train's critical path about 80 minutes against about 2 hours. The length: one constant-rate B trunk from the first run's teacher A with cooldown branches to 10K, 12K, 15K, and 20K steps, each through the targets, the 12K Muon+WSD student, the polish, and the listener panel \(UTMOS / WER / similarity\): students 4.161 / 1.16 / 0.975, 4.136 / 1.61 / 0.974, 4.140 / 1.46 / 0.973, 4.138 / 1.36 / 0.969 - no trend with B's length. The gate, end to end against the shipped recipe's two seeds \(students 4.124 / 1.61 / 0.969 and 4.139 / 1.31 / 0.967\): B 10K WSD with the 12K Muon+WSD student, seed 1 \(the 10K branch\) 4.161 / 1.16 / 0.975, seed 2 \(from the control's seed-2 teacher A\) 4.187 / 1.51 / 0.972 - both above both controls in UTMOS and similarity, WER inside their spread or better; teachers 4.189 / 1.76 and 4.203 / 1.41 against the controls' 4.190 / 1.81 and 4.198 / 1.46

- [eae8004](https://github.com/craigahobbs/picovox/commit/eae8004) - The student's training ships as Muon with the warmup-stable-decay schedule, 12,000 steps \(data/nn/train.json distill.train: optimizer muon - with the lean clip it takes -, schedule wsd, cooldown 0.25, cooldownShape 1-sqrt; 20,000 steps of AdamW and the cosine before\). On the first make train run's teacher, each student polished and through the listener panel \(UTMOS / WER / similarity\): the control \(the shipped recipe\) seed 1 4.124 / 1.61 / 0.969 and seed 2 4.133 / 1.26 / 0.969; Muon+WSD 20K seed 1 4.172 / 1.11 / 0.977 and seed 2 4.146 / 1.11 / 0.974; Muon+WSD 12K 4.147 / 1.11 / 0.974 - every one above both controls on all three; validation total at 12K 0.0334 = the control's at 20K \(0.0333-0.0335\). The student stage takes ~5 minutes \(12,000 steps at 0.024 s/step on an M3 Max, the 60 W adapter\) against ~9-11

- [64f8a6c](https://github.com/craigahobbs/picovox/commit/64f8a6c) - The training step's options, all off by default \(train/src/picovox_train/optim.py; trainmulti.py and distill.py\) - the speedrun branch \(77e1862\) ported to the training package: --lean-clip \(the global gradient norm of one concatenation in rows, the clip as fused AdamW's grad_scale, one shared step counter - a few launches a step against MPS's loop per tensor; the same as the default up to rounding\), --optimizer muon \(Muon for the hidden Linear matrices, all zero-padded into one batched Newton-Schulz, AdamW-matched RMS, Nesterov momentum warmed up 0.85 -> 0.95; AdamW for the rest\), --ema DECAY \(a flat-buffer EMA of the weights, validated beside them - emaVal rows, ema-best.pt, ema-last.pt; a run without it resumed with it starts the EMA from the resumed weights\), --schedule wsd --cooldown F --cooldown-shape linear|1-sqrt, --crop-warmup STEPS:SECONDS \(trainmulti\), --zero-init-proj; picovox_train.distill train also takes --max-frames and the recipe's train settings, polish --train-args. The defaults are exact: a few CPU steps of teacher A, B \(the critic\), the student, and its polish give model, optimizer, and critic tensors identical to main's \(torch.equal\) and identical logs, stepcheck.py's output is identical, and the smoke run's checkpoints, export, and G2P module are identical \(the package's manifest differs only in the run's paths\). picovox_train.smoke runs every option; tests/test_optim.py and the trainmulti, distill, and smoke tests cover every option and branch \(100% branch coverage\)

- [fc3c006](https://github.com/craigahobbs/picovox/commit/fc3c006) - The books bundle unpacks in one pass: every pinned text the books directory lacks, each verified \(a clean fetch of the 133 texts from the site: 166 s -> a few seconds\)

- [e0e4dc0](https://github.com/craigahobbs/picovox/commit/e0e4dc0) - The books bundle is published with the site, not as a release asset: make gh-pages copies it to https://craigahobbs.github.io/picovox/books/training-books-v1.tar.xz \(books publish - from the books directory, verified, downloaded if missing\); data/nn/books.json's bundle points there \(the same bytes, sha256 00f30c6d...\)

- [1865448](https://github.com/craigahobbs/picovox/commit/1865448) - The paper's references, verified and linked: every entry checked against its publisher/proceedings record \(Escudero et al.'s first author and subtitle, Iseli & Alwan's and ECAPA-TDNN's full titles, Fant et al.'s volume, venue and capitalization fixes\), a canonical link for each \(DOI, else the proceedings page\), and 29 works the paper relies on added - 43 in all

- [2dd33bf](https://github.com/craigahobbs/picovox/commit/2dd33bf) - make train's book texts come from our own bundle first: books bundle TAG packs every pinned text into a deterministic xz tar \(training-books-v1: 133 texts, 24.7 MB\), recorded in data/nn/books.json with its SHA-256 and published as the picovox repository's GitHub release asset - the fetch's first source, then the Gutenberg URL, then the Wayback copy; every text verified against its pin either way

- [29eddd7](https://github.com/craigahobbs/picovox/commit/29eddd7) - The paper's title: PicoVox: Realistic Neural Speech Synthesis in Under 1 MB of JavaScript \(was Small Voices: Browser Speech Synthesis for Audiobooks\)

- [decabd0](https://github.com/craigahobbs/picovox/commit/decabd0) - speak's ahead-of-playback test advances its fake clock only once speak is idle \(no new block across a few event-loop turns\) - deterministic on a loaded machine, where the timer-driven clock outran synthesis \(ahead 1.4 s\)

- [193715f](https://github.com/craigahobbs/picovox/commit/193715f) - test_batch_prefetcher_stopping waits until the prefetch thread has tried to put into the full queue \(its put wrapped and counted\) - deterministic on a loaded machine; it left the stopped-while-waiting line uncovered at background QoS

- [426db35](https://github.com/craigahobbs/picovox/commit/426db35) - make train's book texts are pinned - data/nn/books.json records each Project Gutenberg text the recipe reads \(the heteronym labels' 132 books, the distillation's 22\) with its source URL, the SHA-256 and size of its bytes as fetched, the date pinned, its Gutenberg revision, and a byte-identical Wayback Machine capture where one exists \(50 of 133\); picovox_train/books.py fetches a text once \(Gutenberg, else the archived copy - the first with the pinned hash\) and verifies it on every use, and a text that isn't the pinned one fails the stage with the book, both hashes, and how to proceed \(restore the pinned copy, or re-pin deliberately - books pin NAME - and retrain\). The pins are the texts the shipped models were trained on: the G2P labels' copies \(2026-10-15\) and the first full make train's distillation copies \(2026-10-16 - its pg2554 is Gutenberg's revision of that day, pg2554-2026-10-16\); the recipe names them \(context.labels and distill.targets texts, pins\). With them the context labels reproduce the shipped labels byte for byte \(27,461 examples - Gutenberg's current pg2554 moved one\), the distillation's book items match the first run's token for token \(254,166 items, in order\), and the G2P chain reproduces lib/models/g2pModel.js below its header. The G2P chain's independent work runs in parallel with identical outputs: tag's books and utterance batches in a worker pool \(spaCy's batches unchanged\), cross-validate -j \(folds in worker processes of one thread - byte-identical to the shipped context network's sequential two-thread folds\), and a context-network stage \(train-context, cross-validate\) beside the G2P network; G2P_JOBS \(2 in make train, beside the GPU stages\) and make -j train-g2p \(the chain alone, G2P_STANDALONE_JOBS 10, not after the readers' data build\)

- [78e9679](https://github.com/craigahobbs/picovox/commit/78e9679) - The training package's unit tests - 426 tests, on the CPU in about 20 seconds with no network \(tiny synthetic corpora and models, temp directories; the downloads, the listener models, spaCy, the CTC recognizer, and the subprocesses that aren't the repository's own Node tools mocked\), 100% branch coverage, pylint clean with python-build's pylintrc \(missing-function-docstring off - a comment above each function is the code's documentation\). Equivalence tests against the JavaScript: rng and lf \(lib/rng.js, lib/lf.js\), lib/params.js generated from params.py \(replacing --check\), dsynth's counter-noise rendering against lib/render.js, the G2P port against lib/g2p.js \(pronounceLines.js\) and V8's Math.exp/tanh bit for bit, the front end against lib/tokens.js, and an exported model in lib/model.js against PyTorch \(replacing parity.py\); the Metal kernels' wrappers against a PyTorch fake. # pragma: no cover marks only Apple GPU \(MPS\) lines. The export records its training args' paths relative to the repository root \(never absolute - as main's a020e3a fixed the shipped manifest by hand\). Dead code removed: context's --twin, panel's unused listen/speak options, an unreachable align check and number_words return; multi and readings share data.py's LibriTTS-R label constants \(no import cycle\)

- [7bafaf0](https://github.com/craigahobbs/picovox/commit/7bafaf0) - README and CLAUDE.md: the training package \(train/, python-build\) - its layout, gate, environments, and commands; the renamed references re-wrapped

- [2f57bb2](https://github.com/craigahobbs/picovox/commit/2f57bb2) - The training's Python code is a python-build package, train/ \(picovox_train\): nn/\*.py, nn/vocab.json, tools/corpora.py, and tools/listeners.py move to train/src/picovox_train/; make train runs it from its source \(PYTHONPATH=train/src\) in the same build/venv-\* environments, and make commit runs its gate \(make -C train commit\). Dead code removed first: nn/accent.py and the pool selection \(nn.multi settings, pool, stats, listen, and build's accent/settings path - build takes its pool from data/nn/speakers.json\), nn.cremad readme, arctic-check, accent, and emotion \(and ctc_labels' unused settings\), nn/parity.py, context.test_wer, spectral's log_mel and log_spectral_distance, and the g2pmodel, panel, frontend tokens, and params --check command lines. The training is unchanged: a CPU smoke run's every checkpoint tensor and optimizer state, export, and package, a multi build, verify-tokens, the stepcheck, and make -n train \(modulo the module path\) are identical before and after

- [913fb1d](https://github.com/craigahobbs/picovox/commit/913fb1d) - README: the paper's link is its page on the site \(https://craigahobbs.github.io/picovox/paper/\)

- [f1ec331](https://github.com/craigahobbs/picovox/commit/f1ec331) - make gh-pages publishes the paper beside the API documentation \(build/site/: build/doc/ and paper/ at /paper/\)

- [d81270b](https://github.com/craigahobbs/picovox/commit/d81270b) - README: the paper's link at the top

- [a020e3a](https://github.com/craigahobbs/picovox/commit/a020e3a) - The voice model's manifest without local paths: its training arguments recorded the run's absolute paths \(the recipe file and the critic's checkpoint\) - now relative to the run's checkout; the weights are unchanged \(the audio hash 9e267605... is identical\)

- [4127a6b](https://github.com/craigahobbs/picovox/commit/4127a6b) - PicoVox 1.0.0 - the npm package picovox \(realistic speech synthesis in pure JavaScript, under 1 MB\): lib/ is the package - the audio playback code only

- [8407cc5](https://github.com/craigahobbs/picovox/commit/8407cc5) - make train runs its CPU stages beside the GPU stages at the utility QoS \(the G2P chain - its book tagging at -j 2 - the heteronym check, the teacher's evaluation, the samples\): CPU work beside a GPU stage shares the unified memory's bandwidth - teacher B ran 0.22-0.24 s/step beside the tag job vs 0.153 alone, and three memory-bound jobs slowed it 53% at the default QoS, 15% at utility, 1% at background \(gpucut E5\)

- [724bb2d](https://github.com/craigahobbs/picovox/commit/724bb2d) - make train no longer rewrites committed files: nn.multi build and nn.cremad build wrote data/nn/speakers.json, multi-split.json, and cremad.json whenever their output was the NN_DATA_DIR default - make train's own data directory - so a run edited tracked files \(and its own data stage's input, the pool table\); they now write the committed copies only for the repository's build/nn/data, else beside the data. The three files are updated to the first full make train's - the data the shipped voice model was trained on \(the same speakers and pool; the statistics and the held-out lists of the utterances its build included\)

- [e2d91bf](https://github.com/craigahobbs/picovox/commit/e2d91bf) - Teacher validation schedule: A validates every 5,000 steps \(B starts from A's last checkpoint, so A's best is unused\) and B every 1,000 from step 10,000 \(its score falls through the second half; before the critic starts at 1,000 it can score below the critic's checkpoints\) - 23 fewer validations, ~1.8 min off make train's critical path \(A 4.6 s and B 4.9 s each in the run\). nn/trainmulti.py gains --val-from and --save-every \(last.pt every 1,000 steps as before, so a resume loses no more\), and --critic-channels \(default 32, unchanged\) for the narrower-critic experiment. Training is identical: validation draws no training random state - 6 CPU stage-B steps \(critic from step 2\) validating at 2/4/6 vs at 6 only, and against main's code: model, critic, and optimizer state torch.equal

- [6e7c10b](https://github.com/craigahobbs/picovox/commit/6e7c10b) - make train's power check: the hint that a non-Apple USB-C charger may negotiate only 60 W through the MagSafe cable \(a 100 W battery pack did: 20 V at 3 A over MagSafe, 100 W over a USB-C cable\)

- [0ae8fb2](https://github.com/craigahobbs/picovox/commit/0ae8fb2) - PicoVox: the npm package is picovox - realistic speech synthesis in pure JavaScript, under 1 MB \(package.json, README, CLAUDE.md, the dashboard's header; the repository stays voice-synth\)

- [7605807](https://github.com/craigahobbs/picovox/commit/7605807) - The app's voice model from the first full make train \(d56063e's run\): the rebuilt teacher \(UTMOS 4.187, WER 2.11% vs the previous 4.144, 2.37%\) distilled to s80 and polished - UTMOS 4.123, WER 1.61%, similarity 0.969 vs the shipped 4.130, 1.71%, 0.953 on the panel; the listening samples 4.199 vs 4.194; 349,547 parameters, 432 KB gzipped, 94x real time - with its provenance \(the commit, the recipe's and the data's hashes, the gate's scores\). nn/js/packagecheck.js compares the manifest without the provenance record

- [36af687](https://github.com/craigahobbs/picovox/commit/36af687) - make train renders listening samples of each training stage's final model - teacher A, teacher B, the student, and its polish - and of the shipped lib/models/voiceModel.js as the reference \(before the package stage replaces it\), each its own CPU stage beside the next GPU stage \(samples-reference, samples-teacher-a, -teacher-b, -student, -polish\): tools/trainsamples.js exports the checkpoint as the package stage does and speaks data/nn/train.json's new samples texts \(Moby-Dick's opening, a dialogue, heteronyms\) in six training speakers' voices exactly as the app speaks \(static/audiobook/worker.js streamNeuralUnit, the app's G2P\), scores each with UTMOS on the CPU, and lists them in TRAIN_DIR/samples/index.json \(under a lock, replaced atomically\). The dashboard's Listen tab shows them, newest model first, each speaker's sample beside the shipped model's, the stages not rendered yet listed as waiting; the monitor's checkpoint samples stay below. data/nn/train-stages.json and test/testTrainStages.js have the new stages; README's make train section describes them

- [d56063e](https://github.com/craigahobbs/picovox/commit/d56063e) - make train's package stage: the export reads the run's data \(TRAIN_ENV - NN_DATA_DIR\), not build/nn's - the first full run's package failed on it

- [9c0842a](https://github.com/craigahobbs/picovox/commit/9c0842a) - make train's voice gate compares the student with the shipped model, not the teacher: nn.distill eval baseline=lib/models/voiceModel.js speaks the packaged module as the app does \(nn/js/unpack.js\), and pick --against baseline passes a student no worse than it \(UTMOS within 0.05, WER within 1 point, similarity to the teacher at least 0.85\) - a better teacher \(the first full run's: UTMOS 4.187 vs 4.144\) no longer fails an equal student \(4.123 vs the shipped 4.130\)

- [9bdd7cb](https://github.com/craigahobbs/picovox/commit/9bdd7cb) - make train checks the power first: on battery or an adapter under 96 W \(an M3 Max draws ~85 W under the run - on less the battery drains and training slows\), it says so and, at a terminal, waits for Enter; TRAIN_MIN_WATTS, TRAIN_POWER_CHECK=0

- [cb2b00e](https://github.com/craigahobbs/picovox/commit/cb2b00e) - make train keeps a record for the training dashboard, and the dashboard's Train tab follows it

- [4922b9e](https://github.com/craigahobbs/picovox/commit/4922b9e) - make train: the CPU-only checks \(the G2P chain's context labels, the heteronym relabeling\) run after the readers' data build - in the teacher's GPU-bound hours, when the cores are idle - instead of competing with the data build on the critical path

- [affc698](https://github.com/craigahobbs/picovox/commit/affc698) - Dashboard: the training run's teacher A stops at 15,000 steps

- [67e96fc](https://github.com/craigahobbs/picovox/commit/67e96fc) - make train's G2P stage builds the shipped context-aware G2P and checks it against the committed module - never replacing it

- [2229287](https://github.com/craigahobbs/picovox/commit/2229287) - The voice data build 6.6x faster, byte-identical: nn/readings.py check_built rebuilt the heteronym inventory \(~1.7 s\) for every utterance with aligned readings - setdefault evaluates its default on every call - so the build's main process ran serially for ~95 min while its workers idled; built once now. Also the main process gets the workers' pre-pronounced cache. 711-utterance build: 106 s -> 16 s, every output file identical

- [f007282](https://github.com/craigahobbs/picovox/commit/f007282) - Docs for the cleanup: README.md \(the command-line tools, the listener panel, the library, and the corpora for the code that remains; the rule synthesizer's voice specs and human speech targets sections removed\), CLAUDE.md's file map and conventions, nn/__init__.py's module list, package.json's description, and the paper's references to removed code cite the commits that removed them

- [192ce2f](https://github.com/craigahobbs/picovox/commit/192ce2f) - Remove the spectral correction head and the all-convolution encoder \(architecture options not adopted\): nn/model.py, nn/trainmulti.py's --correction-\* and --encoder, nn/dsynth.py's and nn/params.py's correction \(nn/js/params.js regenerated\), and the JavaScript runtime's correction path \(nn/js/model.js, nn/js/render.js, nn/js/speak.js\). The app's audio is bit-identical, the trainer's step-0 losses are identical, the voice module reproduces byte for byte, and the smoke test passes

- [7f667db](https://github.com/craigahobbs/picovox/commit/7f667db) - Remove the legacy single-speaker training path and the dead helpers of the removed experiments: nn/train.py \(its losses were already in nn/losses.py\); nn/model.py's single-speaker Model \(MultiModel takes its shared methods; MULTI_CONFIG unchanged\); nn/export.py's and nn/synthesize.py's single-speaker branches; nn/data.py's single-speaker ARCTIC build, statistics, listening proxy, token check, and LibriTTS-R survey \(the shared featurization and ArcticData kept\); nn/cremad.py's unused datasets; the residual noise \(nn/dsynth.py, nn/fit.py's experiment\); nn/params.py's and nn/spectral.py's helpers only the removed tools used. The trainer's step-0 losses are bit-identical \(nn/stepcheck.py, teacher and student\), the shipped voice and G2P modules reproduce byte for byte, and the smoke test passes

- [33fbfad](https://github.com/craigahobbs/picovox/commit/33fbfad) - Remove the rule-based formant synthesizer \(the classic voices, removed from the app earlier\) and the tools that exist only for it: lib/speak.js, klatt.js, glottal.js, voices.js, room.js, analysis.js and their tests; phonemes.js keeps only isVowel \(the lexicon's\), wav.js drops readWav; the speech, intelligibility, and analysis tools \(tools/say, eval, words, stops, nasals, sweep, prosody, panel, synthstats, spectrogram, ltas, slices, pitchstats, vowels, envelope, targets.py\) and their calibration targets \(data/targets/ but the four ARCTIC speakers nn/data.py reads\); the rule voice's bridge to the parameter set \(nn/js/bridge.js, speakparams.js\); requirements/analysis.txt is now only tools/corpora.py's environment

- [6ef55ff](https://github.com/craigahobbs/picovox/commit/6ef55ff) - Remove the finished experiment harnesses: the size search \(nn/sizesearch.py, nn/archeval.py\), the rule-voice bridge and synthesizer equivalence checks \(nn/bridge.py, nn/equiv.py, nn/js/rendercli.js\), analysis by synthesis \(nn/fit.py\), the oracle experiment \(nn/oracle.py\), and the evaluations superseded by nn/panel.py and nn.distill eval \(nn/evaluate.py, nn/evalmulti.py, nn/samples.py\)

- [825f041](https://github.com/craigahobbs/picovox/commit/825f041) - The ten designed voices \(planned work from the main tree\): data/nn/voices.json, nn/js/voices.js \(a designed voice's blend, style, and transforms\), nn/js/voicescheck.js, and nn/js/speak.js's --voice and --voice-style

- [7ef92de](https://github.com/craigahobbs/picovox/commit/7ef92de) - Dashboard: the cleanup step

- [e5baad9](https://github.com/craigahobbs/picovox/commit/e5baad9) - Dashboard: the first full make train run, its teacher stages drawn against multi-a and multi-b

- [0bf501e](https://github.com/craigahobbs/picovox/commit/0bf501e) - Merge heteronym-sz: S/Z-only readings are left to the front end - no S/Z-only overrides from the audio in the voice data, and the held-out gate leaves out the S/Z-only pairs \(read + lives: 89.3% vs majority 59.1%, +30 points\)

- [131f917](https://github.com/craigahobbs/picovox/commit/131f917) - Heteronyms: S/Z-only readings are left to the front end - the voice data takes no S/Z-only override from the audio, and the held-out accuracy gate leaves out the S/Z-only pairs

- [e71812c](https://github.com/craigahobbs/picovox/commit/e71812c) - Merge integrate: the optimize/simplify loop's rounds - distill targets' preparation 7.4 min -> 32 s and overlapped writes \(byte-identical\), the critic's lazy feature casts, validation in one forward, the reader pool from the committed table, make train's parallel stage graph \(critical path ~118 min, was ~175 serial\), recipe-driven distillation

- [1e0d6ed](https://github.com/craigahobbs/picovox/commit/1e0d6ed) - README make train: the critical path's stages and its time \(about 2 hours on an M3 Max\)

- [4d86fe9](https://github.com/craigahobbs/picovox/commit/4d86fe9) - The critic's features stay in its precision until the feature-matching loss casts them \(the critic's own step doesn't use them\): ~2-3% of the critic's work \(micro, interleaved\), the losses identical \(the same casts, where they're used; CPU stepgrad bit-identical\)

- [54c7434](https://github.com/craigahobbs/picovox/commit/54c7434) - The polish by recipe section \(nn.distill polish runs nn.trainmulti --recipe distill.polish --recipe-file, the recipe's polish keys being the trainer's flags - its settings identical\), and make train's GPU stages with two CPU threads \(OMP_NUM_THREADS=2 - the step time is the same at 2, 6, and 16 threads, and fewer leave the cores to the CPU stages beside them\)

- [3d9aedf](https://github.com/craigahobbs/picovox/commit/3d9aedf) - nn.distill targets: the books' words pronounced once, in batches, for every worker \(was each worker pronouncing its words one at a time with the G2P network\) - the inputs in 32 s, was 7.4 minutes; and each batch's outputs written while the GPU runs the next \(non-blocking host copies, an MPS event\) - the targets byte-identical \(4,000 items: every array and the item table\)

- [c3b3b4c](https://github.com/craigahobbs/picovox/commit/c3b3b4c) - Heteronyms by context: the G2P's reading input and ambiguity head, and a context network \(lib/g2p.js, nn/g2pmodel.py, both in lib/models/g2pModel.js\) - no word list or rule at runtime

- [fe70062](https://github.com/craigahobbs/picovox/commit/fe70062) - Teacher training 10-13% faster on the M3's GPU: the critic's strided convolutions take their input gradient as two stride-1 convolutions \(the kernel's even and odd columns - MPS's own strided input gradient costs 2.8x the forward\), and the spectral loss of the full-length crops is compiled \(torch.compile - its elementwise work fused, the batch size dynamic\)

- [959268d](https://github.com/craigahobbs/picovox/commit/959268d) - nn/trainmulti.py validate: no gradients, and the F0 error from the losses' own model forward \(batch_losses with_values\) - not a second forward; the validation results are identical \(every value, 32 reader and 16 emotional utterances on the CPU\) in 21% less time

- [7cacdae](https://github.com/craigahobbs/picovox/commit/7cacdae) - nn/losses.py: the training losses and targets both trainers use \(the priors, the F0 target, the spectral, log-mel, and duration losses, the frames made ready to render\) moved out of the single-speaker trainer, nn/train.py - the shipped trainer no longer imports the legacy one

- [809b363](https://github.com/craigahobbs/picovox/commit/809b363) - nn.trainmulti's training settings default to the recipe's teacher.defaults \(data/nn/train.json\) - the flags keep their names and help, no numbers: every hyperparameter of the teacher is in the one recipe file

- [5a5df88](https://github.com/craigahobbs/picovox/commit/5a5df88) - nn.multi build --pool data/nn/speakers.json: the reader pool and its analysis settings from the committed speaker table, so make train's voice data needs neither the accent classification \(the listener environment\) nor the settings measurement \(about 8 minutes off the critical path\)

- [61b8304](https://github.com/craigahobbs/picovox/commit/61b8304) - nn/__init__.py: nn/gpulock.py and make train

- [b3a4531](https://github.com/craigahobbs/picovox/commit/b3a4531) - nn/data.py: length_batches removed - unused \(trainmulti's batch_stream makes the batches\)

- [dfaaccc](https://github.com/craigahobbs/picovox/commit/dfaaccc) - nn.trainmulti default_settings\(\): the flags' defaults as a namespace, for nn/archeval.py and nn/stepcheck.py - their copied default-settings tables are gone \(every value was equal to the trainmulti flag's default; checked\)

- [5e9b7bc](https://github.com/craigahobbs/picovox/commit/5e9b7bc) - make train: each stage a Makefile target with a stamp in build/train \(make -j runs independent stages at once; the GPU stages under nn/gpulock.py one at a time, the CPU stages beside them\), the stage graph in the README; the voice data built with the committed G2P and the G2P retrain checked against it \(off the critical path\); nn.heteronyms --check, nn.distill's shipped student \(the recipe's distill student\) and pick as the gate \(exit 1 when no student is within the tolerance\), the distillation's books by Project Gutenberg id \(downloaded when missing\), and nn.package --provenance \(the commit, the recipe's and the data's hashes, the gate's scores in the manifest\)

- [c9dd4e8](https://github.com/craigahobbs/picovox/commit/c9dd4e8) - One recipe for make train: data/nn/train.json holds the G2P, the teacher \(stages A and B\), and the distillation \(data/nn/distill.json merged in - its targets, ladder, polish, evaluation, and tolerance\); nn.trainmulti --recipe teacher.a / teacher.b takes a section's settings \(flags override\) and --stop ends a run early on its schedule \(stage A: the 30,000-step schedule stopped at 15,000, as the shipped teacher was made\); NN_DATA_DIR moves the training data \(with NN_RUNS_DIR and NN_DISTILL_DIR, a make-train tree of its own\); nn/gpulock.py runs a command holding an exclusive lock, so make -j runs one GPU stage at a time

- [0012c51](https://github.com/craigahobbs/picovox/commit/0012c51) - Merge main into integrate

- [9fcc95a](https://github.com/craigahobbs/picovox/commit/9fcc95a) - Dashboard: MLX dropped \(the experiment measured 1.6-1.8x over the optimized PyTorch trainer; the PyTorch trainer stays and is optimized and simplified for the M3 instead\)

- [a8578f6](https://github.com/craigahobbs/picovox/commit/a8578f6) - nn/panel.py: the listener panel's scoring in one place - listen, transcribe, the word error rate, the speaker similarity, speak, and score_model \(an exported model's panel evaluation, no training framework; python -m nn.panel score\) - for nn/distill.py, evaluate, evalmulti, archeval, bridge, and oracle, whose copies are gone; nn.distill eval NAME=CHECKPOINT \[--whisper-workers\] evaluates any checkpoint against the teacher \(was build/nn/scratch/evalone.py\)

- [35c7139](https://github.com/craigahobbs/picovox/commit/35c7139) - Merge integrate: distillation \(nn/distill.py, data/nn/distill.json\), the G2P-network training data \(nn/g2pmodel.py, the network-based nn/frontend.py, build --out\), the G2P training tooling \(nn/g2p.py, nn/heteronyms.py, data/nn/train.json, g2p-words.json, g2p-relabel.json - the shipped G2P reproduces bit for bit\), and nn/smoke.py

- [fab6980](https://github.com/craigahobbs/picovox/commit/fab6980) - nn/align.py: the substitution cost memoized - the voice data build's alignment 8% less CPU per utterance \(59 to 54 ms in one process\); the build's arrays, index, and alignment are byte-identical \(6 speakers x 40 utterances\)

- [383cf05](https://github.com/craigahobbs/picovox/commit/383cf05) - Dashboard: the MLX review step - M3-specific speedups, simplification

- [15ae764](https://github.com/craigahobbs/picovox/commit/15ae764) - The G2P's training in the repo: nn/g2p.py \(train - four stages, resumable; export; evaluate\) with the recipe in data/nn/train.json, its data \(data/nn/g2p-words.json - the unknown words' labels and the book counts; data/nn/g2p-relabel.json - the readers' pronunciations\), nn/heteronyms.py \(the relabeled words from readers' usage\), and nn/smoke.py \(every training entry point a few steps on the CPU\)

- [1cab4fe](https://github.com/craigahobbs/picovox/commit/1cab4fe) - Dashboard: the MLX switch - the full voice-training port, and mlx-s64-b compared with scratch-s64-b

- [98bca82](https://github.com/craigahobbs/picovox/commit/98bca82) - Dashboard: the MLX stage-A run's planned stop at 15,000 steps, like scratch-s64-a

- [f990e28](https://github.com/craigahobbs/picovox/commit/f990e28) - Merge main into integrate: the heteronym G2P \(v3\), its regression set and checker

- [f307204](https://github.com/craigahobbs/picovox/commit/f307204) - Dashboard: the heteronym implementation step

- [35c3770](https://github.com/craigahobbs/picovox/commit/35c3770) - Merge g2p-heteronyms: the readers'-usage relabeling tools, the heteronym regression set \(data/nn/heteronyms-test.json, 105 sentences, 46 words\) and its checker \(tools/heteronyms.js\), and the review

- [d11c1fc](https://github.com/craigahobbs/picovox/commit/d11c1fc) - Dashboard: comparison runs - pipeline.json compare maps a run prefix to a reference run drawn faded on its charts \(the MLX runs against the PyTorch scratch-s64-a, the profile-guided quality runs too\); the voice students' group also holds the MLX runs

- [51294da](https://github.com/craigahobbs/picovox/commit/51294da) - A heteronym regression set and its checker: data/nn/heteronyms-test.json \(105 sentences, 46 words - each context-dependent word in each reading: lives, read, lead, live, close, use, tear, wind, wound, record, present, object, content, minute, bow, the noun/verb stress pairs, the -ate words; and three readings the dictionary lacks\) and tools/heteronyms.js, which scores a G2P through the app's front end \(textToPhrases with the G2P pronouncer\) by kind - the default reading 48/52 \(old G2P 33/52\), the context-dependent readings 4/50 \(none expected without a context model\), the missing readings 0/3 - and exits 1 below the minimums \(default 0.9\), the make train gate

- [188be30](https://github.com/craigahobbs/picovox/commit/188be30) - Dashboard: the MLX training experiment

- [bae9c97](https://github.com/craigahobbs/picovox/commit/bae9c97) - Merge g2p-frontend: the voice training data pronounced by the app's G2P network - nn/g2pmodel.py \(lib/g2p.js's inference, bit for bit\), the network-based nn/frontend.py, nn.multi/nn.cremad build --out, words pronounced once before the workers start

- [bf08ff8](https://github.com/craigahobbs/picovox/commit/bf08ff8) - Merge distill: knowledge distillation of the voice model \(nn/distill.py, data/nn/distill.json\), trainmulti's --init-critic and --keep-every 0, README "Distilling the voice model"

- [70b7cd2](https://github.com/craigahobbs/picovox/commit/70b7cd2) - The app's G2P adds lives \(L AY1 V Z\), subject, and address to the readers'-usage relabeling \(56 words, a statistically clear readers' majority from 3+ speakers; function words and reduction-only variants left alone\): all 56 right, book tokens 0.05%, distinct 2.4%

- [c5a38b8](https://github.com/craigahobbs/picovox/commit/c5a38b8) - The app's G2P also reads "lives" as readers do \(L AY1 V Z, was L IH1 V Z\), and subject \(S AH1 B JH IH0 K T\) and address \(AH0 D R EH1 S\): the readers' majority is now the target wherever it is statistically clear \(Wilson 95% lower bound over one half, 3+ speakers; was a 2/3 share\) - 56 words, all right \(was 53\); book tokens wrong against the readers' targets 0.05% \(was 0.06%\), distinct book words 2.4% \(was 2.5%\); a test pins the heteronyms' readings and the function words' full forms

- [8884f7f](https://github.com/craigahobbs/picovox/commit/8884f7f) - Dashboard: make train's reproducibility and reliability steps

- [4101107](https://github.com/craigahobbs/picovox/commit/4101107) - Dashboard: the merge and the alternating optimize/simplify loop on the training code

- [5d0167f](https://github.com/craigahobbs/picovox/commit/5d0167f) - Dashboard: a make train step for the heteronym review

- [62702ac](https://github.com/craigahobbs/picovox/commit/62702ac) - The app's G2P says what readers say for 53 words whose dictionary-first pronunciation isn't the usual reading - lead L IY1 D \(was L EH1 D, the metal\), live, wind, wound, tears, record, content...: the shipped model fine-tuned on readers' majority pronunciations from the aligned audiobook speech \(function words and reduction-only variants left as they were\); all 53 right \(was 0\), held-out WER 35.71% \(was 35.78%\)

- [1445ade](https://github.com/craigahobbs/picovox/commit/1445ade) - Training-data builds with the G2P network: nn.multi build --out DIR \(its speaker table and split beside it; verify-tokens and stats take --out\) and nn.cremad build --out DIR \(its summary beside it\); the builds pronounce their texts' words once, before the workers start. CREMA-D's tokens are the front end's at build time, its CTC alignment reused when the phones without stress \(all the alignment sees\) are unchanged. The network build of the multi-speaker data includes 37,536 of 40,518 utterances \(the dictionary build 36,053\), 55.97 hours \(52.70\), LibriTTS-R phone match 94.53% \(93.78%\)

- [cce939e](https://github.com/craigahobbs/picovox/commit/cce939e) - The voice training data is pronounced by the app's G2P network, not the dictionary: nn/frontend.py pronounces every word with nn/g2pmodel.py, a Python port of lib/g2p.js on the same lib/models/g2pModel.js that is bit for bit - lib/g2p.js's summation order, and V8's Math.exp and Math.tanh \(fdlibm, with arm64's fused multiply-adds emulated exactly; 0 of 8.5M values differ from Node\). Every decoding step's logits for 2,048 words are bit-identical to lib/g2p.js's, the pronunciations of all 27,535 distinct words of the training, CREMA-D, panel, and ARCTIC texts agree, and nn/js/tokenizeLines.js \(now with the G2P network\) and nn/frontend.py give identical tokens on every training text \(36,779 pool texts, 2.9M tokens; nn.data's 6,850 texts; CREMA-D and the panel\). Words are pronounced in batches, optionally by worker processes \(pronounce_texts\), and the pronunciations can be handed to a pool's workers \(set_pronunciations\)

- [0bc9498](https://github.com/craigahobbs/picovox/commit/0bc9498) - Dashboard: a G2P step for heteronyms and variants learned from readers' usage

- [4c2b96a](https://github.com/craigahobbs/picovox/commit/4c2b96a) - Merge pgo-small: training speedups - profile 2 \(exact critic refactor, fused critic AdamW, bf16 critic convolutions, fixed-shape batches\) and the profile-guided pass \(shared STFTs, one nanmedian, Metal kernels for the frication magnitude and harmonic amplitudes, plans finished on the device, constant F0\) - s64 stage A 2.2-2.4x, stage B about 2x, teacher stage A 2.2x vs main

- [f62db13](https://github.com/craigahobbs/picovox/commit/f62db13) - Dashboard: a make train step for the training data pronounced by the G2P network

- [af06e64](https://github.com/craigahobbs/picovox/commit/af06e64) - G2P decoding allows 3 phones a letter plus 6 \(was 2 plus 4\) - the letter w \(D AH1 B AH0 L Y UW0, 7 phones\) lost its last phone; no other book word changes

- [66cda62](https://github.com/craigahobbs/picovox/commit/66cda62) - Dashboard: \(?\) explanations for the training measurements - every chart \(losses, durations, pitch, step time, learning rate, the quality statistics, the critic's terms, the emotional-speech variants, distillation and G2P measures\) and the run cards' facts; hover for the title, tap or click to open \(kept open across refreshes\)

- [b13e2cd](https://github.com/craigahobbs/picovox/commit/b13e2cd) - The audiobook app's ?model=<name> speaks with a development voice model, build/app-models/<name>.js \(a training checkpoint, packaged\), on a development server - the name is letters, digits, and dashes only, from that folder only

- [c6c3d4a](https://github.com/craigahobbs/picovox/commit/c6c3d4a) - dsynth: a Metal kernel for the harmonic amplitudes after the tract on Apple GPUs \(with a constant F0\)

- [5dfe9b8](https://github.com/craigahobbs/picovox/commit/5dfe9b8) - dsynth: constant_f0 - training's rendered F0 is the recording's, so the amplitudes aren't differentiated in it

- [03acd8e](https://github.com/craigahobbs/picovox/commit/03acd8e) - dsynth, trainmulti: on MPS a plan made ahead is finished on the device - no mid-step read, and fixed harmonic counts

- [7e68e2a](https://github.com/craigahobbs/picovox/commit/7e68e2a) - dsynth: a Metal kernel for the frication filter's magnitude on Apple GPUs

- [d49a61c](https://github.com/craigahobbs/picovox/commit/d49a61c) - trainmulti: the quality statistics' five medians in one nanmedian \(identical values, a fifth of the kernels\)

- [7ff501b](https://github.com/craigahobbs/picovox/commit/7ff501b) - spectral_loss: each signal's STFT once per size - the log-mel terms reuse the multi-resolution loss's 1024- and 256-point spectra

- [a809633](https://github.com/craigahobbs/picovox/commit/a809633) - trainmulti: drop the critic options that didn't hold or weren't judged \(--critic-every, --critic-channels, --critic-scales\)

- [454856c](https://github.com/craigahobbs/picovox/commit/454856c) - trainmulti: the critic's convolutions in bfloat16 by default

- [003f139](https://github.com/craigahobbs/picovox/commit/003f139) - trainmulti: fixed-shape batches by default \(--bucket-ratio 1.125\)

- [822c4fd](https://github.com/craigahobbs/picovox/commit/822c4fd) - trainmulti: --bucket-ratio, fixed-shape batches

- [bf9ef60](https://github.com/craigahobbs/picovox/commit/bf9ef60) - trainmulti: options for a cheaper critic - its update every k steps, its precision, width, and resolutions

- [e6a6061](https://github.com/craigahobbs/picovox/commit/e6a6061) - trainmulti: the critic's fused AdamW

- [6748f66](https://github.com/craigahobbs/picovox/commit/6748f66) - trainmulti: the critic's input spectrograms once per signal, and the generator's pass without the critic's weight gradients

- [5a37ad3](https://github.com/craigahobbs/picovox/commit/5a37ad3) - The app's G2P is the NN-only agent's polished final model \(370K parameters, 6-bit, 275 KB gzipped, trained on every dictionary word\): book tokens mispronounced 0.05% \(was 0.23%\), distinct book words 2.4% \(was 2.9%\); its held-out twin on the same recipe scores 35.2% on unseen words \(was 37.0%\)

- [341f558](https://github.com/craigahobbs/picovox/commit/341f558) - README: the first distillation's results - the distillation set, the student ladder against the teacher, and the chosen student \(s80 + polish: 36% of the parameters, 41% of the gzip bytes, within the tolerance\)

- [31010ba](https://github.com/craigahobbs/picovox/commit/31010ba) - The app's voice is the distilled s80 student, polished \(350K parameters; module 432 KB gzipped\): UTMOS on the 16 panel texts through the browser path - female 4.203 \(s64 4.129, teacher 4.230\), male 4.071 \(s64 3.974, teacher 4.169\), better than s64 on 15/16 each; its network 1.38x s64's cost \(55x real time in Node\)

- [9a0ff0b](https://github.com/craigahobbs/picovox/commit/9a0ff0b) - Dashboard: planned stops \(pipeline.json stopAt\) - scratch-s64-a runs a 30,000-step schedule, for the teacher's learning rate, and stops at 15,000, so its progress and time left count to 15,000

- [8943ed5](https://github.com/craigahobbs/picovox/commit/8943ed5) - Dashboard: never blank - a run whose log lacks a field \(the profiling runs' rows have no total\) shows '-' instead of throwing, a run card that can't render says so, and unplanned rows whose phase can't be found \(the new pipeline has no run step\) go to the current phase instead of a phase index of -1; stale phase assignments from the old pipeline are dropped

- [39642aa](https://github.com/craigahobbs/picovox/commit/39642aa) - Dashboard: a Training speed step for the profile-guided optimization of small-model training

- [dd124e0](https://github.com/craigahobbs/picovox/commit/dd124e0) - Dashboard: the voice students' run group starts open

- [d6f2372](https://github.com/craigahobbs/picovox/commit/d6f2372) - Dashboard: a new pipeline for the current work - choose the voice and G2P models, training speed, merge the worktrees, make train, the small-voices library, publish, and the voices and paper \(each step backed by its log or a marker file in build/nn/pipeline/\); the finished Ten neural voices pipeline is archived in tools/dashboard/pipelines/

- [17c2ac8](https://github.com/craigahobbs/picovox/commit/17c2ac8) - Dashboard: the from-scratch s64 experiment as a Distillation step \(running until its log says runs done, however quiet a long stage leaves it - log steps' new doneWhen and failWhen\), and the voice students' runs \(distill-, scratch-\) in their own group

- [ce6a829](https://github.com/craigahobbs/picovox/commit/ce6a829) - nn/distill.py pick: the fewest parameters within the tolerance, and of one size \(a student and its polish\) the higher UTMOS

- [9f60e23](https://github.com/craigahobbs/picovox/commit/9f60e23) - The app's G2P is the smaller NN-only model \(370K parameters, was 657K; 6-bit, 275 KB gzipped, was 424 KB; 1.4x faster\): book tokens mispronounced 0.23% \(was 0.38%\), distinct book words 2.9% \(was 3.3%\), held-out words 37.0% \(was 35.8%\) - by frequency, fewer words mispronounced overall. The layout test converts to whichever layout the shipped model doesn't use, and compares its large arrays without assert.deepEqual, whose diff of two differing 200K-element arrays took over 100 GB

- [5188f4f](https://github.com/craigahobbs/picovox/commit/5188f4f) - The audiobook app moves to static/audiobook/, and the root index.html shows README.md \(MarkdownUp\) with a link to it - the site \(make site, published by make gh-pages\) keeps the repository's layout

- [789eb0d](https://github.com/craigahobbs/picovox/commit/789eb0d) - data/nn/distill.json: an s80 rung \(350K parameters\) between s96 \(within tolerance\) and s64 \(not\)

- [aadf95d](https://github.com/craigahobbs/picovox/commit/aadf95d) - The app's voice is the s64 student's final polish \(step 2,000\): UTMOS on the 16 panel texts through the browser path - female 4.129 \(was 4.098\), male 3.974 \(was 3.933\), better on 12/16 each

- [e28368c](https://github.com/craigahobbs/picovox/commit/e28368c) - The app's voice is the distilled s64 student \(238K parameters, a quarter of the teacher's; module 314 KB gzipped, was 1,062 KB\), from its polish run's best checkpoint at step 1,000 of 2,000 - the voice network 4.1x faster \(32x real time in Node\), for phones; UTMOS on the 16 panel texts through the browser path: female 4.098 \(teacher 4.230\), male 3.933 \(teacher 4.169\)

- [48cc8cc](https://github.com/craigahobbs/picovox/commit/48cc8cc) - Dashboard: each run group lists its running runs \(live or stalled\) first, then the done ones \(finished, stopped, failed\), each in first-seen order - a run that finishes moves to the done runs and the others keep their places

- [05a8ab7](https://github.com/craigahobbs/picovox/commit/05a8ab7) - Wrap the generated model modules at 120 columns \(nn/package.py LINE_LENGTH, and the G2P export\) - the manifests and weights unchanged

- [8362914](https://github.com/craigahobbs/picovox/commit/8362914) - A faster renderer, sample for sample the same \(hash of every sample unchanged\): the tract and biquad responses return without allocating, the filters' pole coefficients in flat arrays \(no template-string lookups or mapped copies per frame\), noise-frame buffers preallocated, the per-sample interpolation inlined and the frame pair's harmonic amplitudes kept while the pair is, and the FFT's swaps without temporary arrays - 1.4x faster in Node

- [cb7e9ac](https://github.com/craigahobbs/picovox/commit/cb7e9ac) - Playback never drops audio: a late block delays the rest of its unit \(a stall\) instead of being skipped - on a device slower than real time every block after the first underrun was dropped, so the clock ran in silence. A streaming unit starts once its synthesized audio covers playing it through at the measured synthesis speed \(or when complete\), so a slow device pauses between sentences rather than mid-word. The worker skips a unit its superseded run just finished \(a deal race\), the timeline keeps 300 events and marks deals and stalls, and ?slowdown=N stretches synthesis to reproduce a slow device

- [e6ab2e3](https://github.com/craigahobbs/picovox/commit/e6ab2e3) - Architecture options for the next voice training, all off by default \(flags off: losses, gradients, exports, and the app's audio bit-identical\): a spectral correction head \(--correction-bands, +5K params, validation loss lower from step 1,000\), an all-convolution encoder \(--encoder conv, -14.6% params, ties attention on matched proxies\), quantization-aware training and packed 4-6 bit exports \(nn/quant.py - 4-bit/g64 QAT recovers ~70% of the UTMOS loss at a 45% smaller module\), and the size search \(nn/sizesearch.py\) with its checks \(nn/archeval.py, nn/equiv.py, nn/stepcheck.py\)

- [9eafcf3](https://github.com/craigahobbs/picovox/commit/9eafcf3) - The multi-speaker training pipeline: nn/multi.py and nn/trainmulti.py \(stage A readers and actors, stage B with the critic; resume marks the saved optimizer groups fused\), nn/cremad.py \(CREMA-D emotion data and data/nn/cremad.json\), nn/accent.py, nn/evalmulti.py \(checkpoint panels\), the reader split \(data/nn/multi-split.json\), and the profiled training speedups \(1.73x fresh, 2.07x warm\) in nn/dsynth.py, nn/model.py, nn/train.py, and nn/data.py

- [b397db3](https://github.com/craigahobbs/picovox/commit/b397db3) - Stream the voice network: the decoder \(per-frame layers and short convolutions, no attention\) runs a chunk at a time, each layer only half a kernel ahead of the one above, so speech starts after the first third of a second of frames instead of the whole sentence - frames bit-identical to the whole-sentence decode \(16 panel texts, chunks of 1 to 1,000 frames\); the linear layer takes four rows per pass over the weights \(identical sums, 1.4-1.65x faster network in Node and jsc\); the tokenizer uses the voice model's own vocabulary \(manifest.vocab\), so the app no longer imports nn/vocab.json

- [13b19b4](https://github.com/craigahobbs/picovox/commit/13b19b4) - The engine imports the generated model modules once and posts them to its workers - each worker imported them itself, so a desktop page \(3 workers\) downloaded each model up to 3 times

- [0e505e0](https://github.com/craigahobbs/picovox/commit/0e505e0) - README: distilling the voice model - nn/distill.py's pipeline \(targets, students, polish, evaluation, pick\) and data/nn/distill.json

- [23d9281](https://github.com/craigahobbs/picovox/commit/23d9281) - The app's voices are the final stage-B checkpoint \(step 20,000\): UTMOS on 16 panel texts through the browser path - female \(slt\) 4.230 \(was 4.198, better on 11/16\), male \(bdl\) 4.169 \(was 4.107, better on 14/16\)

- [febafe4](https://github.com/craigahobbs/picovox/commit/febafe4) - One synthesis worker on iPhones and iPads \(the iPhone's timeline showed 3 workers each running 3x slower - Safari gives a page about one core's worth there\); workers claim a unit as it starts \(before pronouncing it\), so the engine never deals a started unit to another run

- [a87bd34](https://github.com/craigahobbs/picovox/commit/a87bd34) - The app's G2P network is the NN-only model: 1.42% of book tokens mispronounced on held-out books \(was 30.5%\), held-out word error 35.8% - the 657K encoder-decoder trained on every dictionary word, frequency-weighted, at 5 bits \(424 KB gzipped; 5 bits matches 8 bits\). lib/g2p.js is its fast inference \(lookup-table input projections, per-word attention keys, preallocated buffers\): identical pronunciations to the reference on 9,122 book words, 1.4x faster

- [cad0eab](https://github.com/craigahobbs/picovox/commit/cad0eab) - Faster synthesis for phones: a pool of up to 3 workers \(cores - 1\) speaking different upcoming sentences in parallel, and rendering at 24 kHz \(the voices' training rate - half the renderer's cost, UTMOS within 0.02\)

- [b2cebd6](https://github.com/craigahobbs/picovox/commit/b2cebd6) - Device diagnostics: a JavaScript speed check when the worker starts, each sentence's G2P and voice-network times, and the timeline keeps the startup events across plays

- [cda6aa4](https://github.com/craigahobbs/picovox/commit/cda6aa4) - iPhone audio: on iOS the speech plays through the media element \(Web Audio into a media stream\) - iOS muted the separate Web Audio output while the keep-alive media element held the audio session; and device diagnostics - the worker traces its milestones, errors and the audio clock join the ?timing timeline, and &log=<url> posts it to tools/devlog.js

- [a2bc85c](https://github.com/craigahobbs/picovox/commit/a2bc85c) - Knowledge distillation of the multi-speaker acoustic model \(nn/distill.py\): teacher targets in parameter space from training utterances and book sentences, students sharing the teacher's speaker, style, and equalizer tables, the real-audio polish, evaluation through the browser path, and the pick of the smallest student within a fixed tolerance \(data/nn/distill.json\)

- [19d8f91](https://github.com/craigahobbs/picovox/commit/19d8f91) - nn/trainmulti.py: --init-critic \(start the adversarial critic from a checkpoint's\) and --keep-every 0 \(no step-<n>.pt checkpoints\) - for fine-tuning distilled students

- [8eb2c00](https://github.com/craigahobbs/picovox/commit/8eb2c00) - Faster starts, especially on iPhones

- [19b6f04](https://github.com/craigahobbs/picovox/commit/19b6f04) - Training dashboard: the second profiling step

- [aee7ad5](https://github.com/craigahobbs/picovox/commit/aee7ad5) - Baseline snapshot: the main tree's nn/ training code, data/nn/\*.json, requirements/, and tools/corpora.py as of the distillation start

- [378dab5](https://github.com/craigahobbs/picovox/commit/378dab5) - Training dashboard: the distillation and retraining-pipeline phase

- [fdf958e](https://github.com/craigahobbs/picovox/commit/fdf958e) - The generated model modules live together in lib/models/ - voiceModel.js \(was data/nn/multi.js\) and g2pModel.js \(was lib/g2pModel.js\) - app-time code, imported like any module; generated files aren't linted, and test/testModels.js covers the voice model's format

- [a99fbb6](https://github.com/craigahobbs/picovox/commit/a99fbb6) - The audiobook player pronounces every word with the G2P network - no pronouncing dictionary at runtime

- [8fc181b](https://github.com/craigahobbs/picovox/commit/8fc181b) - Remove unused code: the worker is neural-only \(the rule synthesis, room ambience, and whole-text paths the app no longer sends are gone, with their ~115 KB of imports\), the engine's messages drop the crowd and classic-voice fields, tools/browsercheck.js checks the neural path, and lib/gutenberg.js's unused displayAuthor goes

- [70f87f6](https://github.com/craigahobbs/picovox/commit/70f87f6) - Training dashboard: the NN-only G2P step

- [ba400c6](https://github.com/craigahobbs/picovox/commit/ba400c6) - Paper: the player now ships the multi-speaker model as a generated module

- [31d1951](https://github.com/craigahobbs/picovox/commit/31d1951) - Audiobook player: the multi-speaker neural voices \(stage B, step 5,000\) as a generated JavaScript module

- [de9009d](https://github.com/craigahobbs/picovox/commit/de9009d) - Paper: stage B's first checkpoint - every voice above 4.0 UTMOS, past both single-speaker models, pending human listening

- [9361d33](https://github.com/craigahobbs/picovox/commit/9361d33) - Training dashboard: a resumed run's status and step target come from its latest launch, not earlier failures or targets in the same log

- [b1aae76](https://github.com/craigahobbs/picovox/commit/b1aae76) - Paper: the classical WFST baseline - more accurate at 9 MB, far worse at the network's size, and a teacher for distillation

- [d2dadcf](https://github.com/craigahobbs/picovox/commit/d2dadcf) - Training dashboard: the G2P challenger step

- [d528b4b](https://github.com/craigahobbs/picovox/commit/d528b4b) - Paper: stage A's checkpoint scores through step 15,000, and a correction - validation loss flattened while UTMOS kept rising, so stopping listens to the voices

- [6faa539](https://github.com/craigahobbs/picovox/commit/6faa539) - Data efficiency: the plan \(scaling curves with model size, a per-utterance audit, automatic stopping\) in the paper and on the dashboard

- [2bb9040](https://github.com/craigahobbs/picovox/commit/2bb9040) - Paper: the training-throughput results - 1.73-2.07x per step, where the time went, and how equivalence was verified

- [49c21dd](https://github.com/craigahobbs/picovox/commit/49c21dd) - Training dashboard: the architecture-options phase and the G2P classical baseline step

- [63e5e35](https://github.com/craigahobbs/picovox/commit/63e5e35) - Training dashboard: the next-checkpoint card and the Training tab's progress follow the voice run only, not whichever live run updated last

- [f41f0d5](https://github.com/craigahobbs/picovox/commit/f41f0d5) - Training dashboard: runs in stable groups \(fixed first-seen order, collapsible, each with its overall progress\), the selected run remembered, a continued run's charts over its parent's, and unplanned pipeline rows no longer styled as run cards

- [d18103c](https://github.com/craigahobbs/picovox/commit/d18103c) - Paper: an interim G2P result - the pilot network's pronunciations of frequent book misses, before and after, as Whisper heard them

- [9b6c78c](https://github.com/craigahobbs/picovox/commit/9b6c78c) - Paper: Small Voices - browser speech synthesis for audiobooks, methods and results \(paper/paper.md, rendered with MarkdownUp at paper/ on the published site\), and the README's introduction brought up to date

- [2fa0ff2](https://github.com/craigahobbs/picovox/commit/2fa0ff2) - Training dashboard: a G2P phase; runs and checkpoint samples scoped by run-name prefix

- [4edcfca](https://github.com/craigahobbs/picovox/commit/4edcfca) - Training dashboard: the time until the next checkpoint's samples on the Listen tab - the live run's next checkpoint, estimated from its recent step times plus the render

- [bd71c8c](https://github.com/craigahobbs/picovox/commit/bd71c8c) - Training dashboard: four tabs - Progress, Listen, Training, and Logs - with their key numbers in the tab bar, a dot on Listen for an unheard checkpoint, and the tab in the URL hash

- [3558a2d](https://github.com/craigahobbs/picovox/commit/3558a2d) - Training dashboard: load its script fresh and reload itself when its code changes, so a browser never runs a stale copy

- [46f1935](https://github.com/craigahobbs/picovox/commit/46f1935) - Training dashboard: checkpoint samples - the monitor renders each new checkpoint with six speakers, scores them with UTMOS on the CPU, and the dashboard plays them newest first beside today's app voices

- [73cfed5](https://github.com/craigahobbs/picovox/commit/73cfed5) - Training dashboard: a planned profiling step, and the monitor counts processes in the repository's worktrees

- [e164831](https://github.com/craigahobbs/picovox/commit/e164831) - Training dashboard: collapsible phase groups \(progress, what's happening now, steps done\) that open to their rows; a monitor \(tools/dashboard/monitor.js\) so steps without logs show - the pipeline's processes and the workflow's agents with their latest actions

- [01ddf4d](https://github.com/craigahobbs/picovox/commit/01ddf4d) - Training dashboard: finished steps show done 20 s after a complete count or a summary line \(not 3 min\), and unplanned logs join the phase of their folder

- [6e5577d](https://github.com/craigahobbs/picovox/commit/6e5577d) - Training dashboard: a stable pipeline checklist - every step always listed with its bar and latest line, updated in place; unplanned work joins the running phase and stays; the popping Other activity section is gone

- [e024d33](https://github.com/craigahobbs/picovox/commit/e024d33) - Training dashboard: the pipeline overview \(phases and steps from tools/dashboard/pipeline.json, with weighted overall progress\), one-second refreshes that redraw only what changed, and a working file cache \(about 10 requests a second\)

- [95eeff2](https://github.com/craigahobbs/picovox/commit/95eeff2) - Training dashboard: an In progress section - progress bars and time-left estimates for the active logs \(downloads, alignment, featurizing\)

- [1ea7445](https://github.com/craigahobbs/picovox/commit/1ea7445) - Training dashboard - tools/dashboard/ charts the neural training runs \(build/nn/runs/\*/log.jsonl\), tails the data pipelines' logs, and plays the samples, refreshing every 10 seconds from python3 -m http.server

- [92f6f90](https://github.com/craigahobbs/picovox/commit/92f6f90) - The app's narrators are the neural voices only \(Male, Female\) - saved classic choices map to their neural counterparts; the worker no longer warms up the rule voices

- [0a04cf2](https://github.com/craigahobbs/picovox/commit/0a04cf2) - Neural voices in the audiobook player - Neural male \(the default\) and Neural female beside the classic voices; the worker loads a voice's model on first use \(or ahead, on prepare\) and streams each sentence with the narrator's pauses and speed; make site ships nn/js and the models

- [507b481](https://github.com/craigahobbs/picovox/commit/507b481) - Neural voice experiment: a 0.75M-parameter network driving a differentiable source-filter synthesizer

- [a378bd5](https://github.com/craigahobbs/picovox/commit/a378bd5) - Intonation calibrated to audiobook readers: accent placement \(given words, accent spacing, unaccented reporting verbs and subordinators\), higher sentence onset, gentler declination carried across commas, livelier accents with deeper valleys, monotone-cubic contour, per-voice F0 baseline and final-fall recalibration

- [a7b34e9](https://github.com/craigahobbs/picovox/commit/a7b34e9) - Vowel quality and consonant spectra - vowel tables fit to American speakers \(CMU ARCTIC and LibriTTS-R\), back-vowel fronting after coronals \(coronalFronting\), late-gliding wide diphthongs with central off-glides, and non-sibilant fricative spectra \(nonSibilantScale, less hissy /v/, lower /f/ and /th/\)

- [0745927](https://github.com/craigahobbs/picovox/commit/0745927) - Rhythm and pauses calibrated to human speech - phone durations fit to ARCTIC and LibriTTS-R rate-normalized durations, phrase-final lengthening 1.6, longer unstressed vowels, and audiobook pauses \(comma pauses that grow with the phrases' lengths, semicolon and colon pauses, longer sentence pauses, and leading, final, and intermediate-phrase pause parameters\)

- [0f9a041](https://github.com/craigahobbs/picovox/commit/0f9a041) - Voice source and micro-dynamics: aperiodic F0, rd, and formant drift \(no periodic warble\), F0 flutter, breath-noise shelf, louder accents, and source calibration to human measures

- [97405eb](https://github.com/craigahobbs/picovox/commit/97405eb) - Voice development process: listener panel, human speech targets, and voices as data

- [79fd5c6](https://github.com/craigahobbs/picovox/commit/79fd5c6) - Catalog: 567 Project Gutenberg books \(the catalog builder's full run\), reference works filtered out, and builder fixes - one shared DOMParser \(memory\), title casing, duplicate matching, author particles, and subject names

- [72a346e](https://github.com/craigahobbs/picovox/commit/72a346e) - Audiobook player for public-domain books, javascript-build, and 100% test coverage

- [2e2bfb8](https://github.com/craigahobbs/picovox/commit/2e2bfb8) - Text-position player - the timeline is the text's sentences, synthesized on demand in a window ahead of the position and cached \(bounded for books\); skip by 1% and 5%, a position slider, and per-text bookmarks in browser storage

- [49ed445](https://github.com/craigahobbs/picovox/commit/49ed445) - Player skips by 1 and 5 minutes \(no 5-second skips\), and schedules blocks only a few seconds ahead

- [5c95fd2](https://github.com/craigahobbs/picovox/commit/5c95fd2) - Player - play/pause, skip back and ahead 5 s, and a time readout over the streamed audio \(kept as per-member block timelines\); crowd playback waits for every member's first block

- [f7532b5](https://github.com/craigahobbs/picovox/commit/f7532b5) - Crowd - six speakers at once \(pitchScale and tractScale voice parameters, one worker per member, panned and offset\), with a mono crowd mixdown for download

- [647cc9b](https://github.com/craigahobbs/picovox/commit/647cc9b) - Prosody - intermediate phrase breaks, accent clash, floored downstep, bridged accents, deeper continuation falls, phrase-final lengthening, loudness declination, and a prosody statistics tool

- [1f4700b](https://github.com/craigahobbs/picovox/commit/1f4700b) - Liquids and clusters - lateral pole-zero pair for /l/, quieter prevocalic liquids and glides, long devoiced VOT in stop+liquid clusters with /tr/ affrication, retuned L/R/Y targets

- [ae711f1](https://github.com/craigahobbs/picovox/commit/ae711f1) - Nasals, /h/, and utterance onsets - unreduced phrase-initial vowels, breathy-voiced intervocalic /h/, velar VC loci from the preceding vowel, pauses hold a neighboring nasal's resonances, and a nasal/h/onset test tool

- [d8d9c1f](https://github.com/craigahobbs/picovox/commit/d8d9c1f) - Female voice - darker source tilt, Samantha-matched aspiration and burst levels, lower accent peaks and deeper final fall, unscaled nasal F1, closer IY

- [e2fcaea](https://github.com/craigahobbs/picovox/commit/e2fcaea) - Stops: place-specific release frication and VOT, louder bursts and aspiration, stronger velar loci, VC loci from the preceding vowel, voiced stop voice bar and onset, partly devoiced /t/ flaps; tools/stops.js minimal-pair test

- [2d1062c](https://github.com/craigahobbs/picovox/commit/2d1062c) - Segmental quality - tense/lax vowel durations, lax-vowel inherent spectral change, happy-tensing, fronted OW, AA F2; voiced fricative voice bar and devoicing, weaker V/DH/TH, pre-pausal fricative fade, post-pause frication rise

- [c6c3858](https://github.com/craigahobbs/picovox/commit/c6c3858) - Streaming playback - synthesize\(\) block hook, speak\(\) onBlock at a calibrated gain with a soft limiter, worker warm-up and block streaming, room ambience by a Web Audio convolver

- [1ddc88d](https://github.com/craigahobbs/picovox/commit/1ddc88d) - Root index.html redirect so python3 -m http.server opens the app

- [2e4a82f](https://github.com/craigahobbs/picovox/commit/2e4a82f) - Browser port - Web Audio page with a synthesis worker; lexicon setDictionary and browser-safe wav.js; README

- [2abb1a5](https://github.com/craigahobbs/picovox/commit/2abb1a5) - Voice synthesizer baseline - LF source, Klatt cascade, CV transitions, rule prosody, Whisper evaluation
