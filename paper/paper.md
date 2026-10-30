# PicoVox: Realistic Neural Speech Synthesis in Under 1 MB of JavaScript

Craig A. Hobbs, with Claude (Anthropic)

October 2026 - a working paper, updated as the work it describes finishes; this version describes picovox 2.0.0.
Sections marked *in progress* report interim results.


## Abstract

We build American English text-to-speech that runs entirely on the listener's device - in a web browser or in Node -
from a package of 727 KB (npm's compressed tarball; 1.1 MB unpacked), with no server, no recorded speech, no
WebAssembly, and no runtime dependencies. A small neural network predicts the 32 interpretable parameters of a classic
source-filter synthesizer every 5 ms, trained end to end through a differentiable copy of the synthesizer; a
rule-based version of the same synthesizer, calibrated against measurements of 43 human speakers, had been rated 1-2
out of 5 for naturalness. A teacher of 932K parameters, trained on 46 hours of speech that a local text-to-speech
model generates at train time in twelve voices - no human recordings - is distilled to a student of 283K parameters
that scores 4.31 on an automatic naturalness predictor (UTMOS; human recordings score 3.8-4.4 on it) with a 1.1%
Whisper word error rate, and runs about 52 times faster than real time in Node and about 2.7 times real time on an
iPhone running JavaScript without a JIT (as Safari does on any page served over plain http). Every word is pronounced
by a
neural grapheme-to-phoneme network of 373K parameters, with a 48K-parameter context network for heteronyms, in place of
a 3.6 MB pronouncing dictionary. A cross-validated audit of that dictionary against audiobook readers' speech found
label errors the network had faithfully learned - "Hobbes" read as "hobbies" - and corrected them; the heteronyms'
readings are now measured in the synthetic speech by our own phoneme recognizer, each measure trusted only if it passes
its controls. One command generates the training speech, measures it, and rebuilds both networks from pinned book
texts in 83 minutes on a laptop GPU (the speech cached), and ships a new model only if it is no worse than the one it
replaces.


## 1. Introduction

An audiobook player for the 2,000-book public-domain catalog we assembled (Standard Ebooks and Project Gutenberg) needs
voices that are pleasant for hours, intelligible, and free: no server, no per-character fees, no recordings to
license. The player is a static web page - it runs from GitHub Pages or `python3 -m http.server` - so the voices must
run in the browser, start speaking within a second, and fit in a page's download.

Modern neural text-to-speech systems sound excellent but are large (tens to hundreds of millions of parameters) and
usually need a neural vocoder or an audio codec to produce sound. Classic formant synthesis is tiny and fast but
sounds robotic. We take a hybrid path: keep a classic source-filter synthesizer as the sound generator, and train a
small network to drive it. The network's outputs remain interpretable - pitch, voicing, breathiness, formant
frequencies and bandwidths - so voices and speaking styles can also be shaped by simple transforms of its output.

The work proceeded in stages, each judged by measurement: a rule-based synthesizer calibrated to human speech
(Section 4), a single-speaker neural parameter predictor (Sections 5 and 6), a multi-speaker, multi-style teacher
(Section 7), and its distillation to the small student that ships (Section 8). The text front end went through the
same arc - from a pronouncing dictionary to one small network that pronounces every word, reads heteronyms from their
context, and was cleaned of the dictionary's own errors (Section 9). Section 3 describes how we measure; Section 10
describes `make train`, which rebuilds every model from scratch; Section 11, how its time came down from 114 to 71
minutes; Section 12, the package (picovox on npm) and the audiobook player built on it (PicoShelf); and Section 13,
the switch from human recordings to speech a local TTS model generates at train time.


## 2. System overview

![The speech pipeline, from text to audio](pipeline.svg)

1. **Text to phonemes** (`lib/lexicon.js`, `lib/g2p.js`). The text is normalized (numbers - a four-digit number as a
   year, ordinals, decades, percentages, money, times, Roman numerals; abbreviations - including the month and day
   abbreviations, Nov. and Mon. - punctuation, an aside's parentheses as commas), and every word is pronounced by the
   grapheme-to-phoneme (G2P) network, with phonemes and stress; there is no dictionary at runtime. For a word whose reading depends on its neighbors ("read", "lead", "lives"), the network's
   ambiguity head says so and a small context network chooses the reading (Section 9).
2. **Phonemes to tokens** (`lib/tokens.js`). A fixed table maps phonemes and prosodic cues to 78 token IDs: the
   phonemes, word boundaries, a start marker, and punctuation tokens (comma, period, question, wh-question, exclamation)
   that tell the network how a phrase should rise, fall, or pause.
3. **Tokens to synthesizer parameters** (`lib/model.js`). The acoustic network (305,035 parameters) predicts each
   token's duration and then, every 5 ms, 32 synthesizer parameters. A speaker embedding and a style embedding condition
   it. Its decoder streams: it runs a chunk of frames at a time, so speech starts before the sentence is decoded.
4. **Parameters to audio** (`lib/render.js`). A source-filter synthesizer - Liljencrants-Fant (LF) glottal pulses and
   noise through resonant filters - renders the audio, by default at 24 kHz, the voices' training rate. There is no
   neural vocoder.

Everything runs in JavaScript, in Node and in a browser (in a Web Worker for long texts), with no dependencies.


## 3. Evaluation method

We judge every acoustic change by measurement, using three instruments, and every pronunciation change by a fourth.

### 3.1 An automatic listener panel

A panel of pretrained models scores a fixed evaluation set (`tools/panel-texts.json`: six Harvard sentences, six
narrative passages with dialogue from public-domain books, two questions, and two exclamations):

| Listener | Model | Measures |
| --- | --- | --- |
| Naturalness | UTMOS22 (strong) | Predicted mean opinion score, 1-5 |
| Intelligibility | Whisper `base.en` | Word error rate (WER) against the text |
| Speaker identity | ECAPA-TDNN (VoxCeleb) | 192-dimension speaker embedding; cosine similarity |
| Style | emotion2vec+ base | Emotion probabilities (angry, disgusted, fearful, happy, neutral, sad, ...) - for the style experiments; retired with the style network (Section 12.7) |

We calibrated the panel on human and commercial speech before trusting it:

| Speech | UTMOS mean | WER |
| --- | --- | --- |
| Human - CMU ARCTIC speakers (bdl, rms, aew, slt, clb) | 4.28-4.39 | 2.7-4.7% |
| Human - LibriTTS-R audiobook readers (six) | 3.77-4.40 | 2.6-8.2% |
| Apple Samantha (unit selection) | 3.94 | 3.9% |
| Apple Daniel (unit selection, British) | 3.41 | 5.7% |
| Apple Fred (1980s formant synthesis) | 2.81 | 10.6% |

Two speaker-embedding facts guided later work: shifting a voice's pitch alone does not make a new speaker (similarity
0.99 to the original), while different human speakers score 0.0-0.5. On the CREMA-D emotional corpus (Section 7.1),
the emotion model recognizes the actors' intended emotion 86.8% of the time (unweighted recall 87.1%) - far above the
crowd's audio-only agreement of about 40%, which suggests CREMA-D was in its training lineage; we treat its scores on
our own voices as optimistic.

From Section 8 on, a model's panel score is its mean over six of the model's speakers (three female, three male) by
the 16 panel texts, spoken exactly as the package speaks (every word pronounced by the G2P network); its similarity is
to the teacher's rendering of the same speaker and text. Similarities are therefore comparable only against one
teacher.

### 3.2 Human speech targets

To calibrate voices against real speech, we measure human recordings and our synthesized speech with the same
Praat-based analysis (`tools/targets.py`, removed with the rule-based synthesizer in commit 33fbfad; see git history):
F0 level, range, declination, and final falls; vowel formants F1-F4 at 20%, 50%, and 80% of each vowel; phone durations,
articulation rate, and phrase-final lengthening; pauses per minute and per punctuation mark; the long-term average
spectrum; voice quality (H1-H2, H1-A3, harmonics-to-noise ratio, jitter, shimmer, cepstral peak prominence); intensity
dynamics; and fricative spectra. The targets are 43 speakers from CMU ARCTIC and LibriTTS-R dev-clean. A voice is
compared to one target speaker, or placed among the human speakers of its sex as z-scores - which separates "another
speaker's voice" from "something no human does".

### 3.3 Human listening rounds

A listening page (the Voice Lab) presents each round's samples - the same three passages from *Moby-Dick*, *Pride
and Prejudice*, and *Walden* in every round - for ratings of naturalness and listenability (1-5), A/B preferences,
and free-text notes. Round 3 added blind tests: a real recording of the speaker the network learned from beside the
network's rendering of the same sentence. One listener (the first author) has rated every round so far; the human
results are therefore indicative, and the automatic panel carries the quantitative comparisons. The same listener's
reports from everyday use of the audiobook player started two of the pronunciation fixes of Section 9.

### 3.4 Pronunciation measures

The G2P network is measured four ways, each against its training labels (the CMU Pronouncing Dictionary's first
pronunciation, as corrected in Sections 9.4, 9.6, and 9.7):

- **Book-token error** - the share of the dictionary-word tokens of 22 evaluation books (2,385,182 tokens) the network
  pronounces differently from the label, phonemes and stress, with the number of distinct words wrong. This is what a
  listener hears.
- **Top-10K wrong** - how many of the 10,000 most frequent book words are wrong.
- **Monitoring WER** - the word error rate on 2,000 dictionary words held out of training (sampled half uniformly and
  half by frequency); mostly rare names, so its absolute level is high.
- **Regression sets** (`train/data/heteronyms-test.json`, `train/data/names-test.json`, run through the package's front end
  by `tools/heteronyms.js`): sentences with the word in context and its accepted and rejected pronunciations, by kind -
  *default* (52 sentences: a heteronym's usual reading), *context* (54: the readings only the context decides),
  *missing* (3: readings the dictionary lacks), and *names* (73: names with verified pronunciations). Each kind has a
  minimum, and the G2P check of Section 10.2 fails below it.


## 4. Rule-based formant synthesis

The rule-based synthesizer (`lib/speak.js`, `lib/klatt.js`, `lib/glottal.js`, `lib/voices.js`) and its analysis and
evaluation tools were removed from the repository in commit 33fbfad once the neural voices replaced it; see git history.

### 4.1 The synthesizer

The baseline is a from-scratch JavaScript implementation of a Klatt-style cascade formant synthesizer with an LF glottal
source, driven by rules: Klatt's duration rules, a rule-based intonation model, and per-phoneme formant targets with
coarticulation. Voices are data (`lib/voices.js`, removed in commit 33fbfad): 80 documented parameters per voice, specs
that inherit from a base, and style transforms (scale, add, set, blended by intensity). The two voices, male and female,
started as one speaker's model with a pitch and formant shift - which the speaker-embedding panel saw through
(similarity 0.66 between them).

### 4.2 Calibration against human speech

We calibrated the rules in four tracks against the human targets, each integrated only if the panel did not regress:

- **Voice source and micro-dynamics**: replacing sinusoidal pitch and formant drift (an audible warble) with
  low-passed noise; F0 flutter; a breath-noise high shelf while voicing; louder accented syllables; jitter,
  breathiness, and Rd set from human voice-quality measures.
- **Rhythm and pauses**: phone durations fit to ARCTIC and LibriTTS-R rate-normalized durations, phrase-final
  lengthening, longer unstressed vowels, and audiobook-style pauses that grow with phrase length.
- **Vowels and consonants**: vowel tables fit to American speakers, back-vowel fronting after coronal consonants,
  late-gliding diphthongs, and corrected non-sibilant fricative spectra.
- **Intonation**: accent placement (given words, accent spacing, unaccented reporting verbs), a higher sentence
  onset, gentler declination carried across commas, livelier accents, and recalibrated final falls.

| Measure | Male before | Male after | Female before | Female after |
| --- | --- | --- | --- | --- |
| UTMOS (panel mean) | 3.16 | 3.14 | 3.25 | 3.33 |
| Whisper WER | 11.4% | 9.4% | 9.3% | 9.8% |
| Distance from the human population (RMS of z-scores) | 1.69 | 0.90 | 1.63 | 1.04 |

The calibration moved the voices most of the way into the human population on the measured statistics - roughly
halving their distance - but barely moved the naturalness predictor, and the human listener rated both rounds 1-2
out of 5 (Section 6). The listener's notes named the problem: "it sounds like a non-native English speaker -
intonations a little off - a little warble here and there." Measured gaps remained in liquids (about 2 dB weak),
intensity range, declination (-1.8 against -0.7 semitones per second), pause frequency (11-12 against 21-23 per
minute), and voice-source detail. Rule tuning had reached diminishing returns.


## 5. Neural parameter prediction

### 5.1 A differentiable source-filter synthesizer

To train a network to drive the synthesizer without hand-labeled formant targets, we wrote a differentiable version
of it in PyTorch (`picovox_train/dsynth.py`), DDSP-style - no recursive filters:

- Voicing is harmonic synthesis: each frame's harmonic amplitudes are the product of the voicing level, the LF
  pulse's Fourier coefficients for the frame's Rd (a precomputed table), a source shelf, and the analytic frequency
  response of the vocal tract (six formants, a nasal pole and zero, higher poles) at each harmonic.
- The glottal phase is computed cycle by cycle on the CPU with jitter and shimmer; it is not differentiated (F0's
  gradient flows through the harmonic amplitudes).
- Aspiration and frication noise are shaped in the frequency domain (short-time Fourier transform, overlap-add).

A JavaScript renderer (`lib/render.js`) implements the same synthesizer and matches the PyTorch output sample for
sample when given the same noise.

### 5.2 The network

The single-speaker model (`nn/model.py`'s `Model`, 748,251 parameters, trained by `nn/train.py` - both removed in
commit 7f667db in favor of the multi-speaker model of Section 7; see git history) is a small non-autoregressive
network in the FastSpeech family: token embeddings, an encoder of convolution and self-attention blocks, a duration
predictor (log duration per token, plus a pause probability and pause length at boundary tokens), a length regulator,
and a convolutional decoder whose head outputs two 5 ms parameter frames per 10 ms frame, each parameter squashed
into its physical range. A static 24-band channel equalizer per speaker models the recording channel. There is no
absolute positional encoding, so any length works; long texts are spoken a sentence at a time.

### 5.3 Training

Training runs end to end through the differentiable synthesizer against CMU ARCTIC recordings of one speaker (slt,
female; bdl, male; about 50 minutes each), with the recorded durations and recorded F0 (teacher forcing). The losses
are a multi-resolution STFT loss and a log-mel loss; duration losses (phone durations, pause classification, pause
lengths); an F0 regression loss on Praat's pitch (F0 gets no useful gradient through the synthesizer); and weak,
fading guides (formant targets from Praat, formant order, trajectory smoothness). Front-end tokens are produced
identically by the JavaScript and Python front ends, verified on 6,850 texts.

### 5.4 What limits quality

Two diagnostics located the ceiling:

- **Analysis by synthesis.** Fitting the synthesizer's parameters directly to each recording, frame by frame, with no
  network (`nn/fit.py`, removed in commit 6ef55ff; see git history), reaches a UTMOS of about 3.84 - the best this
  synthesizer can sound under that fitting procedure, given the right parameters.
- **A prosody oracle.** Rendering the network's spectral parameters with the recording's own durations and pitch
  (`nn/oracle.py`, removed in commit 6ef55ff) barely helps: prosody is not the limit; the spectral envelope is.

The network over-smooths formant bandwidths (and over-produces aspiration noise); narrowing the bandwidths to 0.6 of
their predicted values at inference raises UTMOS from about 3.6 to about 3.85. That fixed scale is a stopgap; the
multi-speaker training addresses the cause (Section 7).

### 5.5 Export and inference

Weights are quantized to 8 bits per row (one scale per row) and exported with a JSON manifest: 811,844 bytes per
voice. `lib/model.js` runs the network in plain JavaScript at 10-14 times real time in Node on an Apple M3 Max,
and the JavaScript and PyTorch outputs agree to within quantization.


## 6. Results

### 6.1 Objective scores

| Speech (held-out sentence and panel texts) | UTMOS |
| --- | --- |
| Human recording - ARCTIC slt (female) | 4.34 |
| Human recording - ARCTIC bdl (male) | 4.14 |
| Neural - bdl (male) | 4.07-4.21 |
| Neural - slt (female) | 3.79-3.99 |
| Analysis-by-synthesis ceiling | about 3.84 |
| Rule-based, calibrated | 3.49-3.54 |

Whisper transcribed the held-out sentence exactly from both neural voices.

### 6.2 Human listening

Mean ratings on a 1-5 scale, over the same three passages per voice (one listener):

| Round | Voices | Naturalness | Listenability | Preference |
| --- | --- | --- | --- | --- |
| 1 - baseline rules | male, female | 1.5 | 1.0 | dry over room ambience ("a little echo-y") |
| 2 - calibrated rules | male, female | 1.8 | 1.2 | calibrated over round 1, both voices |
| 3 - neural | bdl (male) | 3.7 | 3.7 | neural over calibrated rules, both voices |
| 3 - neural | slt (female) | 3.0 | 2.0 | "tinny and a little static" |

In the round-3 blind tests, the listener correctly picked the real recording for the female voice - and picked the
**network's** rendering as the real recording for the male voice.

The step from rules to a neural predictor of the same synthesizer's parameters mattered far more than all the rule
calibration: naturalness roughly doubled. The female voice's remaining defect matches the diagnosis of Section 5.4 -
over-smoothed bandwidths and excess high-frequency noise read as "tinny" and "static".


## 7. A multi-speaker, multi-style teacher

The goal was ten voices (five female, five male) for reading many kinds of books - for example a warm literary
narrator, an older authoritative voice, a young bright voice, a brisk nonfiction voice, and a gentle voice - each
with emotional styles, from one model of about 1M parameters. The model was trained; it became the teacher of the
model that ships (Section 8). The ten designed voices were specified then (`test/voices.json`: blends of the speaker
table's rows, a style, and parameter transforms) and ship now as `lib/synthesis.js`'s voice and style tables (checked
against it), beside the model's ARCTIC bdl (male) and slt (female) speakers.

### 7.1 Data

*Through version 1.6. Since 2.0 the voice trains on speech a local TTS model generates, with no human recordings
(Section 13.2).*

- **American audiobook readers.** LibriTTS-R train-clean-100 and dev-clean (CC BY 4.0) plus four CMU ARCTIC
  speakers. An accent classifier (SpeechBrain's CommonAccent ECAPA model), run on 16 clips per speaker, kept speakers
  confidently classified as American: **237 speakers (120 female, 117 male), 52.7 hours**.
- **Emotional speech.** CREMA-D (ODbL): 91 actors, 12 sentences, 7,442 clips in six emotions (angry, disgusted,
  fearful, happy, neutral, sad) at several intensities. CREMA-D has no phone alignments; a wav2vec2 phoneme CTC model
  force-aligned all 7,442 clips with no failures.
- Each speaker's pitch range and formant ceiling were measured for the F0 and formant analysis, and each speaker's
  naturalness was scored (UTMOS, four clips each) to guide voice design.
- **Pronounced as the package pronounces.** The training texts were first tokenized with the dictionary. They are now
  pronounced by the G2P network itself (Section 9), through a Python port of `lib/g2p.js` that is bit-exact - V8's
  `Math.exp` and `Math.tanh` included - so the voice trains on the tokens it will be given. With the network's
  pronunciations the build includes 37,536 of 40,518 utterances (55.97 hours; 36,053 with the dictionary), and the
  readers' phones match the LibriTTS-R alignments slightly better (94.53% against 93.78%). A heteronym token takes the
  reading the forced aligner heard where it differs from the front end's (3,364 tokens), except where the two differ
  only by S against Z, which the aligner cannot judge (Section 9.5).

### 7.2 Model and training

The architecture of Section 5.2 gains a learned speaker embedding table and a style embedding (one vector per emotion,
scaled by intensity, neutral at zero), both conditioning the encoder, duration predictor, and decoder. A voice at
inference can be any convex blend of speaker embeddings - so designed voices need not clone any one reader - and a
style any emotion at any intensity. The model has **965,803 parameters** (952,683 excluding the speaker and
equalizer tables). Training is staged: readers first (stage A, the first 15,000 steps of a 30,000-step schedule), then
the emotional styles with an adversarial spectrogram critic and feature matching (stage B). Over-smoothing is attacked
directly - by the critic and loss changes - instead of with the fixed bandwidth scale.

One incident is worth recording: the first launch of stage A did not learn (losses flat through step 350) because
non-finite gradients came from logarithms of exactly-zero spectra in the spectral losses; a small epsilon fixed it,
and a 120-step smoke test confirmed falling losses before the relaunch.

### 7.3 Results

UTMOS of the checkpoints of the first teacher's runs, reading the *Moby-Dick* opening (the single-speaker neural
voices score 4.21 for bdl and 3.99 for slt on the same text):

| Voice | A: step 5,000 | A: step 10,000 | A: step 15,000 | **B: step 5,000** |
| --- | --- | --- | --- | --- |
| Reader 2843 (male) | 3.82 | 3.75 | 3.99 | **4.22** |
| Reader 5750 (male) | 3.74 | 3.85 | 4.07 | **4.23** |
| ARCTIC bdl (male) | 3.73 | 3.71 | 3.98 | **4.24** |
| ARCTIC slt (female) | 3.61 | 3.78 | 3.97 | **4.23** |
| Reader 5895 (female) | 3.50 | 3.37 | 3.54 | **4.11** |
| Reader 6345 (female) | 3.38 | 3.42 | 3.59 | **4.09** |

By the end of stage A, one network for 237 speakers matched the dedicated single-speaker slt model (3.97 against 3.99).
Stage B - the emotional data plus the adversarial critic with feature matching, aimed at the over-smoothing behind the
female voice's "tinny" sound - lifted every voice above 4.0 within 5,000 steps (female mean 3.70 to 4.14, male 4.02
to 4.23), past both single-speaker models (4.23 against 3.99 for slt, 4.24 against 4.21 for bdl) and into the range of
human recordings (4.1-4.3 on this predictor). Validation loss rose slightly meanwhile (3.154 to 3.185): the critic
trades spectral loss for realism.

Stage B ran 20,000 steps; its best validation checkpoint (step 19,000) scores UTMOS 4.144 and WER 2.37% on the panel
(Section 3.1's six speakers by 16 texts) and was the first distillation's teacher. The teacher rebuilt by the first
from-scratch `make train` scored 4.187 / 2.11%, and the second, with the shorter stage B of Section 11.4,
4.189 / 1.56%.

Two lessons carried forward. Validation loss is a poor proxy for perceived quality here: stage A's validation loss
flattened near step 8,000 while its training loss kept falling (3.00 to 2.84 by step 15,000, validation 3.17 to
3.15) - yet the sample voices' UTMOS rose most in exactly that stretch, from 3.52 to 3.70 (female) and 3.77 to 4.02
(male) between steps 10,000 and 15,000, as the learning rate decayed; Section 11.4 meets the same lesson again. And
because a neural judge and an adversarial critic may reward the same artifacts, these scores are not human ratings;
the shipped voices have been heard by one listener in daily use, not in a formal round.


## 8. Distillation to a small student

The teacher is 1.45 MB as a generated module (1,062 KB gzipped) and runs 18 times faster than real time in Node - too
large and too slow for phones (Section 12.3). We distill it (`picovox_train/distill.py`) and ship the smallest student
within a fixed tolerance.

**Targets.** Sequence-level distillation in parameter space: the teacher speaks each input to its durations (pauses
included) and its 32 synthesizer parameters every 5 ms, as inference does, with no audio rendered. The inputs are the
training utterances with their own speakers and styles, and every sentence of 22 Project Gutenberg books spoken twice,
each time by a random reader (8% an emotional-corpus actor, 10% a blend of two readers) in a neutral or, 30% of the
time, an emotional style at a random intensity. The first set was 294,678 sentence-speaker pairs (40,624 training
utterances and 254,054 book sentences), 397 hours of teacher speech, 18.5 GB, made in 9 minutes on an M3 Max's GPU.

**Students** are the same architecture at smaller widths and depths. They share the teacher's conditioning - its
speaker, style, and channel-equalizer tables are copied and frozen, so a voice designed as a blend of the teacher's
speakers is the same voice in a student - run with the teacher's durations, and learn L1 on the normalized parameter
frames plus duration losses against the teacher's durations. A **polish** then fine-tunes the student briefly (2,000
steps) on the real recordings with the teacher's real-audio losses: the spectral loss through the differentiable
synthesizer, and the adversarial critic, started from the teacher's, with feature matching.

**The pick.** Each model is exported, packaged, and scored on the panel; the smallest student whose mean UTMOS is
within 0.05 of the teacher's, whose WER is at most 1 point worse, and whose speaker similarity is at least 0.85 is
chosen. The first distillation (each student 20,000 steps; real-time factors are `tools/speak.js`'s at 44.1 kHz in
Node, the best of three, about +/-15%):

| Model | Parameters | Module (gzip) | UTMOS | WER | Similarity | Real time (network / total) |
| --- | --- | --- | --- | --- | --- | --- |
| Teacher (dim 128, 3 + 6 layers) | 965,803 | 1,447 KB (1,062 KB) | 4.144 | 2.37% | 1 | 18x / 8.4x |
| s96 (dim 96, 3 + 6) | 570,987 | 925 KB (666 KB) | 4.147 | 1.86% | 0.978 | 24x / 8.8x |
| **s80 + polish (dim 80, 3 + 4) - chosen** | 349,547 | 614 KB (431 KB) | 4.130 | 1.71% | 0.969 | 50x / 12.1x |
| s80 | 349,547 | 613 KB (431 KB) | 4.096 | 1.91% | 0.966 | 48x / 11.5x |
| s64 + feature matching + polish | 238,251 | 459 KB (313 KB) | 4.076 | 1.76% | 0.959 | 73x / 13.6x |
| s64 + polish | 238,251 | 459 KB (313 KB) | 4.070 | 1.86% | 0.956 | 67x / 12.9x |
| s64 (dim 64, 3 + 4) | 238,251 | 457 KB (312 KB) | 4.045 | 2.17% | 0.949 | 73x / 13.0x |
| s64, trained from scratch (not distilled) | 238,251 | 459 KB (313 KB) | 3.973 | 2.37% | 0.884 | - |
| s48 (dim 48, 2 + 4) | 127,035 | 302 KB (196 KB) | 3.935 | 2.32% | 0.924 | 147x / 17.6x |
| s32 (dim 32, 2 + 3) | 64,683 | 209 KB (125 KB) | 3.593 | 2.72% | 0.831 | 220x / 16.8x |

The chosen student has 36% of the teacher's parameters and 41% of its gzipped bytes. Distillation is worth a size: the
s64 architecture trained from scratch with the teacher's own two-stage recipe scores well below the distilled s64 on
all three measures, and below s48. The polish adds 0.025-0.034 UTMOS, most of it to the two ARCTIC voices (bdl and
slt), which pure distillation loses most; intermediate-feature matching (weight 0.05) adds about 0.02, and the two
barely stack. In an architecture experiment, quantization-aware training at 4 bits (groups of 64) recovered about 70%
of the UTMOS that 4-bit weights lose, at a 45% smaller module; the shipped voice weights stay at 8 bits per row (the
QAT and the packed 4-7-bit export were removed after 1.6.4 - see git history).

The student's training was later changed - Muon on its hidden matrices with a warmup-stable-decay schedule, 12,000
steps instead of 20,000 of AdamW and a cosine (Section 11.4) - and that change, not a new size, is most of the
difference between picovox 1.0.0's voice and 1.0.1's (Section 12.5).

**A decoder rung.** On a phone without a JIT (Section 12.4), the decoder - per frame, unlike the encoder's per token -
was over half the voice network's time, so the ladder gained rungs that shrink only it, keeping s80's encoder (each on
one teacher, polished, through the panel):

| Student (s80's encoder, dim 80, 3 layers) | Parameters | Module (gzip) | UTMOS | WER | Similarity |
| --- | --- | --- | --- | --- | --- |
| s80 - decoder 80 wide, 4 layers (two seeds) | 349,547 | 432 KB | 4.144, 4.165 | 1.46%, 1.26% | 0.977, 0.973 |
| **d64 - decoder 64 wide, 4 layers (three runs) - chosen** | 305,035 | 385 KB | 4.147, 4.126, 4.135 | 1.06%, 1.41%, 1.26% | 0.972, 0.968, 0.972 |
| L3 - decoder 80 wide, 3 layers | 317,947 | 397 KB | 4.136 | 1.56% | 0.974 |
| d64-L3 - decoder 64 wide, 3 layers | 283,851 | 361 KB | 4.095 | 1.76% | 0.964 |

d64 averages 0.02 UTMOS below s80 - about the seed spread - with a lower word error rate, and its voice network runs
1.24 times faster on an iPhone without a JIT; it ships in picovox 1.2.0. (Its third run is the recipe's own student, retrained to
check the recipe: the same losses through step 1,000 and within 0.02% at the end, but not bitwise the same weights - the
voice's training on the GPU is not bit-reproducible, which the model's export and packaging, from a checkpoint, are.)


## 9. Grapheme-to-phoneme conversion without a dictionary

### 9.1 Dictionary misses

The first front end looked each word up in the CMU Pronouncing Dictionary (about 135,000 words). Misses are not rare in
books, and they are the words that matter most - character names that repeat hundreds of times:

| Book | Words missing from the dictionary | Most frequent misses |
| --- | --- | --- |
| *Moby-Dick* | 3.1% | Stubb, Queequeg, Pequod, harpooneer, Bildad |
| *Frankenstein* | 1.3% | Clerval, endeavoured, Safie, labours, Ingolstadt |
| *Pride and Prejudice* | 1.2% | Longbourn, Netherfield, Meryton, Pemberley, Rosings |
| *The Great Gatsby* | 1.1% | Wolfshiem, Klipspringer, Eckleburg |

The letter-based fallback mispronounced them badly (Queequeg became `K AH1 EH1 K AH1 EH1 G`). Across 22 books (2.36
million words), 1.86% of word tokens miss the dictionary (11,956 distinct words); British-to-American spelling rules
alone resolve 3.4% of those. A pilot letters-to-phonemes network (an encoder-decoder of 178K parameters; 37.6% word
and 10.4% phoneme error on held-out dictionary words, which are mostly hard names) already changed how the books
sounded. We spoke six sentences built from frequent misses with the neural male voice, with the old fallback and with
the pilot's pronunciations, and transcribed them with Whisper:

| Sentence | Old fallback, as Whisper heard it | Pilot network, as Whisper heard it |
| --- | --- | --- |
| *Pride and Prejudice* | "walked from lung bar into net hair failed ... saw Pemberle" | "walked from Longburn to Netherfield ... saw Pemberley" |
| *Moby-Dick* | "Clack-clack and stubs sailed on the peck-odd with TAS-TAGAW and DAGAW" | "Pekwigan stubs sailed on the Peekwad with Tashtego and Daggu" |
| *Moby-Dick* | "Captain Bill Dad ... the hair punters obeyed them" | "Captain Builded ... the Harpoon ears obeyed them" |

UTMOS rose from 3.96 to 4.05 (mean of the six).

### 9.2 Against the classical approach

A Phonetisaurus joint-sequence WFST (a Kneser-Ney 7-gram over aligned letter-phoneme pairs, trained on the same split)
was more accurate than any of our first neural models - 34.4% test word error and 9.0% phoneme error, where order 7 is
the plateau - but it is 9.1 MB in a compact 8-bit form (6.2 MB gzipped). Size decides it for the browser:

| Model | Size | Test word error |
| --- | --- | --- |
| Neural pilot (178K parameters, 8 bits) | 164 KB | 38.0% |
| Neural encoder-decoder (657K parameters) | about 650 KB | 36.1% |
| WFST, pruned | 209 KB (gzipped) | 56.3% |
| WFST, pruned | 1.6 MB (gzipped) | 39.2% |
| WFST, full order 7 | 6.2 MB (gzipped) | 34.4% |
| **Distilled student (219K parameters, 5 bits)** | **197 KB (149 KB gzipped)** | **33.9%** |

At equal size the network won by about 18 points. The two families also err differently - they agree on 58.5% of test
words (right 85.8% of the time there), and an oracle choosing the better of the two would reach 23.6% - so we combined
the WFST, a free and deterministic teacher, with five neural models into one teacher (27.96% test word error, too large
to ship) and distilled a small student from it. The student (219,383 parameters, 5-bit weights in groups of 64, beam
2) reached 33.9% at 197 KB - 2.2% of the WFST's size.

### 9.3 One network for every word

Held-out word error measures the wrong thing for books. Most of what a listener hears is the frequent words, which the
dictionary has; the dictionary itself is 3.6 MB (915 KB gzipped) - more than the rest of the system. So we asked one
network to pronounce *every* word: trained on every dictionary word, weighted toward the words books use, it must
reproduce the dictionary on the frequent words and generalize on the rest.

Used alone, the best network at that point (a CTC model of 453,511 parameters, trained for held-out accuracy) pronounced
29.9% of book tokens differently from the dictionary. The network that replaced the dictionary is an autoregressive
encoder-decoder (embedding 32, a bidirectional GRU of 96 over the letters, a GRU decoder of 192 with attention; 369,767
parameters). It trains on every dictionary word but the 2,000-word monitoring set, sampled half uniformly and half by
the square root of the word's count in 128 Project Gutenberg books, plus - 30% of the samples - 100,812 words the
dictionary lacks, labeled by the combined teacher of Section 9.2 or by the spelling and `-s` rules where they apply. A
base stage (20,000 steps of 512 at learning rate 3e-3) is polished twice with more of the samples on the most frequent
words (5,000 steps over the top 20,000 words, then 5,000 over the top 12,000), and exported at 6 bits per weight. It
mispronounces 0.05% of book tokens (2.4% of the distinct book words), in 275 KB gzipped; a twin trained without the test
split scores 35.2% held-out word error - about the generalization of the held-out models, with the dictionary's coverage
of the frequent words. Training on one CPU core is deterministic, so the recipe reproduces the shipped weights bit for
bit. The player has shipped without the dictionary since this network; since picovox 1.0.0 the library has none even
under Node.

### 9.4 What readers say

*Through version 1.6, from the human readers. Since 2.0 the readings are measured in the training speech (Section
13.3).*

The dictionary's first pronunciation is not always the usual reading: its first "lead" is the metal, its first "lives"
the verb. For each word with several dictionary pronunciations, we counted which one the LibriTTS-R readers used, by
the forced alignments, and relabeled the word where the readers' majority is statistically clear (the Wilson 95% lower
bound above one half, from three or more speakers), the difference is lexical (not vowel reduction or stress alone,
not a function word), and the vowel's formants (or the fricative's voicing) agree where measured. A last training stage
(3,000 steps) teaches the 56 relabeled words (`train/data/g2p-relabel.json` - lead L IY1 D, lives L AY1 V Z, live,
wind, wound, tears, record, content, subject, address, ...); all 56 come out right, at no cost to the held-out error.

### 9.5 Heteronyms by context

*The aligned examples below were the human readers' through version 1.6; since 2.0 they are the training speech's,
held out by book (Section 13.3).*

Some words have readings their neighbors decide - "I read it yesterday" and "I will read it", "their lives" and "she
lives", "a lead pipe", the noun/verb stress pairs. The G2P module holds two small additions for them, and no word list
or rule at runtime: the G2P takes a reading index (a learned vector added to its decoder's first state; reading 0 is
the default) and has an ambiguity head (a 16-unit layer on the encoder's final states that says whether the word's
reading depends on its context). For a word the head flags, a context network chooses the reading: a character GRU
over the three words and phrase-end marks on each side, a bidirectional GRU, and a layer gated by the word's own
vector - 47,779 parameters at 6 bits. Ambiguous words are 0.52% of the tokens of *Pride and Prejudice*, at 0.24 ms each
in Node.

The labels cover 46 ambiguous words (27,461 examples before the seeds of Section 9.9, built from pinned texts - Section
10.3) from three sources: the readers' readings by the forced alignments; part-of-speech tags over 132 Project Gutenberg
books (spaCy, at training time only), used for a word only where the tag-to-reading mapping agrees with the aligned
audio on at least 95% of at least 10 tokens (29 words); and 450 hand-written seed sentences and reviewed collocations
for the readings neither can separate (lead the metal, bass, bow, row, does the deer). The aligner cannot judge
word-final voicing - where it disagreed with the tags on use, close, and excuse, hand checks found the aligner wrong in
most cases - so S/Z-only readings are left to the front end and to the regression set.

| Regression set | No context network | With it |
| --- | --- | --- |
| Default readings | 48/52 | 48/52 |
| Context readings | 4/50 | 44/50 |
| Missing readings | 0/3 | 3/3 |

The regression set is hand-written, so the context network is also gated on held-out speech: in a 10-fold
cross-validation by speaker, it chooses the aligned reading of held-out readers' "read" and "lives" (the reviewed
words, less the S/Z-only pairs) 89.3% of the time against the majority reading's 59.1%; the G2P chain requires 20
points over the majority.

### 9.6 Auditing the dictionary: "Hobbes"

*The audit ran on the human readers' speech; its fixes stand as committed. Since 2.0 its audio step uses the training
speech (Section 13.3).*

A listener reported that the audiobook player read "Hobbes" as "hobbies". The network was not wrong about its training
data: the dictionary's first entry is HOBBES HH AA1 B IY0 Z - "hobbies" - while its second variant, HOBBS, and every
other -bes word after a consonant (COOMBES, ERBES, FORBES, ZIRBES) end B Z. The network had learned the label
faithfully, and generalized from it: it read "mobbes", a word it had never seen, as M AA1 B IY2 Z. A network that
reproduces the dictionary reproduces its errors, so we audited the dictionary (`picovox_train/g2paudit.py`):

1. **Screening by cross-validated disagreement.** Two partitions of the dictionary into five folds (a word's
   possessives in its fold) train ten screening models (the shipped recipe's network and sampling, 6,000 steps each),
   so every word is held out by two of them (held-out accuracy 62%, any listed pronunciation, stress included). A word
   is a suspect when both its models' phones differ from its label (not by stress alone) by a log-probability margin of
   3, or both prefer another of its listed variants by that margin: 12,750 of 124,101 words, 3,070 of them in the
   books - mostly legitimate irregulars (of, colonel) and foreign names.
2. **Adjudication by readers' speech.** Where three or more LibriTTS-R readers spoke a suspect, a CTC phoneme
   recognizer scores the label and each lexically different alternative (the models' predictions and the dictionary's
   variants) on up to 12 of the word's aligned tokens, with their neighbors. A token votes for a candidate at least 2
   (log) more likely than the rest; a confirmed error needs the votes' Wilson 95% lower bound above one half, from three
   or more speakers.
3. **Controls.** 140 words where both models agree with the label, scored against a model's second guess, measure the
   method: no false confirmations, and 1.5% of the tokens' votes (14 of 912) for the wrong candidate. (A first
   setting of the adjudication confirmed one control falsely; the final setting confirmed none.)

The audio confirmed 143 labels and 13 errors; 12,463 suspects had no audio evidence. With HOBBES - on the dictionary's
own evidence, as it is not in the corpus - the audit made 14 fixes (`train/data/g2p-fixes.json`, each with its evidence):

| Word | Dictionary | Fixed | Evidence (votes for the fix, speakers) |
| --- | --- | --- | --- |
| hobbes | HH AA1 B IY0 Z | HH AA1 B Z | the dictionary's variant, HOBBS, the -bes words; both fold models |
| picturesque | P IH1 K CH ER0 AH0 S K | P IH2 K CH ER0 EH1 S K | 14 of 15 tokens, 10 speakers |
| reverend | R EH1 V ER0 AH0 N D | R EH1 V R AH0 N D | 13 of 13 tokens, 8 speakers |
| sancho | S AA1 N K OW0 | S AE1 N CH OW0 | 15 of 16 tokens, 4 speakers |
| danglars | D AH0 NG L AA1 R Z | D AE1 NG G L ER0 Z | 7 of 12 tokens, 4 speakers |
| leant | L IY1 AH0 N T | L EH1 N T | a third candidate (below) |
| actual, grandeur, inexplicable, mortification, rattler, serenely, superfluous, unreasonable | | | 4-6 votes each, 3-6 speakers |

"Leant" shows the method's care: the audio rejected the label, but neither the models' candidate (L IY1 N T) nor the
label was what the readers said - the recognizer's own phones were l ɛ n t - so a third candidate, scored the same way,
won. The fixes are applied to the labels in every training stage, and the words derived from a fixed word follow it
(HOBBES'S HH AA1 B Z IH0 Z). A names regression set (73 names with verified pronunciations, HOBBES first) guards them;
its minimum started at the shipped network's 63/73, so it can only go up.

### 9.7 The listener in the loop

Most suspects had no audio. From them we rendered a shortlist of 58 - frequent book words and names from the player's
catalog - with the voice, each pronounced the dictionary's way and the models' way, for the listener. Five more fixes
came of it: welsh (W EH1 L SH, not "welch"), basil (B EY1 S AH0 L), auguste (AH0 G AH1 S T), and two the listener
pronounced neither way - gough (G OW1, rhyming with "go") and monday (M AH1 N D EY2). The shortlist's most frequent
entry was not a word: "Nov." (2,697 tokens in the books) - so the front end now spells out the month and day
abbreviations with their period (Jan. to Dec.; Mon., Tue., Thu., Fri. - not Mar., Sat., Sun., or Wed., which are also
words). The rest of the 58 kept the dictionary's pronunciation.

### 9.8 The retrained G2P

The 19 fixes were applied and the whole G2P chain retrained from the pinned texts (29 minutes); measured against the
corrected labels:

| Measure | Before the fixes | After |
| --- | --- | --- |
| Book-token error | 0.081% | 0.062% |
| Top-10K words wrong | 21 | 9 |
| Names | 63/73 | 67/73 |
| Context readings | 44/50 | 45/50 |
| Default readings | 48/52 | 48/52 |
| Distinct book words wrong | 622 | 636 |
| Monitoring WER (held out) | 42.15% | 42.80% |
| Cross-validated gate ("read", "lives") | 89.3% | 88.8% (majority 59.1%) |

The book and frequent-word errors fell and the names rose, at a small held-out cost - about the spread between training
seeds, but outside `make train`'s reproduction tolerance (0.5 points of monitoring WER), so the author adopted the
retrained module over the check; the names minimum rose to 0.91.

### 9.9 "The lead story" and a steadier context network

The listener heard "They read the lead story" with the metal. The fix was in the labels: 20 hand-labeled seeds of
lead's leading sense before a noun (lead story, role, singer, actor, article, ...) and 4 more of the metal
(lead pencil, paint, poisoning, crystal), with four regression cases. Retrained (5 minutes), the module reads
lead story, role, singer, and article as L IY1 D while the metal stays L EH1 D; the default readings went from 48/52 to
50/52 and the context readings from 47/54 to 48/54, with the book measures unchanged. But two other cases flipped (fresh
produce, the sow), and more seeds for those destabilized the network: a retrain with 19 more seeds fell to 45/52 on the
default readings, below the gate. The small context network is unstable - its readings can move when unrelated labels
change; picovox 1.0.1 carries the lead fix without the produce and sow seeds.

The instability was the training's seed, not the data. Trained with five seeds on the same labels (with the produce and
sow seeds), the network's choices differed, a pair of seeds, on 11.0% of the ambiguous words' 805 windows in the two
held-out books and on 12.8 of the regression set's cases; the default readings ranged from 47 to 51 of 52. The seeds
worked - "fresh produce" and "the sow" were right in all five - but each run drew its own borderline cases. Label
smoothing, dropping context words at random, a moving average of the weights, stronger regularization, and distillation
from an ensemble on the training examples alone each moved the churn little (screened on 6,650 windows of the
training books: 12.7-14.9% a pair, against 13.9%) - on its own examples the ensemble agrees with the labels, so its
soft targets add nothing there.

What steadied it was distillation on unlabeled text. Four teacher networks, each its own seed, are trained first; the
shipped network - the same 47,779 parameters, the only one exported - is trained toward their mean probabilities (80%
of the loss; the rest, the labels' cross entropy) on its examples and on 27,997 unlabeled transfer windows (the
ambiguous words' other windows in the training books, up to 3,000 a word), half a batch of them in each batch, with
context words dropped at random (15%). Five seeds each:

| Context network (five seeds) | Churn, held-out books | Churn, regression cases | Default (of 52) | Context (of 54) | Held-out examples (of 805) |
| --- | --- | --- | --- | --- | --- |
| Trained alone | 11.0% | 12.8 | 48.6 | 45.4 | 764.2 |
| Distilled (four teachers) | 5.0% | 3.8 | 50.2 | 48.8 | 773.6 |
| A full batch of transfer windows | 4.8% | 3.0 | 51.0 | 48.8 | 769.6 |
| Eight teachers | 4.9% | 4.2 | 50.2 | 48.2 | 772.4 |

(Means over the seeds; churn is per pair of seeds.) The variants were no steadier, and eight teachers cost twice the
training, so the recipe keeps four and half a batch. The steadier network also made the labels' gaps readable: two
cases it got wrong in four of five seeds - "Please read the letter aloud" and "They followed his lead" - and "The bow of
the ship", wrong in the build's seed, were cases 1.0.1 had right by its seed's luck. Seeds for them (22 sentences:
read's imperative, lead's following sense, a ship's bow) put all three right in all five seeds, with the default
readings 51.4 and the context readings 50.2 on average and a churn of 5.6% and 3.2 cases.

Retrained with `make train-g2p` (the G2P network and its reading stage unchanged), the module is picovox 1.0.2's:

| Measure | picovox 1.0.1 | picovox 1.0.2 |
| --- | --- | --- |
| Default readings | 50/52 | 52/52 |
| Context readings | 48/54 | 50/54 |
| Missing readings, names | 3/3, 67/73 | 3/3, 67/73 |
| Cross-validated gate ("read", "lives") | 89.3% | 90.7% |
| Held-out examples | 763/805 | 769/805 |
| Book-token error, top-10K wrong | 0.062%, 9 | 0.062%, 9 |
| Module | 419,535 bytes | 419,575 bytes (8 more gzipped) |

Four cases changed, all to right - fresh produce, the sow, "the muscles contract", "a bass in the lake" - and none
broke. (The gate's 89.3% for 1.0.1 is from fresh folds: the cross-validation had kept earlier folds' results whatever
their inputs, and now keeps a fold only when its labels and settings are the same.) The cost is training time: the
teachers, each fold's too, make the context stage 25 minutes instead of 6 - still beside the 30-minute G2P network.

### 9.10 A listener's book: Don Quixote

A listener started Don Quixote (John Ormsby's translation, 457,619 words) and heard it mispronounced. Every word of the
book through the package's front end and its G2P showed three kinds of fault:

- **The front end.** Its words were letters a to z, so an accented letter split a word: "señor" (511 times) was "se" and
  "or", "Doña" was "do" and "a", "Amadís" was "amad" and "s". The front end now reads an accented Latin letter as its
  base letter (á as a, ñ as n) and the ligatures as English says them (Cæsar, Œdipus), a letter for a letter, so the
  training port's offsets are unchanged; texts without such letters sound bit for bit the same. The same pass found a
  crash: "constructor" - a word, but also a property every JavaScript object inherits - matched the abbreviations
  table's lookup, so any text containing it made `synthesize` throw; the lookups now check the table's own keys.
- **A dictionary word never learned.** "Quixote" (2,188 times, and "Quixote's" 147) was "kwik-SOTE", though the
  dictionary has K IY0 HH OW1 T IY0: the irregular spelling, against "quixotic" and the rules of English spelling, had
  only the 23 occurrences of the 128 books to weigh it. Majordomo, Luis, caballero, Galatea, and Malaga were the same.
- **Words the dictionary lacks.** Rocinante (188 times) was "RAH-sə-nənt", Cide Hamete "side HAM-et", Luscinda
  "loo-SCHIN-da"; "goatherd" was "goath-erd" and "shouldst" "showldst".

The fixes are data, as the dictionary's were (Section 9.6). A new file, `train/data/g2p-books.json`, holds the words of
the books listeners hear: each book's word counts (the front end's words that train - the dictionary's, the labels', and
their possessives), which join the frequency weights, and labels for its words the dictionary lacks or has wrong -
61 for Don Quixote, its names as American readers say anglicized Spanish (CMUdict's own convention: QUIXOTE, SENOR) and
its archaic words' standard pronunciations. A label trains as a dictionary word, weighted by its count (an unknown
word's teacher label is sampled with a hundred thousand others, too rarely to be learned), and a possessive the
dictionary lacks takes its stem's target. The monitoring set is drawn as before and loses only the book's seven
words. The first retrain learned the book's names and pulled six unlabeled words toward Spanish ("wineskins" as
"WIN-is-kins", Barcelona with a CH); labels for four of them (two are dictionary words) and the retrain put all six
right.

The same retrain carried the context network's open cases. "Tears" the verb (He tears the page) had 10 seed sentences
beside 2,899 nouns in the books' tags - below the 2% an alternate reading needs to make a word ambiguous - so the
network never saw the word; 18 more seeds made it one. "Increase" and "progress" have no tag labels (the audio doesn't
validate the noun/verb stress mapping), and their verb seeds were infinitives ("began to increase"); 18 seeds of the
finite verb ("Our troubles increase daily") and 8 of the noun fixed both. A minute amount went to the collocations.
These seeds cost "a bass in the lake" and "Polish the silver", the network's borderline cases moving with its labels
(Section 9.9); 12 seeds for them and a collocation (caught a bass) put both right and moved "He spoke Polish at home" to
the verb. Against the books' part-of-speech tags, on 15 ambiguous words with a clear tag mapping (30,180 tokens), the
module reads 96.97% of the tokens the tag's way (picovox 1.3.0: 96.82%) - "tears" 2,495 of 2,497 (2,464), "increase"
406 of 462 (391) - and "read" and "lives" (4,407 tokens) the same.

| Measure | picovox 1.3.0 | picovox 1.3.1 |
| --- | --- | --- |
| Default readings | 52/52 | 52/52 |
| Context readings | 50/54 | 52/54 |
| Missing readings, names | 3/3, 71/86 | 3/3, 80/86 |
| Cross-validated gate ("read", "lives") | 90.7% | 88.8% |
| Book-token error, distinct words, top-10K wrong | 0.062%, 636, 9 | 0.058%, 600, 10 |
| Don Quixote's labeled words (tokens) | 3 of 61 (43 of 1,780) | 48 of 61 (1,592 of 1,780) |
| Module | 419,575 bytes | 419,475 bytes |

(The names include 13 of Don Quixote's, which 1.3.0 had 3 of right; the gate's drop, 4 of its 215 aligned tokens, is
within its fold-to-fold spread, and the books' tags read "read" and "lives" the same.) The thirteen labels not learned
are rare names - Maritornes, Clemencín, Benengeli, at most 31 times each - a small network's limit. Of the book's
457,642 spoken tokens 7,412 changed: Quixote, Rocinante, and the other labels, the dictionary words now as the
dictionary has them, and the stress of a few unlabeled names.


### 9.11 A listener's reports: gravest, mindset, paprika

A listener heard three words wrong. "Gravest" (G R AE1 V AH0 S T - "gravvest") was the dictionary's error, learned: its
stem and the stem's other forms have EY1 (GRAVE, GRAVER, GRAVELY, GRAVESTONE), as BRAVEST does, so
`train/data/g2p-fixes.json` corrects it on the dictionary's own evidence (LibriTTS-R has three of its tokens, too few for
the audit's bound). "Mindset" (M IH1 N D S EH2 T) and "paprika" (P AE2 P R IH0 K AA1 AH0) were the network's own errors
on words the dictionary has right - words the frequency weights barely reach: a dictionary word with no count in the
128 books is wrong 18.9% of the time (14,074 of 74,332), one counted 6 to 20 times 4.7%, and one counted over 100 times
almost never. "Mindset" has no count, "paprika" 6.

Retrained with the fix (`make train-g2p`), the module reads all three as the dictionary does - mindset M AY1 N D S EH2
T, gravest G R EY1 V AH0 S T, paprika the dictionary's second, P AH0 P R IY1 K AH0 - but mindset and paprika came right
by the retraining's churn, not by a weight of their own: the one changed label moved 740 of the 30,000 most frequent
words' pronunciations (326 to their targets, 314 from them, mostly a reduced vowel or a secondary stress). The gate
measures, against the committed module:

| Measure | picovox 1.6.3 | Retrained |
| --- | --- | --- |
| Book-token error, distinct words, top-10K wrong | 0.058%, 601, 10 | 0.059%, 604, 9 |
| Monitoring WER (held out) | 41.75% | 42.25% |
| Default, context readings; missing, names | 52/52, 52/54; 3/3, 80/86 | 52/52, 51/54; 3/3, 80/86 |

The monitoring WER, ten more of its 2,000 held-out words, is at the reproduction tolerance (0.5 points) - the spread
between training runs (Section 9.8) - so the check failed by a hair, and the module was adopted over it.

The weights' gap is modern words, and the player reads Wikipedia. On a year of Wikipedia's featured articles and four
weeks of its most-read (776 articles, 2.9 million dictionary-word tokens), the shipped module's token error is 1.10%
(the retrained module's 1.05%) - against 0.058% on the books - wrong on putin (P Y UW1 T IH0 N), muslim, psychiatrist,
debuted, and podcast. Counting Wikipedia's words into the weights as a listeners' "book" fixes most of them - the G2P
network's stages alone, 0.94% without them, 0.60% with two weeks' articles, 0.36% with the year's (measured on the
year's) - and paprika exactly (P AE0 P R IY1 K AH0), but costs the books: their book-token error rose from 0.043-0.046%
to 0.056% and 0.066%, past the gate's tolerance (0.01 points). Taking that trade - and measuring the gate on Wikipedia's
text too - is left open.

## 10. Reproducible training: `make train`

### 10.1 One recipe, one command

`make train` recreates both generated models - `lib/models/g2pModel.js` and `lib/models/voiceModel.js`'s modules, in
`build/train/` - from a clean checkout, by one recipe, `train/data/train.json`, which holds every stage's settings and
seeds. Each stage is a Makefile target with a stamp, so `make -j train` runs independent stages at once; the GPU stages
run one at a time under a lock (two at once each run at about half speed), and the CPU stages run beside them:

| Stage | Runs on |
| --- | --- |
| the training speech - generated by the local TTS model and aligned (Section 13; before 2.0, the corpora: CMU ARCTIC, LibriTTS-R and its alignments, CREMA-D) | GPU (once, then cached) |
| the heteronyms' readings, measured in it | GPU (once, then cached) |
| the voices' data | CPU (every core) |
| teacher A, teacher B | GPU |
| teacher evaluation | CPU |
| distillation targets, student, polish | GPU |
| student evaluation and the gate | CPU |
| package - the voice's module, with its provenance | CPU |
| listening samples of each stage's final model and of the shipped model | CPU, under a minute each |

The voice data is built with the committed G2P module. The G2P chain - the network, the heteronym labels, the context
network, its cross-validation, and the relabeling of Section 9.4 - runs beside the voice stages (`make train-g2p`
alone): it reproduces the G2P and gates it against the committed module. `make train` ends in a report - each model's
size, the voice's scores against the shipped model, and whether each module differs from the committed one - and
installs nothing: `make train-check` runs the package's tests with the trained models, then a blind A/B of the trained
voice against the shipped one in the voice lab, served locally (both with the shipped G2P; each of the twelve voices
speaks the text, of nine, where the two voices' parameter frames differ most, aligned by dynamic time warping, with three
repeats - fifteen forced choices, finished early after six Same answers in a row), and reports the listener's win
rate for the trained voice with its interval and sign test and a recommendation (no audible difference when at most a
fifth of the answers chose a voice); `make train-install` installs the trained modules
that differ (`make train-install-voice` and `train-install-g2p`, one).

The first run of this flow showed both checks earning their place: the trained voice ran 2% longer on one test sentence,
past a duration bound the shipped voice had met by 0.02 seconds (the bound was widened), and the G2P's context network
did not reproduce the committed module - the G2P network itself did, checkpoint for checkpoint. The committed module
predated version 1.6.5's reading fixes ("No." ends a sentence only before a number, an en dash between numbers is "to",
titles spelled out), which changed the context labels the books give through the front end; that release's check had
retrained and gated the new network but kept the module. The run reproduced that check's network exactly - the regression
set's context sentences 53 of 54 (51 before: "a minute amount", "spoke Polish"), 38 of 519,548 book words pronounced
differently, all heteronyms - and version 1.6.8 installs it, so `make train` reproduces both modules again. And its A/B
was thirty Same answers: a retrain of an unchanged recipe is, by ear, the shipped voice (its panel scores within noise -
UTMOS 4.131 against 4.146), so the A/B is a retrain's guardrail against a bad run, and the judge of a recipe change.

### 10.2 Gates and provenance

A retrain never silently replaces what ships:

- **The voice gate** compares the new student with the *shipped* model, spoken as the package speaks it: mean UTMOS
  within 0.05, WER within 1 point, and similarity to the new teacher at least 0.85. (It first compared the student with
  its own teacher, so a better teacher - the first from-scratch run's, 4.187 against 4.144 - failed an equal student.)
- **The G2P check** passes a module identical to the committed one below its comment header; otherwise it requires the
  regression set's minimums, no kind more than 2 cases below the committed module, and the book measures within the
  recipe's tolerances - and it keeps the committed module either way.
- **The held-out gate** of Section 9.5, and the relabeling's reproduction.
- **The install** is its own step, after the gates, the listening, and the package's tests with the trained models
  (`make train-check`): `make train-install`.
- **Provenance.** The packaged voice module records the commit it was trained from (and whether the tree was clean),
  the SHA-256 of the recipe and of the training data, and the gate's scores; picovox 1.0.1's names commit dddc189,
  clean, with UTMOS 4.1667, WER 1.1078%, and similarity 0.9784.
- **Power.** On a laptop, `make train` first checks the power adapter: under about 90 W an M3 Max cannot hold the GPU
  at full speed and charge the battery (Section 11.3).

### 10.3 Pinned texts

The training reads 133 Project Gutenberg texts (the heteronym labels' 132 books and the distillation's 22). Project
Gutenberg revises its books in place and keeps no old revisions; it revised *Crime and Punishment* (pg2554) on
2026-10-16, mid-project, which moved the heteronym labels by one example and changed the retrained context network.
So the texts are pinned: `train/data/books.json` records each text's source, the SHA-256 and size of its bytes, the date
pinned, its Gutenberg revision, and a byte-identical Wayback Machine capture where one exists (50 of 133). A text that
does not match its pin fails the stage, naming the book and both hashes. The pinned texts are also published as one
deterministic archive, the fetch's first source
([training-books-v1.tar.xz](https://craigahobbs.github.io/picovox/books/training-books-v1.tar.xz), 133 texts, 24.7 MB).

With the pins, the heteronym labels reproduce the shipped labels byte for byte (27,461 examples), the distillation's
book inputs match the first run's token for token (254,166 items, in order), and the G2P chain, retrained from scratch
inside the second full `make train`, reproduced `lib/models/g2pModel.js` byte for byte. Exporting and packaging the
shipped voice checkpoint reproduces `lib/models/voiceModel.js` byte for byte; the voice training itself runs on the GPU,
whose backpropagation is not deterministic, which is why the voice gate compares quality rather than bytes.

### 10.4 Two from-scratch runs

| Run | Commit | Critical path | Teacher (UTMOS / WER) | Student (UTMOS / WER / similarity) | Shipped as |
| --- | --- | --- | --- | --- | --- |
| First | d56063e (2026-10-16) | 113.7 min, 60 W adapter | 4.187 / 2.11% | 4.123 / 1.61% / 0.969 | picovox 1.0.0 |
| Second | dddc189 (2026-10-17) | 71.2 min, 94 W | 4.189 / 1.56% | 4.167 / 1.11% / 0.978 | picovox 1.0.1 |

Both students passed the gate against the model they replaced (the first against the first distillation's s80 of
Section 8: UTMOS 4.130, WER 1.71%). The second run's shorter time is Section 11's subject.


## 11. Training speed

### 11.1 Profiling the training step

A first profiling pass made a training step **1.73 times faster on fresh input shapes (0.430 to 0.249 s) and 2.07
times faster once the GPU's shape caches are warm (0.295 to 0.142 s)**, measured in interleaved A/B runs while sharing
the GPU with a live run. The largest hidden cost was not computation: on Apple's GPU (PyTorch's MPS backend), every
blocking host-to-device copy waits for the whole GPU queue to drain - we measured 130-190 ms waits - and the baseline
made about 50 such copies and 25 device reads per step, so the CPU and GPU took turns.

| Change | Step time |
| --- | --- |
| Synthesizer: a batch-vectorized glottal cycle plan, non-blocking uploads, cached constants, shared filter trigonometry | (enables the next) |
| Each step's data, crops, and cycle plans prepared ahead on a thread; losses read only when logged | -14% |
| A fused AdamW that skips non-finite steps on the GPU (no per-step gradient-norm read) | -9.5% |
| A Metal kernel for the harmonic sum (no 570 MB phase and sine tensors) | -11% |
| Metal kernels for the vocal tract's response (14 resonators fused, analytic backward) | -17% |

The optimized code computes the same thing: step-0 losses are bit-identical; over 240-step training runs the curves
differ from the baseline by less than the baseline differs from its own rerun (MPS backpropagation is
nondeterministic); the Python synthesizer still matches the browser renderer; and a run with the multi-style
settings (emotional data, adversarial critic, feature matching) matches to 4-5 significant digits on every loss.
`torch.compile` was tried first; its backward pass fails on MPS.

Two later passes, each checked the same way - bit-identical on the CPU, or a matched 2,000-step run inside the spread of
two control seeds - made teacher stage A 2.2 times faster again (a small model's stages 2-2.4 times): fixed-shape
batches (token and frame counts padded up a geometric ladder, so MPS compiles about 119 batch shapes instead of about
2,000), the critic's convolutions in bfloat16, the critic's spectrograms once per signal and the generator's pass
without the critic's weight gradients, shared STFTs between the spectral losses, two more Metal kernels (the frication
filter's magnitude, the harmonic amplitudes after the tract), and the synthesizer's cycle plan finished on the device
(no mid-step read). A third took 10-13% more off the teacher: the critic's strided convolutions' input gradient as two
stride-1 convolutions (MPS's own costs 2.8 times the forward), and the full-length spectral loss compiled. Updating the
critic only every second step did not hold: it weakened the critic and widened the formant bandwidths, the
over-smoothing the critic is there to fix. Outside the trainer, the readers' data build became 6.6 times faster when we
found it rebuilt a 1.7-second table for every utterance (Python's `setdefault` evaluates its default on every call).

A port of the trainer to Apple's MLX framework was also built: its quality matched, and in a fair A/B against the
optimized PyTorch trainer (interleaved, the same code, the same GPU load) it was 1.78x faster from a fresh start and 1.60x
warm - well short of an earlier, unfair 4x comparison against older PyTorch code under contention - with a memory
problem of its own (about 43 GB). The gain didn't justify maintaining two trainers, so the MLX port was dropped and the
PyTorch trainer kept and optimized instead.

### 11.2 The stage graph

Making each stage a target, so the CPU stages run beside the GPU stages, took the estimated critical path from about
175 minutes serial to about 118. The first full run's critical path was 113.7 minutes - readers' data 6.1, teacher A
27.5, teacher B 61.3, targets 4.0, student 8.1, polish 5.5, gate and package 1.1.

### 11.3 The machine: power and memory bandwidth

Two findings about the computer mattered more than several code changes.

**The slow mode was power.** Teacher A's step times were bimodal (about 0.078 against 0.108 s), and teacher B ran
10-45% slower than the same code had the night before. Thread priorities and deeper pipelining did nothing (0.0770
against 0.0771 s for A). Across 20 timed runs, the slow ones were exactly those with the battery charging hardest and
the system drawing least (39-45 W against 50-89 W in the fast ones): on a 45 W or 60 W adapter the power controller
caps the system to charge the battery, while a training run draws 64-87 W. An Apple 140 W adapter (which a 14-inch
MacBook Pro negotiates at about 94 W) removed it; `make train` now checks the adapter first.
On that adapter a second slow mode remained, and it is heat: about 2.5 minutes into a GPU stage drawing about 70 W,
macOS's thermal pressure rises to moderate, then heavy, the system is held near 40 W, and teacher A's step goes from
0.078 to about 0.12 s - at step about 2,000 in every run, for either seed - so the long GPU stages run throttled nearly
throughout.

**CPU work beside a GPU stage costs memory bandwidth, not cores.** On Apple's unified memory, three single-core compute
jobs beside teacher B cost nothing measurable, while three memory-bound jobs (large array copies) slowed it 53% at the
default scheduling priority, 15% at the utility QoS, and 1.3% at the background QoS. In the first run, teacher B ran
0.22-0.24 s per step beside the heteronym labels' tagging job against 0.153 alone. So the CPU stages beside the GPU run
at the utility QoS, and the G2P chain moved out of `make train` to `make train-check`: beside teacher A in the second
run it cost A about 9 minutes. It later moved back, as `make train-g2p`, so that `make train` trains both models (its
memory-bound stages after the readers' data build).

### 11.4 Ideas from the NanoGPT speedrun

The modded-nanogpt speedrun cut the time to train a GPT-2-small language model to a target loss more than sixtyfold,
largely by convergence improvements - the Muon optimizer, and learning-rate schedules that hold the rate constant and
cool it down at the end. Its record-to-record gains don't transfer by default to a small speech model driving a
synthesizer, so we screened the candidates, judging each by the **final student's panel scores** against two control
seeds (validation loss for screening only), and each speed change by interleaved timings on a quiet machine at full
power:

| Idea | Where | Result | Decision |
| --- | --- | --- | --- |
| Muon on the hidden matrices + warmup-stable-decay (WSD: constant, the last 25% cooled down by 1 - sqrt) | student | validation 0.0317 (two seeds) against the controls' 0.0333-0.0335; at 12,000 steps equal to the controls' 20,000 | **shipped**, 12,000 steps |
| The same | student, through the panel | 20K: 4.172 / 1.11% / 0.977 and 4.146 / 1.11% / 0.974; 12K: 4.147 / 1.11% / 0.974 - against the controls' 4.124 / 1.61% / 0.969 and 4.133 / 1.26% / 0.969 | |
| Teacher B length, WSD trunk with cooldown branches at 10K, 12K, 15K, 20K | teacher B | students 4.161, 4.136, 4.140, 4.138 UTMOS - no trend with length | **shipped**, 10,000 steps (was 20,000 cosine) |
| End to end: B 10K WSD + the 12K Muon/WSD student | the whole chain | 4.161 / 1.16% / 0.975 and 4.187 / 1.51% / 0.972 against the shipped recipe's 4.124 / 1.61% / 0.969 and 4.139 / 1.31% / 0.967 | **shipped** |
| A 16-channel critic (32 before) | teacher B, polish | B -20% per step; students 4.166 / 1.11% / 0.975 and 4.147 / 1.21% / 0.978 | **shipped** |
| A lean gradient clip (one norm, applied inside fused AdamW) | teachers | A -4.4% per step, B -2.3%; the student no faster | **shipped** for the teachers |
| Muon | teacher A | validation 2.038 at 5K against 2.069 and 2.066 (about 10 times the seed spread) - but its students 4.094 / 1.46% and 4.135 / 1.46%, below the AdamW teacher's 4.161 and 4.187 | rejected |
| Muon | teacher B | validation at 3K 1.995 and 1.994 against 1.997-2.001 - about the spread | not adopted |
| Batch 96, 10K steps | student | validation 0.0342, worse | rejected |
| Zero-initialized residual projections | teacher A | no effect | rejected |
| Weight EMA | student | -0.0001 validation at the end - too small | not adopted |
| Short crops early in A | teacher A | quality inside the spread at 5K; those steps -20% | measured, not adopted |

Teacher A's Muon is the lesson of Section 7.3 again: a clearly better validation loss gave a worse voice. The decision
belongs to the final student, heard through the panel, not to the loss of an intermediate model.

### 11.5 From 114 to 71 minutes

| Stage | First run (60 W) | Second run (94 W) | What changed |
| --- | --- | --- | --- |
| Readers' data | 6.1 | 7.3 | |
| Teacher A | 27.5 | 28.6 | the lean clip, but the G2P chain beside it (about +9) |
| Teacher B | 61.3 | 20.4 | 10,000 WSD steps (was 20,000), the 16-channel critic, the lean clip, full power |
| Targets | 4.0 | 3.6 | |
| Student | 8.1 | 4.9 | 12,000 Muon/WSD steps (was 20,000) |
| Polish | 5.5 | 4.4 | the 16-channel critic |
| Gate and package | 1.1 | 2.0 | |
| **Critical path** | **113.7** | **71.2** | |

The two runs differ in power as well as recipe, so the table attributes the time only roughly. Without the G2P chain
beside it, teacher A should take about 19 minutes (from its quiet step time), and `make train` about 62 - an estimate;
that run has not yet been made.


## 12. The package and its showcase

### 12.1 picovox

The package is [picovox on npm](https://www.npmjs.com/package/picovox) - 1.0.0 and 1.0.1 were both published on
2026-10-17 (UTC), and 1.0.2, with the steadier context network (Section 9.9), on 2026-10-19. It is ESM-only, runs in
current browsers and in Node 22 or later, and has no dependencies:

| API | |
| --- | --- |
| `synthesize(text, options)` | a Promise of `{samples, sampleRate}` |
| `stream(text, options)` | an async iterable of sample blocks, a sentence and a fraction of a second at a time; concatenated, exactly `synthesize`'s samples |
| `speak(text, options)` | Web Audio playback in a browser, `{stop(), finished}` |
| `voices()` | the twelve voices - male and female, and the ten designed voices (1.3.0) |
| `styles()` | the eleven speaking styles - neutral, angry, happy, sad, fearful, disgusted, whisper, soft, shouting (1.4.0), dialogue (1.5.0), and exclaim (1.6.1) |
| `toWav(samples, sampleRate)` | a 16-bit WAV file |

The options are the voice, the speaking style (or `'auto'`, expressive reading - Section 12.7), the speaking rate
(0.5-2), the noise seed, and the sample rate (24 kHz by default); the output is bit-for-bit deterministic in browsers
and Node. The package's audio is bit-identical to the
audiobook player's own synthesis path, from which it was extracted (76 cases). Version 1.6.1's tarball holds 22 files,
774,472 bytes compressed and 1,211,095 unpacked; its two generated model modules are 419,475 bytes (the G2P and context
networks, 316 KB gzipped) and 554,315 bytes (the voice, 385 KB gzipped), the rest is code and the README. "Under 1 MB"
is the package as downloaded, and the two models as a static host serves them gzipped; the raw model modules are
about 0.97 MB together.

### 12.2 Speed and streaming

On an M3 Max in Node, the voice network runs about 127 times faster than real time, the renderer about 97 times, and
synthesis end to end (network and rendering, 24 kHz) about 55 times (in 1.0.2, 94, 65, and 40 - Section 12.4's
changes for phones without a JIT help with one too); loading both networks takes about 0.1 s, and `stream` yields the first speech in
about 70 ms. Three changes made it so. The voice network streams: its decoder (per-frame layers and short
convolutions, no attention) runs a chunk at a time, each layer only half a kernel ahead of the one above, so speech
starts after the first third of a second of frames instead of the whole sentence - with frames bit-identical to the
whole-sentence decode. Rendering at 24 kHz instead of 44.1 kHz halves the renderer's cost (UTMOS within 0.02). And
the renderer and the network's linear layers were made 1.4 and 1.4-1.65 times faster with sample-for-sample identical
output.

### 12.3 On a phone

Device diagnostics from an iPhone (a startup timeline posted from the page) showed synthesis there about 4 times slower
than real time with the teacher-sized model at 44.1 kHz (a 9-word sentence: G2P 0.8 s, voice network 4.9 s, rendering
7.4 s), and three parallel workers each running 3 times slower than one - Safari gives a page about one core's worth
there. The distilled student, 24 kHz rendering, one worker on iPhones, and the speedups above brought it to **about 1.18
times real time**, measured in Safari on an iPhone with the 1.0.0 voice (1.0.1's has the same architecture and size) -
fast enough for continuous playback, with little margin. These diagnostics were served over plain http from the laptop
(to post the timeline back, which Safari blocks from an https page), so Safari ran them without a JIT (Section 12.4).
The player's playback had to change too: iOS muted Web Audio while a keep-alive media element held the audio session, so
the speech plays through the media element; a late block now delays the rest of its sentence instead of being dropped
(on a device slower than real time every block after the first underrun had been dropped, so the clock ran in silence);
and a streaming sentence starts only once its synthesized audio covers playing it through at the measured synthesis
speed, so a slow device pauses between sentences rather than mid-word.

### 12.4 Without a JIT

The iPhone's speed puzzled us: a phone CPU is a few times slower than an M3 Max, not thirty. A performance lab
(`tools/perf/`) - a backend on the laptop serving the repository to the phone over the local network, the page polling
it for commands (timings of each pipeline stage in a worker, as the player runs them) and posting the results back -
found why. A fixed arithmetic loop took about 93 ms on the phone and 1 ms on the laptop, and never got faster when
repeated: Safari ran JavaScript without a JIT, interpreting every operation (not Lockdown Mode - WebAssembly ran, though
interpreted too, about 3 times JavaScript's speed, which we did not pursue). Built-in functions ran at native speed
(`Math.cos` was only 4 times slower than the laptop's), so the cost was the number of JavaScript operations per second
of audio. Node without its JIT (`node --jitless`) ran the pipeline within 20% of the phone, so changes were made against
it and checked on the phone:

- **Voicing from tables.** Each output sample summed about 95 harmonics, twice (the two frames it falls between) -
  about 1,900 operations a sample. A frame's voicing is one period of a waveform of the glottal phase, so it is now
  computed once per frame by an inverse FFT (two frames at once, as the real and imaginary parts; about 8 points per
  harmonic) and read at the phase with cubic interpolation: within 4e-5 of the summed voicing (SNR 96-99 dB on speech -
  4 points per harmonic gave 78 dB but exceeded the renderer's 1e-4 parity test against the training synthesizer).
- **A radix-4 FFT**, 1.7 times faster interpreted, for the noise filtering's two FFTs per 128-sample hop.
- **The noise filters' gains** from squared magnitudes as cosine polynomials, the per-sample noise draws inline, and the
  buffers read directly - the noise sample for sample as before.
- **The network**, bit-identical: the decoder's input layer once per token, attention four keys at a time, and the
  convolution's taps summed in locals.
- **Two workers.** The network and the renderer now take about equal time, so they run at once in two workers - the
  package exports the two halves (`sentenceFrames`, `createSentenceRenderer`), and the player runs a render worker
  beside each synthesis worker, which streams it frame chunks as the network makes them. Two such pipelines at once
  (four workers) gave 3.7 times real time in all, but each sentence slower; the player uses one.

| On an iPhone without a JIT | 1.0.2 | 1.1.0 | 1.2.0 |
| --- | --- | --- | --- |
| Voice network | 3.6x real time | 4.1x | 5.4x |
| Renderer | 2.1x | 3.8x | 4.6x |
| Synthesis, one worker | 1.4x | 2.0x | 2.6x |
| Synthesis, network and renderer in two workers (warm - the texts three times) | - | 2.9x | 3.1-3.4x |

Version 1.2.0 adds two things. The renderer's voicing tables became 4 points a harmonic with cubic B-spline
interpolation (each harmonic divided by the B-spline's response, sinc^4, before the table's inverse FFT, so the
interpolation passes it unchanged): more accurate than 1.1.0's 8 points and Catmull-Rom (SNR 105-110 dB against the
summed harmonics, against 96-99) with half the table FFTs; the noise's inverse FFT became a real transform (half the
spectrum, a half-size FFT), and the noise draws and gains a little cheaper - the renderer 1.2 times faster. And the
voice is the d64 student (Section 8): the network 1.24 times faster. With the two apart in workers, the slower one
sets the speed, so neither alone moved the pipeline much (with only the renderer faster, it was 3.0x).

**Why there was no JIT.** The lab serves its page over plain http, from the laptop on the local network, and that is the
cause: **Safari on the iPhone runs JavaScript without a JIT on a page served over plain http, and with one over https.**
The player's worker runs the lab's fixed loop in 4 ms on its https site and in 103 ms served over plain http from the
laptop (iOS 18.7, Safari 27, in the same hour; its posted log names the device), and the lab's own bench over http takes
about 100 ms on the page and in a worker. This is documented nowhere we found - pages tried on a phone over local http
are usually unminified development builds, which take the blame for being slow. So the table above, and Section 12.3,
measured the interpreted case - Safari over http, and in Lockdown Mode (which disables the JIT) - not what the player's
https visitors get. The changes still count: they made synthesis faster with a JIT too (Section 12.2), with the audio
unchanged. How much faster than real time an iPhone runs with its JIT was not measured; the lab must serve https for
that.

In the player, five minutes of a book played through with no stall. Two things measured did not make it: the noise
filtering at half the overlap (the renderer 1.4 times faster, but the noise itself changed - SNR 25-31 dB against the
current filtering - and the network is the bottleneck once the two run apart), and two whole-sentence workers instead of
a pipeline (1.6 times one worker in all, each sentence slower). Unlike Section 12.3's earlier measurement, an iPhone in
Safari 27 ran two workers at nearly twice one's speed and four at about 2.4 times, so the player's one worker on iPhones
is now one pipeline of two.

### 12.5 picovox 1.0.1's voice

| Voice (panel, scored against the second run's teacher) | UTMOS | WER | Similarity |
| --- | --- | --- | --- |
| picovox 1.0.0 | 4.124 | 1.61% | 0.937 |
| **picovox 1.0.1** | **4.167** | **1.11%** | **0.978** |
| The teacher (965,803 parameters) | 4.189 | 1.56% | 1 |

The new voice has the same architecture, size (349,547 parameters, 433 KB gzipped), and speed (the network 79 times real
time, old and new alike, in the gate's own measurement); it comes from the second from-scratch run (Section 10.4), with
the voice data pronounced by the corrected G2P (Section 9.8). Version 1.0.1 also ships the corrected G2P with the lead
fix and the spelled-out month and day abbreviations; 1.0.2 changes only the context network (Section 9.9).

### 12.6 PicoShelf

[PicoShelf](https://craigahobbs.github.io/picoshelf/) - every public-domain book, read aloud in the browser - is the
audiobook player that motivated the work, now a site of its own built on the published package. It reads books from
Standard Ebooks (EPUB, fetched in byte ranges) and Project Gutenberg (plain text), with a 2,097-book catalog published
as static JSON. Synthesis runs in a Web Worker through the package's own pipeline and sentence splitting, so the app
and the package share one implementation (its audio bit-identical to its earlier worker path, and its sentence splits
identical on 12 books' 95,924 units). It prepares the first sentences' audio when a book opens, so Play starts
from audio already synthesized. Positions are kept as character offsets into the book's text, so the player can skip
by time, jump to chapters, bookmark, and resume after a reload without having synthesized the audio in between. A
static site cannot resolve a bare `import 'picovox'`, so PicoShelf commits a copy of the installed package (pinned to
1.0.x).

**Packaging models as generated JavaScript.** Each network ships as a generated JavaScript module: a small manifest and
the packed weights as base64, decoded (and decompressed with the built-in `DecompressionStream`, which browsers and
Node both provide) by one loader - so the same `import` works in Node, the page, and the worker, with no `fetch` or
file-system code. General-purpose compression barely helps quantized weights: deflate shrinks the 8-bit voices by
only 3-5% (the shipped voice's payload by 3.3%), because quantized weights look random. Base64 adds 33% raw but about
1% after the HTTP compression that static hosts apply to JavaScript. Size reductions must therefore come from fewer
parameters and fewer bits per weight - distillation (Section 8) and, for the G2P, 6-bit weights.

### 12.7 Expressive reading: style `'auto'`, and a dead end

A narrator does not read a novel in one voice. Style `'auto'` (1.5.0) reads by the text's structure: its quotations
(lib/sentences.js `quoteSpans` - double quotation marks, curly or straight, a speech left open across paragraphs
followed into the next) in a *dialogue beat*, a quoted sentence ending in "!" in a livelier *exclamation beat*
(1.6.1), and each paragraph as one arc (the *paragraph beat*). Every choice
follows the text, never chance, so the reading is deterministic, and without `'auto'` the audio is unchanged, bit for
bit.

**The beats.** The dialogue style is a mild lift of the voice - pitch +7%, pace +5%, a little pressed and 1 dB louder;
each paragraph's first sentence is lifted (pitch +8%, 2 dB), its pitch settles to -4% by its last sentence, which is
7% slower. Both were chosen by one listener in A/B listening (Pride and Prejudice's first chapter for the dialogue beat,
Moby-Dick's for the paragraph beat - six paragraphs of narration, where a subtler and a stronger arc were both
preferred to the flat reading, the stronger slightly). The panel can't hear monotony - it scores sentences - but it
guards the voice: in four voices the dialogue style's UTMOS is within 0.04 of neutral's (4.13-4.31), its word error
rate unchanged (0.6-1.5%).

**The exclamation beat.** The voice model already marks an exclamation, but only by pitch: across twelve lines in four
voices, a sentence ending in "!" is 0.5-1.4 semitones higher than the same sentence ending in ".", yet no louder (at
most 0.3 dB) and no wider in pitch (its range 0.0-0.8 semitones narrower). So the beat adds energy rather than height:
the exclaim style is 3 dB louder, a little more pressed (3 dB of press), pitch +9% and its range 20% wider (each frame's
pitch moved from the sentence's mean in log scale - transforms.js `pitchRange`, the sentence's frames gathered first),
and 6% quicker. One listener preferred it to the 1.6.0 reading on Treasure Island's "The Last of the Blind Man", a
mild and a stronger version (a 40% wider range, 5 dB) both, and chose the mild: on the panel, read over whole texts,
the strong version cost the bright voices most (Amy's UTMOS 0.27, Kate's 0.21), the mild one at most 0.13, every word
error rate at most 2.1%. It is not shouting - the shouting style is strained (10 dB of press) and high, the right voice
only for a real shout, which no classifier could find (below); the exclamation beat is the right voice for nearly
every "!".

**A dead end: choosing a quotation's emotion.** The plan was a third layer - a small network reading a quotation angry,
shouting, or fearful where it was confident. A 27-billion-parameter chat model (qwen3.8, run locally by Ollama, its
answer constrained to the four styles, its probabilities from the answer's log-probabilities) labeled 24,119 quotations
of the 132 pinned books' 236,598, sampled by part (exclamations with a high-energy word nearby, other exclamations,
plain quotations - each weighted back to the books' mix); on 300 hand-labeled quotations its average precision for
styled against neutral is 0.86 (chance 0.19). A wrong style is worse than none, so the goal was precision, not recall:
each style's threshold had to show a one-sided 95% lower bound (Wilson's, at the weights' effective count) of 95%
precision over every label held out once (five folds), at the books' mix. Nothing reached it:

| Classifier | Hand-set AP | Best 95% bound - shouting / angry / fearful |
| --- | --- | --- |
| The student - a hashed word table (4,096 x 16) over the quotation, its attribution, and its context, mean and max, eight cues, a 32-unit layer; 69,028 parameters, 93 KB, the table pretrained on the attributions' words | 0.58 | none above 0.5 |
| ... the attribution taken from the next sentence ('"Help!" he screamed.' - the splitter ends a sentence at the exclamation) | 0.64 | 0.92 / 0.06 / 0.33 |
| A linear classifier on a 4-billion-parameter embedding model's vectors (qwen3-embedding, 2,560 dimensions) | 0.78 | 0.93 / 0.74 / 0.83 |
| A 256-unit MLP on the same vectors | 0.77 | 0.91 / 0.74 / 0.89 |

The embedding probes are a ceiling on any small student that imitates a large encoder, and even they fall short - so
the network was dropped, with its training and labels (1.6.0; style `'auto'` is the two beats, its audio unchanged).
Two observations. Shouting came closest because it is signaled on the surface (*he shouted*, *roared the captain*);
anger and fear live in what a line means, which a bag of words can't read. And the gate measures agreement with the
teacher, whose own errors cap it: among the student's top shouting picks, the teacher called "Bravo!" *shouted
Charles* and "The paper! Of course!" *yelled Holmes* neutral. A punctuation rule is no substitute: 81% of the
quotations ending in "!" are neutral by the teacher, 7% shouted.


## 13. Training on synthetic speech

Through version 1.6 both networks learned from human recordings, and from those recordings' tools: the voice's phone
durations came from LibriTTSLabel's forced alignments of LibriTTS-R, the G2P's heteronym evidence from the same
alignments (Sections 9.4 and 9.5), and the emotional styles from CREMA-D's actors. Neither network could improve past
its corpora and their aligner. Version 2.0 replaces all three: `make train` generates its own training speech from a
local text-to-speech model and measures it with its own tools, and both networks train from that speech alone.

### 13.1 A pilot and its control

Before the switch, one pilot and one control tested whether a TTS model's speech helps. The pilot added twelve voices of
Gradium's Phonon (a local TTS model served by Ollama; Apache 2.0) - 9.3 hours of book sentences - to the 237 readers'
56 hours; the control trained the same recipe on the readers alone. Each ran the full `make train` voice chain against
the shipped model:

| Run | Training speech | Student UTMOS | Student WER |
| --- | --- | --- | --- |
| The shipped model (1.6.8) | the readers and CREMA-D | 4.131 | 1.36% |
| Control - the same recipe retrained | the readers and CREMA-D | 4.136 | 1.56% |
| Pilot - Phonon's twelve voices added | the readers, CREMA-D, and 9.3 h of Phonon | 4.151 | 0.96% |

The control matched the shipped model, as the retrain of Section 10.1 had. The pilot's gains, small but on the right
side of both measures, came from the synthetic speech. The pilot's voices still sounded like the shipped ones: each
shipped voice was a blend of the readers' rows, which the pilot left as they were, so the new speakers added little but
pronunciation. A voice that sounds like a TTS model's voice has to be trained on that voice alone, so 2.0 trains on
the synthetic speech only.

### 13.2 The training speech

`picovox_train/tts.py` makes the corpus from the recipe's `tts` section: the engine (an Ollama URL, the model, and its
digest - the pin), a TTS voice for each of the package's twelve voice keys, and the hours per voice (4).

- **The texts.** Book sentences from the pinned books, first passed through the package's own front end (Section 9), so
  the engine speaks exactly the words the tokens spell - numbers, abbreviations, and dashes already read out. The panel
  texts and the listening samples' texts are excluded.
- **The heteronyms first.** Coverage quotas fill first, rarest word first: 200 sentences for each of the context gate's
  words (read, lives), 24 for each context network word, and 12 for each other measured word. A random fill from the
  distillation's books makes up the rest of the hours.
- **Stable under change.** Each clip's voice and split (5% test, 5% validation) come from a hash of its text, and its id
  from a hash of the voice and the text, so a larger corpus or a changed selection adds clips without regenerating the
  ones it keeps. The engine is not deterministic, so the clips are generated once and cached
  (`downloads/tts/<model>-<digest>/`; `make clean` keeps them).
- **Measured with our own tools.** Each clip is force-aligned to its tokens by CTC with a wav2vec2 phoneme recognizer
  (`facebook/wav2vec2-lv-60-espeak-cv-ft`, the model that aligned CREMA-D before) and checked: its greedy phones against
  the expected phones (an error rate above 0.35 is garbled), a truncated ending, and clipping. Only 21 of 29,475 clips
  failed, all garbled.

Phonon speaks at about 40-78 times real time on the M3 Max's GPU. The corpus - 29,475 clips, 46.1 hours, about 4 hours a
voice - took about an hour of the GPU to generate and 26 minutes to align, once. On the listener panel the training speech itself scores
UTMOS 4.43 on its held-out book sentences (16 per voice) and a Whisper word error rate of 4.2% (the books' archaic words
and names, which the panel texts avoid, cost Whisper more than any voice does). Section 3.2's human readers score
3.8-4.4 on the same measure.

### 13.3 Heteronyms measured in synthetic speech

The G2P's heteronym evidence (Sections 9.4 and 9.5) came from the human readers' alignments: which reading of "lead",
"live", or "close" each reader said. In 2.0 it comes from the training speech, measured by `picovox_train/measure.py`.
Each heteronym token's two readings are classed by how they differ, and each class has its own measure:

- **strong** - a full vowel or consonant differs (lead, live, wind): the recognizer's CTC vote, the likelihood of each
  reading's phones over the token's aligned audio, with a margin;
- **voicing** - S against Z only (close, use, house): the same vote, with strict voicing tokens;
- **weak** and **merger** - a reduced against a full vowel, or vowels the recognizer merges: the vowels' formants
  against each voice's own vowel space;
- **stress** - stress alone (conduct, insult): not measured.

Each measure is checked before it is trusted. The controls are words with one pronunciation, scored against a reading
with one phone swapped, and a measure is used only if at most 3% of 100 or more control votes choose the swap. The CTC
vote erred on 6 of 882 controls (0.7%) and the voicing vote on 6 of 866 (0.7%); both are used. The formants erred on 54
of 568 (9.5%) and are not used, so the weak and merger classes go unmeasured, like stress. Of the 23,346 heteronym
tokens measured, the strong and voicing votes judge 8,897.

**Relabeling.** A word is relabeled - its dictionary's first pronunciation replaced by what the voices say - when the
majority reading is significant (a Wilson 95% lower bound above one half) in votes from at least 3 books. The book is the
independent unit, not the voice, since the twelve voices share one engine. Function words, names (words the books
capitalize), and the context network's words are never relabeled: names would teach the network spellings it cannot
see, and the context network reads its own words from context. A committed word the corpus cannot judge (its class
unmeasured, or too few votes) keeps its earlier evidence, marked kept. The result is 98 words: 43 by the CTC vote, 7 by
the voicing vote, and 48 kept from 1.6's 56. Two of 1.6's words were dropped ("getting", "paths"), and 44 are new.

**The context labels.** The context network's aligned examples now come from the training speech, held out by book (two
held-out books, and the cross-validation's folds by book, not by speaker). The TTS model is a fallible reader: it read
"feet of lead" with the verb's vowel, and its readings disagree with the curated collocations on 11 of their 21 tokens.
So the curated words (the context network's own) learn from their seeds, collocations, and hand-written mappings, and
the training speech teaches only the noun/verb families and the gate's two words. The labels: 129 ambiguous words, 810
validated mappings, 54,731 examples.

**The G2P's result.** The first retrains failed the regression set's context sentences (36, then 44 of 54, against
the committed module's 53), and three faults the human evidence had hidden came out:

- **Too many context words.** With no measure of stress, admitting every noun/verb stress pair as a context word gave
  the network 129 words (1.6 had 47) at the same 48K parameters, and the curated words lost to them. A stress word
  is now a context word only if its less common reading has at least 16 examples in the books: 86 words.
- **Contradicting labels.** A book token a collocation decides also got a part-of-speech label, sometimes the other
  reading - "minute" had 62, mostly "a minute amount" tagged as the noun. A collocation now decides its token alone,
  as it already did in the aligned speech.
- **Thin words.** The books rarely put "speak" before "Polish", or a plural noun before the verb "contract", and the
  silver labels are full of "to polish". Varied hand-labeled seed sentences give those readings, and a seeded reading
  always counts as common: the verb "subject", 59 examples beside 2,965 nouns, otherwise fell under the 2% share an
  alternate reading needs.

A larger context network (79K parameters) and doubled training steps did not help; the network's size is unchanged.

| G2P | Context set | Default set | Names | Book error | Top-10K wrong | Fresh probes |
| --- | --- | --- | --- | --- | --- | --- |
| 1.6.8 (the human readers' evidence) | 53/54 | 52/52 | 80/86 | 0.093% | 21 | 26/38 |
| **2.0 (the training speech's)** | **54/54** | 51/52 | 80/86 | **0.061%** | **7** | **34/38** |

The context set's 54 sentences are few, and its thin words flip between training seeds, so we also scored 38 fresh
sentences - both readings of polish, contract, winds, bass, minute, and invalid, none in the seeds or the set. The
committed module, which reads "She speaks Polish" as the verb, scores 26; the new one 34. Its misses are the next
G2P release's: "invalid" before a noun ("her invalid mother", whose adjective tag maps to "not valid"), "cold makes
the metal contract", "the bass swam away", and "every minute change". The cross-validated gate (read and lives, ten
folds by book) holds 86.5% against the majority reading's 50.7%.

### 13.4 One speaker row per voice

The model loses its style table and its blends. Each voice key is one row of a twelve-row speaker table, `male` (row 0)
the default; the teacher has 931,619 parameters and the distilled student 282,627 (305,035 before, with 328 speaker rows
and a style table). The voices are the TTS voices as the model learned them, with no pitch, tract, or rate adjustments
of their own - only the -5 dB source shelf every voice has had since 1.6.6.

Without CREMA-D the four emotions became transforms, as angry already was. Their transforms are the former emotion
embeddings' measured effects on the parameter frames - the mean change in pitch, pitch range, pulse, frication, level,
and duration over the panel texts in every voice - less the breathiness they added:

| Style | Transforms | Pace |
| --- | --- | --- |
| happy | pressed (+5), pitch x1.23, level -0.8 dB | 1 |
| sad | pitch range x0.93, pressed (+2), pitch x0.97, level -3.3 dB, frication +2.9 dB | 0.97 |
| fearful | whisper 0.15, pitch x1.23, level -2.4 dB, frication +2.9 dB | 1.1 |
| disgusted | pressed (+4.8), pitch x0.95, level -2.4 dB, frication +1.9 dB | 0.83 |

On the listener panel's texts, in the female and male voices, every style stays intelligible (Whisper's word error
rate 0.6-3.0%, whispering 2.4-3.9%), every voiced style's UTMOS is within 0.05 of neutral's but fearful's (0.08 and 0.15
below - the breathy, high voice UTMOS likes least), and every style but whisper and soft is within 4.1 dB of neutral's
loudness. Style `'auto'` (Section 12.7) keeps UTMOS within 0.04 of neutral's in all twelve voices, its word error rate
at most 2.4%.

### 13.5 Results

The voice chain ran in 83 minutes with the training speech cached (71 before - Section 11.5): the voices' data
10, teacher A 28, teacher B 25, the targets 7, the student and its polish 5 each. The student passed the gate against the
shipped model:

| Model | Parameters | Module (gzip) | UTMOS | WER | Similarity to its teacher |
| --- | --- | --- | --- | --- | --- |
| 1.6.8's model (human recordings) | 305,035 | 554 KB (385 KB) | 4.131 | 1.36% | - |
| The teacher (2.0) | 931,619 | 1,333 KB (1,003 KB) | 4.327 | 1.59% | 1 |
| **The student (2.0)** | 282,627 | 451 KB (334 KB) | **4.311** | **1.13%** | 0.983 |

Each voice, against its TTS source (the training speech's held-out book sentences; the voices on the panel texts):

| Voice | TTS voice | Sex | Median F0 | TTS UTMOS | TTS WER | Voice UTMOS | Voice WER |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `female` | Harper | female | 185 Hz | 4.47 | 3.6% | 4.38 | 0.6% |
| `sarah` | Juniper | female | 164 Hz | 4.42 | 4.0% | 4.34 | 1.2% |
| `helen` | Delia | female | 173 Hz | 4.42 | 6.1% | 4.31 | 1.2% |
| `amy` | Zoey | female | 188 Hz | 4.43 | 2.4% | 4.38 | 1.2% |
| `kate` | Hadley | female | 175 Hz | 4.50 | 3.2% | 4.41 | 1.2% |
| `grace` | Everly | female | 171 Hz | 4.44 | 3.3% | 4.28 | 0.9% |
| `male` | Trevor | male | 122 Hz | 4.40 | 3.9% | 4.27 | 1.2% |
| `james` | Miles | male | 130 Hz | 4.39 | 6.9% | 4.24 | 0.9% |
| `robert` | Preston | male | 103 Hz | 4.43 | 2.8% | 4.17 | 0.9% |
| `michael` | Micah | male | 116 Hz | 4.34 | 5.6% | 4.31 | 1.5% |
| `david` | Easton | male | 131 Hz | 4.43 | 5.8% | 4.33 | 1.5% |
| `thomas` | Russell | male | 115 Hz | 4.46 | 2.2% | 4.30 | 1.2% |

Each voice scores 0.03-0.26 below its TTS source's UTMOS (on different texts, so the comparison is loose), the
lowest-pitched voice losing the most (robert, 103 Hz: 4.43 to 4.17). In a
blind A/B (pre-rendered pairs of the shipped and the new voice, the same text, one trial per voice and three repeats,
fifteen forced choices) the listener chose the new voices in all fifteen.

### 13.6 What the switch costs

- **A model's reading errors become the teacher's.** The TTS model's mistakes ("feet of lead") are measured as readings.
  Seeds and collocations outrank them, and relabels need three books, but a consistent error in the engine can still
  teach the G2P a wrong reading.
- **The engine is not deterministic.** A regenerated corpus is a different corpus; its relabels and context labels can
  move at the margins, so the corpus is generated once and cached, and the relabel file names the corpus it came from.
- **The dependency moves.** LibriTTS-R and its aligner are gone, but the voices now come from Phonon, whose training
  data is undocumented. The model is one line of the recipe: any Ollama speech model with distinct voices can replace
  it, and a better local model would improve both networks with no code change.
- **The style evidence is frozen.** The four emotion transforms are measurements of embeddings learned from CREMA-D,
  kept as numbers; a new emotional corpus would have to be synthetic too.


## 14. Limitations and future work

- One human listener has rated the voices; the human results are indicative only. The shipped voices' listening is
  blind A/B against the voices they replace (fifteen trials for 2.0), not an absolute rating.
- **The training speech is one TTS model's.** The voices can be no better than Phonon's, and its reading errors and
  habits are the networks' (Section 13.6); a better local model is a recipe change, untried.
- UTMOS and the emotion model are imperfect proxies - UTMOS barely moved under rule calibration that a listener could
  hear, and the emotion model has likely seen CREMA-D. The gate's similarity is to each run's own teacher, so it does
  not compare across teachers.
- The source-filter synthesizer bounds the achievable quality (Section 5.4); neural vocoders avoid that bound at
  hundreds of times the size.
- **The context network's remaining errors.** Distilled from four teachers it is steadier (Section 9.9), but four
  context cases stay wrong ("prices increase", "a minute amount", "the talks progress", "he tears the page"), and its
  choices still move between seeds most for polish, lead, row, and bow - words whose labels are thin.
- **The dictionary audit is partial.** Most of the 12,750 suspects had no audio evidence; the listener heard 58. The
  names regression set and the readers' audio are the only checks on the long tail.
- **Training time.** Teacher A is now the longest stage (about 19 minutes alone, by estimate); a shorter or
  WSD-scheduled stage A is untested, as is the crop warm-up measured in Section 11.4.
- **Phones.** Interpreting JavaScript, an iPhone synthesizes at about 3.1-3.4 times real time with two workers and 2.6
  times with one (Section 12.4); the network and the renderer now take about equal time, so each further gain needs
  both. Safari on the iPhone interprets JavaScript over plain http and compiles it over https (Section 12.4); the
  speed of a phone compiling JavaScript - the player's case - was not measured.
- **Unmeasured heteronym classes.** The formant measure failed its controls (9.5%), so heteronyms that differ by a
  reduced vowel, a merged vowel, or stress alone are judged by no evidence from the training speech (Section 13.3).
- **Expressive reading** follows the text's structure only (Section 12.7): no quotation is read angry, shouted, or
  frightened - a classifier precise enough to choose them was not found, even on a large model's embeddings.
- The voices are American English only, by design.


## 15. Data and licenses

| Data | Use | License |
| --- | --- | --- |
| CMU Pronouncing Dictionary | the G2P network's training labels (not shipped) | BSD-style |
| Phonon (Gradium, through Ollama) | the training speech (2.0 on): every voice, and the heteronyms' readings | Apache 2.0 (its training data undocumented) |
| wav2vec2 phoneme recognizer (`facebook/wav2vec2-lv-60-espeak-cv-ft`) | the training speech's alignment, checks, and reading votes (CREMA-D's alignment before) | Apache 2.0 |
| CMU ARCTIC | training; targets (before 2.0) | CMU ARCTIC free license |
| LibriTTS-R (train-clean-100, dev-clean) | multi-speaker training; targets; the readers' pronunciations and the dictionary audit (before 2.0) | CC BY 4.0 |
| LibriTTSLabel alignments | phone durations; readers' readings (before 2.0) | none stated - used locally, not redistributed |
| CREMA-D | emotional styles; emotion-model calibration (before 2.0 - the emotion transforms are its embeddings' measured effects) | ODbL |
| Standard Ebooks, Project Gutenberg | the books read aloud; the G2P's frequency weights and heteronym labels; the distillation's sentences | public domain |

PicoVox is distributed under the MIT License.


## 16. Reproducing the results

```sh
npm install picovox                                             # the package
make commit                                                     # tests, lint, docs, 100% coverage of lib/ and train/
make build/venv-train.build                                     # make train's Python environment (the listener panel's too)
make -j train                                                   # both models from scratch: the training speech (Ollama running, the first time), the G2P, and the voice - the teacher, the student, the gate
make -j train-g2p                                               # reproduce the G2P chain and check it against the committed module
make train-check                                                # the package's tests with the trained models, and the voice lab staged to listen
make train-install                                              # the trained models that differ into lib/models/
PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.smoke    # every training entry point, a few steps on the CPU
node tools/unpack.js lib/models/voiceModel.js build/voice && node tools/speak.js -m build/voice "Call me Ishmael." -o ishmael.wav
```

The dictionary audit is `picovox_train.g2paudit` (folds, suspects, audio, adjudicate). The rule-based synthesizer's
tools (`tools/panel.js`, `tools/targets.py`, `tools/say.js`) were removed in commit 33fbfad, the single-speaker
trainer (`nn/train.py`) in commit 7f667db, and its exports (`data/nn/slt` and `bdl`, Section 5's voices) later; their
versions are in git history.

The training dashboard (`tools/dashboard/`) follows `make train` live: the stages and the critical path, CPU and GPU
utilization, loss curves, logs, and listening samples of each stage's final model beside the shipped model's.


## References

- D. H. Klatt, "Software for a cascade/parallel formant synthesizer," *JASA* 67(3), 1980.
  [doi:10.1121/1.383940](https://doi.org/10.1121/1.383940)
- D. H. Klatt, "Synthesis by rule of segmental durations in English sentences," ICPhS 1979 (Proceedings of the 9th
  International Congress of Phonetic Sciences, vol. 2; also in B. Lindblom, S. Öhman (eds.), *Frontiers of Speech
  Communication Research*, Academic Press, 1979). [ICPhS 1979
  proceedings](https://www.coli.uni-saarland.de/Phonetics/icphs/ICPhS1979/p9.2_290.pdf)
- G. Fant, J. Liljencrants, Q. Lin, "A four-parameter model of glottal flow," *STL-QPSR* 26(4), 1985. [KTH STL-QPSR
  archive](https://www.speech.kth.se/qpsr/1985/1985_26_4_001-013.pdf)
- G. Fant, "The LF-model revisited. Transformations and frequency domain analysis," *STL-QPSR* 36(2-3), 1995. [KTH
  STL-QPSR archive](https://www.speech.kth.se/qpsr/1995/1995_36_2-3_119-156.pdf)
- J. Engel, L. Hantrakul, C. Gu, A. Roberts, "DDSP: Differentiable Digital Signal Processing," ICLR 2020.
  [OpenReview](https://openreview.net/forum?id=B1x1ma4tDr)
- Y. Ren et al., "FastSpeech: Fast, Robust and Controllable Text to Speech," NeurIPS 2019. [NeurIPS
  proceedings](https://proceedings.neurips.cc/paper/2019/hash/f63f65b503e22cb970527f23c9ad7db1-Abstract.html)
- Y. Ren et al., "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech," ICLR 2021.
  [OpenReview](https://openreview.net/forum?id=piLPYqxtWuA)
- A. Łańcucki, "FastPitch: Parallel Text-to-Speech with Pitch Prediction," ICASSP 2021.
  [doi:10.1109/ICASSP39728.2021.9413889](https://doi.org/10.1109/ICASSP39728.2021.9413889)
- R. Yamamoto, E. Song, J.-M. Kim, "Parallel WaveGAN: A Fast Waveform Generation Model Based on Generative Adversarial
  Networks with Multi-Resolution Spectrogram" (the multi-resolution STFT loss), ICASSP 2020.
  [doi:10.1109/ICASSP40776.2020.9053795](https://doi.org/10.1109/ICASSP40776.2020.9053795)
- K. Kumar et al., "MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis" (the discriminator
  feature-matching loss), NeurIPS 2019. [NeurIPS
  proceedings](https://proceedings.neurips.cc/paper/2019/hash/6804c9bca0a615bdb9374d00a9fcba59-Abstract.html)
- I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularization" (AdamW), ICLR 2019.
  [OpenReview](https://openreview.net/forum?id=Bkg6RiCqY7)
- G. Hinton, O. Vinyals, J. Dean, "Distilling the Knowledge in a Neural Network," arXiv:1503.02531, 2015.
  [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- Y. Kim, A. M. Rush, "Sequence-Level Knowledge Distillation," EMNLP 2016.
  [doi:10.18653/v1/D16-1139](https://doi.org/10.18653/v1/D16-1139)
- K. Jordan, Y. Jin, V. Boza, J. You, F. Cesista, L. Newhouse, J. Bernstein, "Muon: An optimizer for hidden layers in
  neural networks," blog post, 2024. [kellerjordan.github.io](https://kellerjordan.github.io/posts/muon/)
- K. Jordan et al., *modded-nanogpt: Speedrunning the NanoGPT baseline*, software, 2024.
  [GitHub](https://github.com/KellerJordan/modded-nanogpt)
- A. Hägele, E. Bakouch, A. Kosson, L. Ben Allal, L. Von Werra, M. Jaggi, "Scaling Laws and Compute-Optimal Training
  Beyond Fixed Training Durations" (warmup-stable-decay and the 1 - sqrt cooldown), NeurIPS 2024. [NeurIPS
  proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8b970e15a89bf5d12542810df8eae8fc-Abstract-Conference.html)
- A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," NeurIPS 2019. [NeurIPS
  proceedings](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html); software:
  PyTorch 2.14.1, [pytorch.org](https://pytorch.org/)
- *MLX* (Apple), an array framework for Apple silicon, software.
  [ml-explore.github.io](https://ml-explore.github.io/mlx/)
- "IEEE Recommended Practice for Speech Quality Measurements" (the Harvard sentences), *IEEE Trans. Audio and
  Electroacoustics* 17(3), 1969. [doi:10.1109/TAU.1969.1162058](https://doi.org/10.1109/TAU.1969.1162058)
- T. Saeki et al., "UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022," Interspeech 2022.
  [doi:10.21437/Interspeech.2022-439](https://doi.org/10.21437/Interspeech.2022-439); the UTMOS22 strong predictor
  through SpeechMOS v1.2.0, [GitHub](https://github.com/tarepan/SpeechMOS/tree/v1.2.0)
- B. Desplanques, J. Thienpondt, K. Demuynck, "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
  TDNN Based Speaker Verification," Interspeech 2020.
  [doi:10.21437/Interspeech.2020-2650](https://doi.org/10.21437/Interspeech.2020-2650)
- A. Nagrani, J. S. Chung, A. Zisserman, "VoxCeleb: A Large-Scale Speaker Identification Dataset," Interspeech 2017.
  [doi:10.21437/Interspeech.2017-950](https://doi.org/10.21437/Interspeech.2017-950)
- J. S. Chung, A. Nagrani, A. Zisserman, "VoxCeleb2: Deep Speaker Recognition," Interspeech 2018.
  [doi:10.21437/Interspeech.2018-1929](https://doi.org/10.21437/Interspeech.2018-1929)
- M. Ravanelli et al., "Open-Source Conversational AI with SpeechBrain 1.0," *JMLR* 25(333), 2024.
  [JMLR](https://jmlr.org/papers/v25/24-0991.html); software: SpeechBrain 1.1.2,
  [GitHub](https://github.com/speechbrain/speechbrain); the speaker model,
  [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- Z. Ma et al., "emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation," Findings of ACL 2024.
  [doi:10.18653/v1/2024.findings-acl.931](https://doi.org/10.18653/v1/2024.findings-acl.931); the emotion2vec+ base
  model, [Hugging Face](https://huggingface.co/emotion2vec/emotion2vec_plus_base)
- A. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper), ICML 2023.
  [PMLR](https://proceedings.mlr.press/v202/radford23a.html)
- *whisper.cpp* (ggml-org), a C/C++ port of Whisper, with the ggml base.en model, software.
  [GitHub](https://github.com/ggml-org/whisper.cpp)
- *The CMU Pronouncing Dictionary* (cmudict.dict, commit 0f8072f, 2024), Carnegie Mellon University.
  [GitHub](https://github.com/cmusphinx/cmudict)
- J. Kominek, A. W. Black, "The CMU Arctic speech databases," ISCA SSW5, 2004. [ISCA
  Archive](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.html)
- H. Zen et al., "LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech," Interspeech 2019.
  [doi:10.21437/Interspeech.2019-2441](https://doi.org/10.21437/Interspeech.2019-2441)
- Y. Koizumi et al., "LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus," Interspeech 2023.
  [doi:10.21437/Interspeech.2023-1584](https://doi.org/10.21437/Interspeech.2023-1584)
- T. Hayashi, *LibriTTSLabel*: word and phone alignment labels for LibriTTS (Montreal Forced Aligner), dataset.
  [GitHub](https://github.com/kan-bayashi/LibriTTSLabel)
- M. McAuliffe et al., "Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi," Interspeech 2017.
  [doi:10.21437/Interspeech.2017-1386](https://doi.org/10.21437/Interspeech.2017-1386)
- H. Cao et al., "CREMA-D: Crowd-Sourced Emotional Multimodal Actors Dataset," *IEEE Trans. Affective Computing* 5(4),
  2014. [doi:10.1109/TAFFC.2014.2336244](https://doi.org/10.1109/TAFFC.2014.2336244)
- J. Zuluaga-Gomez, S. Ahmed, D. Visockas, C. Subakan, "CommonAccent: Exploring Large Acoustic Pretrained Models for
  Accent Classification Based on Common Voice," Interspeech 2023.
  [doi:10.21437/Interspeech.2023-2419](https://doi.org/10.21437/Interspeech.2023-2419); the classifier,
  [Jzuluaga/accent-id-commonaccent_ecapa](https://huggingface.co/Jzuluaga/accent-id-commonaccent_ecapa)
- A. Baevski, Y. Zhou, A. Mohamed, M. Auli, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
  Representations," NeurIPS 2020. [NeurIPS
  proceedings](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html)
- Q. Xu, A. Baevski, M. Auli, "Simple and Effective Zero-shot Cross-lingual Phoneme Recognition," Interspeech 2022.
  [doi:10.21437/Interspeech.2022-60](https://doi.org/10.21437/Interspeech.2022-60); the phoneme recognizer,
  [facebook/wav2vec2-lv-60-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft)
- A. Graves, S. Fernández, F. Gomez, J. Schmidhuber, "Connectionist Temporal Classification: Labelling Unsegmented
  Sequence Data with Recurrent Neural Networks," ICML 2006.
  [doi:10.1145/1143844.1143891](https://doi.org/10.1145/1143844.1143891)
- E. B. Wilson, "Probable Inference, the Law of Succession, and Statistical Inference" (the Wilson score interval),
  *Journal of the American Statistical Association* 22(158), 1927.
  [doi:10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953)
- *spaCy* (Explosion), industrial-strength natural language processing in Python, software: spaCy 3.8.16 with the
  en_core_web_sm 3.8.0 pipeline. [spacy.io](https://spacy.io/)
- *Standard Ebooks*, public-domain ebooks. [standardebooks.org](https://standardebooks.org/)
- *Project Gutenberg*, public-domain ebooks. [gutenberg.org](https://www.gutenberg.org/)
- *Internet Archive Wayback Machine* (the pinned texts' archived copies). [web.archive.org](https://web.archive.org/)
- P. Boersma, D. Weenink, *Praat: doing phonetics by computer*, software. [praat.org](https://www.praat.org/)
- Y. Jadoul, B. Thompson, B. de Boer, "Introducing Parselmouth: A Python interface to Praat," *Journal of Phonetics* 71,
  2018. [doi:10.1016/j.wocn.2018.07.001](https://doi.org/10.1016/j.wocn.2018.07.001); software: praat-parselmouth 0.4.7,
        [PyPI](https://pypi.org/project/praat-parselmouth/0.4.7/)
- P. Escudero, P. Boersma, A. S. Rauber, R. A. H. Bion, "A cross-dialect acoustic description of vowels: Brazilian and
  European Portuguese," *JASA* 126(3), 2009. [doi:10.1121/1.3180321](https://doi.org/10.1121/1.3180321)
- M. Iseli, A. Alwan, "An improved correction formula for the estimation of harmonic magnitudes and its application to
  open quotient estimation," ICASSP 2004. [doi:10.1109/ICASSP.2004.1326074](https://doi.org/10.1109/ICASSP.2004.1326074)
- J. Hillenbrand, R. A. Cleveland, R. L. Erickson, "Acoustic correlates of breathy vocal quality" (cepstral peak
  prominence), *Journal of Speech and Hearing Research* 37(4), 1994.
  [doi:10.1044/jshr.3704.769](https://doi.org/10.1044/jshr.3704.769)
- D. Hirst, "The analysis by synthesis of speech melody: from data to models" (the two-pass pitch range), *Journal of
  Speech Sciences* 1(1), 2011. [doi:10.20396/joss.v1i1.15011](https://doi.org/10.20396/joss.v1i1.15011)
- M. Bisani, H. Ney, "Joint-sequence models for grapheme-to-phoneme conversion," *Speech Communication* 50(5), 2008.
  [doi:10.1016/j.specom.2008.01.002](https://doi.org/10.1016/j.specom.2008.01.002)
- R. Kneser, H. Ney, "Improved backing-off for M-gram language modeling," ICASSP 1995.
  [doi:10.1109/ICASSP.1995.479394](https://doi.org/10.1109/ICASSP.1995.479394)
- J. R. Novak, N. Minematsu, K. Hirose, "Phonetisaurus: Exploring grapheme-to-phoneme conversion with joint n-gram
  models in the WFST framework," *Natural Language Engineering* 22(6), 2016.
  [doi:10.1017/S1351324915000315](https://doi.org/10.1017/S1351324915000315); software,
  [GitHub](https://github.com/AdolfVonKleist/Phonetisaurus)
