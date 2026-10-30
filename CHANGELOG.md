# Changelog

## 2.0.0 (unreleased)

- new voices - every voice is new: the voice network is trained on speech a local TTS model (Gradium's Phonon, through Ollama) speaks at train time, one of its voices for each of the package's twelve, instead of human recordings (CMU ARCTIC, LibriTTS-R, CREMA-D). The voice keys (male - the default - female, sarah, helen, amy, kate, grace, james, robert, michael, david, and thomas), the style keys, and the API are unchanged; each voice is its own row of the model's speaker table, with new descriptions in voices() - no blends of readers, and no per-voice pitch, tract, effort, pulse, or rate adjustments (every voice's -5 dB source shelf kept). On the listener panel's texts the voices' mean UTMOS is 4.31 (was 4.13) and their Whisper word error rate 1.13% (was 1.36%), and a blind A/B preferred the new voices in all 15 trials; the voice network is 282,627 parameters, 334 KB gzipped (was 305,035, 385 KB)
- the happy, sad, fearful, and disgusted styles are transforms only, like angry: the model has no style table - each style's transforms are the measured effects (pitch, pitch range, pulse, frication, level, and pace) the former emotion embeddings, learned from CREMA-D's actors, had on every voice; on the listener panel every style stays intelligible (word error rate 0.6-3.0%, whispering 2.4-3.9%) and every voiced style's UTMOS is within 0.05 of neutral's but fearful's (0.08-0.15 below); style auto's UTMOS is within 0.04 of neutral's in all twelve voices
- pronunciation: the G2P network and its heteronym context network are retrained from the training speech's measured readings (no LibriTTS-R): 98 relabeled words (43 by the phoneme recognizer's vote, 7 by its voicing vote, 48 kept from 1.6), and 86 context words (stress pairs only where their less common reading is common in the books); the regression set's context sentences 54 of 54 (was 53), book words mispronounced 0.061% (was 0.093%), 7 of the 10,000 most frequent words wrong (was 21); known miss: "invalid" before a noun ("her invalid mother")
- breaking, for imports of lib/ modules (the API, lib/index.js, is unchanged): lib/model.js conditions on a speaker only - its speaker blends ({id: weight}) and its style option removed; lib/synthesis.js's voice table is {key, label, sex, description} - a voice is the model's speaker of its key, with no blend, style, or adjustments; a 1.x voice module (lib/models/voiceModel.js) has none of 2.0's voices (its speakers are the human readers), and 2.0's isn't 1.x's; tools/speak.js loses --blend, --style, and --intensity
- training: make train generates its training speech - book sentences, the heteronyms' coverage first, about 4 hours a voice - from the local TTS model (pinned by digest, cached in downloads/tts/), aligns and checks it with a wav2vec2 phoneme recognizer, and measures its heteronym tokens' readings, the G2P's heteronym evidence; the human speech corpora, the speaker pool, and the CREMA-D pipeline are gone

## 1.6.8 (2026-10-27)

- [bd62a80](https://github.com/craigahobbs/picovox/commit/bd62a80) - pronunciation: the heteronym context network retrained on the context labels 1.6.5's reading fixes changed - "a minute amount of dust" and "he spoke Polish" now right (the context regression set 53 of 54, was 51); 38 of 519,548 words of the pinned books are pronounced differently, all heteronyms. The voice retrained from scratch by the new make train flow - no audible difference in blind A/B listening, its listener-panel scores within noise
- [961cfbf](https://github.com/craigahobbs/picovox/commit/961cfbf) - the package's internals simplified (the packed weights' decode and unused transform options removed), its audio bit-identical

## 1.6.7 (2026-10-23)

- [ab9f58b](https://github.com/craigahobbs/picovox/commit/ab9f58b) - no clicks in speak's playback: its audio context runs at the synthesis sample rate, so the blocks join exactly - a context at the device's rate resampled each block apart, and every join (each 85 ms) clicked; the README says to play stream's blocks the same way
- [a71531f](https://github.com/craigahobbs/picovox/commit/a71531f), [7285b69](https://github.com/craigahobbs/picovox/commit/7285b69) - the package's internals simplified: the single-speaker model support and the voice network's unused options (a raw speaker embedding, a channel equalizer, a formant bandwidth scale, and the pause threshold) removed, its audio bit-identical

## 1.6.6 (2026-10-23)

- [cb5a8e6](https://github.com/craigahobbs/picovox/commit/cb5a8e6) - less tinny voices: every voice's source 5 dB darker than the network predicts, and the Male and Female voices' glottal pulse a little softer - in blind A/B listening, less tinny in half the voices and never worse, never less natural, and the warble on drawn-out words like "times" gone; UTMOS, word error rate, and speaker similarity unchanged

## 1.6.5 (2026-10-23)

- [afa5018](https://github.com/craigahobbs/picovox/commit/afa5018) - reading fixes: a number of 309 or more digits no longer stops synthesis (numbers past 15 digits are read digit by digit); "I said no. Then I left." is two sentences ("No." is an abbreviation only before a number - "No. 5" as "number five"); an uppercase P.M. or A.M. ends a sentence; an en dash between numbers is "to" ("1812–15"); Rev., Capt., Col., Prof., Esq., Jr., Sr., Lt., Mt., Vol., and Fig. spelled out; MI, DI, MV, and DV aren't Roman numerals
- [d91ddc1](https://github.com/craigahobbs/picovox/commit/d91ddc1) - the package's internals simplified (one base64 and inflate, one encoder path, the unused openFraction removed), its audio bit-identical

## 1.6.4 (2026-10-22)

- [a7ed6d6](https://github.com/craigahobbs/picovox/commit/a7ed6d6) - parentheses pause: an aside's opening and closing parentheses are each a comma's phrase break ("He left the room (quietly, as always) and closed the door."), punctuation after a parenthesis keeping its own break
- [02e7436](https://github.com/craigahobbs/picovox/commit/02e7436) - pronunciation: gravest (the dictionary's "gra-vest" corrected), mindset, and paprika

## 1.6.3 (2026-10-22)

- [ad192ce](https://github.com/craigahobbs/picovox/commit/ad192ce) - money, percentages, and times: $1,234.56 as dollars and cents, $5 million as "five million dollars" (£ and € too), 50% as "fifty percent", 10:30 as "ten thirty" and 9:00 p.m. as "nine pee em"; "5 p.m. on Monday" stays one sentence; World War I as one, not the pronoun

## 1.6.2 (2026-10-22)

- [c7c57e8](https://github.com/craigahobbs/picovox/commit/c7c57e8) - number reading: a four-digit number reads as a year (1972 as "nineteen seventy-two"; 2000 to 2009 as numbers), commas group thousands (23,000,000 as "twenty-three million"), ordinals (1st, 22nd) and decades (1970s, 80's), and Roman numerals in running text (World War II, Henry VIII)

## 1.6.1 (2026-10-21)

- [85010b5](https://github.com/craigahobbs/picovox/commit/85010b5) - the exclamation beat: style "auto" reads a quoted sentence ending in "!" in the new exclaim style - louder, livelier in pitch, and quicker than the dialogue beat (excited, not shouted); the pitchRange transform

## 1.6.0 (2026-10-21)

- [27c79d1](https://github.com/craigahobbs/picovox/commit/27c79d1) - drop the style network: no classifier chose a quotation's emotion precisely enough (Section 12.7 of the paper). Style "auto" is the dialogue and paragraph beats, its audio unchanged; lib/style.js keeps autoStyles (now autoStyles(sentences)) and paragraphBeats, without loadStyleModel and the classifier's functions; the package is 774 KB again

## 1.5.0 (2026-10-21)

- [c1b38e8](https://github.com/craigahobbs/picovox/commit/c1b38e8) - expressive reading - style "auto": the quotations in the new dialogue style (a little higher, brighter, and quicker than the narration) and each paragraph read as one arc (its first sentence lifted, its last slower); a style network (93 KB, loaded only for "auto") reads a quotation angry, shouting, or fearful only where it's provably confident - in this release, none yet

## 1.4.0 (2026-10-20)

- [7f422d3](https://github.com/craigahobbs/picovox/commit/7f422d3) - speaking styles - the style option of synthesize, stream, and speak, and styles(): neutral, angry, happy, sad, fearful, disgusted, whisper, soft, and shouting, with any voice

## 1.3.1 (2026-10-20)

- [a34ba7c](https://github.com/craigahobbs/picovox/commit/a34ba7c) - better pronunciation of names and rarer words, Don Quixote's in particular (Quixote, Rocinante, Cide Hamete; goatherd, shouldst); accented letters read as their base letters (señor, Doña, Amadís were split into pieces); fix synthesize throwing on any text containing "constructor"; heteronyms read right in more contexts (he tears the page, prices increase, talks progress)

## 1.3.0 (2026-10-19)

- [5f89016](https://github.com/craigahobbs/picovox/commit/5f89016) - twelve voices - add Sarah, Helen, Amy, Kate, and Grace; James, Robert, Michael, David, and Thomas (blends of the model's audiobook readers, each with its own pitch, tone, and pace) beside male and female; voices() gives each voice's description

## 1.2.0 (2026-10-19)

- [c35e39d](https://github.com/craigahobbs/picovox/commit/c35e39d) - a smaller voice network - the d64 student, a 64-wide decoder (305,035 parameters, 385 KB gzipped - was 433 KB): its network 1.24x faster on an iPhone without a JIT, UTMOS within 0.02 of the 80-wide decoder's and a lower word error rate

- [3271126](https://github.com/craigahobbs/picovox/commit/3271126) - the renderer 1.2x faster on an iPhone without a JIT, and nearer the training synthesizer - synthesis 2.6x real time in one worker there (was 2.0x)

## 1.1.0 (2026-10-19)

- [24a8fc9](https://github.com/craigahobbs/picovox/commit/24a8fc9) - faster playback, most of all for JavaScript engines without a JIT - on an iPhone whose Safari interprets JavaScript, the renderer is 1.8x faster and synthesis runs at 2.0x real time (was 1.4x); 52x real time on an M3 Max (was 40x)

- [e5e66f9](https://github.com/craigahobbs/picovox/commit/e5e66f9) - add sentenceFrames and createSentenceRenderer (synthesis.js) - the voice network and the renderer can run in two workers at once (2.7x real time on that iPhone)

- [1c5e2c4](https://github.com/craigahobbs/picovox/commit/1c5e2c4) - add the performance lab (tools/perf) - time the playback pipeline on a phone over the local network

- [370e03c](https://github.com/craigahobbs/picovox/commit/370e03c) - paper: speech synthesis without a JIT (Section 12.4), the steadier heteronyms, and the 1.0.1 work

## 1.0.2 (2026-10-19)

- [6ea6bcd](https://github.com/craigahobbs/picovox/commit/6ea6bcd) - a steadier heteronym context network, the same size - "fresh produce", "the sow", "the muscles contract", and "a bass in the lake" now read right, and none read wrong

## 1.0.1 (2026-10-17)

- [8d11f7c](https://github.com/craigahobbs/picovox/commit/8d11f7c) - a better voice - UTMOS 4.124 to 4.167, Whisper word error rate 1.61% to 1.11%, the same size and speed

- [dddc189](https://github.com/craigahobbs/picovox/commit/dddc189) - pronunciation fixes for 19 pronouncing dictionary errors (Hobbes, Welsh, Gough, Monday, picturesque, ...)

- [1b6bb51](https://github.com/craigahobbs/picovox/commit/1b6bb51) - "lead" before a noun reads in its leading sense (lead story, lead role, lead singer)

- [5f593c9](https://github.com/craigahobbs/picovox/commit/5f593c9) - spell out the month and day abbreviations (Nov. as November, Mon. as Monday)

## 1.0.0 (2026-10-16)

- [4127a6b](https://github.com/craigahobbs/picovox/commit/4127a6b) - PicoVox 1.0.0 - realistic speech synthesis in pure JavaScript, under 1 MB
