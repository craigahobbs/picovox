# Changelog

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
