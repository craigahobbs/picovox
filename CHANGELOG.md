# Changelog

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
