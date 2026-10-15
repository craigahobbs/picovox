# CLAUDE.md

Notes for coding agents working in this repository.

## Project overview

voice-synth is a from-scratch formant speech synthesizer in JavaScript (no neural nets, no recorded speech, no runtime
dependencies) and a public-domain audiobook player web app built on it.

- `lib/` - the synthesizer and the pure (DOM-free) library code: the LF glottal source, the Klatt-style cascade
  synthesizer, phoneme tables, the lexicon (CMU Pronouncing Dictionary in `data/cmudict.dict`), text-to-speech
  (durations, transitions, intonation), and book-text preparation. Runs in Node and browsers. 100% test coverage.
- `static/audiobook/` - the audiobook player app (static files, ES modules, no build step): the interface (`app.js`), playback
  (`engine.js`), the synthesis worker (`worker.js`), listener state (`store.js`), the downloaded-book cache
  (`bookcache.js`), and the book sources (`sources/`). The root `index.html` shows `README.md` (MarkdownUp).
- `tools/` - Node command-line tools: speech output (`say.js`), analysis (spectrograms, spectra, pitch), and
  intelligibility evaluation with whisper.cpp (`eval.js`, `words.js`, `stops.js`, `nasals.js`, `prosody.js`).
- `test/` - unit tests (`node:test`), one `test<Module>.js` per `lib/` module.

## javascript-build

This is a [javascript-build](https://github.com/craigahobbs/javascript-build#readme) package. Read the javascript-build
skill before running tests, lint, coverage, or changing the Makefile:
[`../javascript-build/SKILL.md`](../javascript-build/SKILL.md) if that file exists, otherwise
[https://raw.githubusercontent.com/craigahobbs/javascript-build/main/SKILL.md](https://raw.githubusercontent.com/craigahobbs/javascript-build/main/SKILL.md).

Local Makefile overrides:

- `ESLINT_ARGS` - also `static/audiobook/` and `tools/`
- `GHPAGES_SRC` - `build/site/` (the published app and catalog)

`make commit` (test, lint, doc, cover) is the gate before committing.

## Conventions

- Code style: 4-space indent, single-quoted strings, single-quoted object keys, a short comment above each function and
  block, descriptive names (no one-letter names - `id-length`), lines at most 140 characters.
- Acoustic changes to `lib/` are judged by measurement, not by ear: `node tools/eval.js -v male --all` (and `-v female`)
  reports the word error rate with Whisper; compare averages over several seeds - single runs vary by about +/-2 points.
  Refactors (lint fixes, restructuring) must leave the synthesized audio bit-identical - compare a hash of
  `tools/say.js` output for both voices before and after.
- The browser app must keep working from plain static hosting (`python3 -m http.server` and GitHub Pages) - book texts
  are fetched from their sources with CORS, never through a proxy.
