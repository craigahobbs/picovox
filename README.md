# voice-synth

A from-scratch formant speech synthesizer in JavaScript, aiming for realistic male and female voices - no neural nets and
no recorded speech. Text goes through a rule-based front end (normalization, the CMU Pronouncing Dictionary, Klatt's
duration rules, intonation) to frames of synthesizer parameters every 5 ms, which drive a Klatt-style cascade
synthesizer with an LF glottal source.

It runs under Node (command-line tools and evaluation) and in the browser (Web Audio, synthesis in a Web Worker). There
are no dependencies.


## The audiobook player

A public-domain audiobook player, read aloud by the synthesizer. Serve the repository root statically and open it - the
root `index.html` redirects to `web/`:

```sh
python3 -m http.server 8000
open http://localhost:8000/
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
| `web/app.js` | The library and player interface |
| `web/engine.js` | Playback - the text timeline, on-demand synthesis, scheduling, skipping, speech normalization |
| `web/worker.js` | The synthesis Web Worker - unit (sentence) runs and whole-text synthesis |
| `web/sources/standardebooks.js` | Standard Ebooks - browse, search, and EPUB loading |
| `web/sources/gutenberg.js` | Project Gutenberg - Gutendex browse and search, GITenberg texts, boilerplate and chapters |
| `web/store.js` | The listener's state in browser storage - places, bookmarks, settings |
| `web/bookcache.js` | Downloaded books in IndexedDB |

`node tools/browsercheck.js` checks the browser build under Node: it loads every `lib/` module and `web/worker.js` with
`node:` imports forbidden, synthesizes with the worker's functions from a dictionary string, and checks the audio
matches the Node path sample for sample.


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


## The library

| Module | |
| --- | --- |
| `lib/glottal.js` | LF glottal pulse, parameterized by Rd |
| `lib/klatt.js` | Klatt-style cascade synthesizer - voicing, aspiration, nasal pole/zero, F1-F6 and higher formants, frication |
| `lib/phonemes.js` | ARPAbet tables - vowel formants, consonant loci and couplings, noise spectra |
| `lib/lexicon.js` | Text normalization, the pronouncing dictionary, function words |
| `lib/speak.js` | Voices, durations, stop phases, formant transitions, smoothing, intonation - `speak(text, voice, options)` |
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


## License

The CMU Pronouncing Dictionary is distributed under its own license, in `data/LICENSE.cmudict`.
