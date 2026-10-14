// The synthesis Web Worker - loads the pronouncing dictionary, then synthesizes text paragraph by paragraph so playback
// can start before a long text is finished. The synthesis functions are exported so they also run under Node.
//
// Messages in: {'id', 'text', 'voice', 'rate', 'seed', 'room'}
// Messages out: {'type': 'ready'}, {'type': 'chunk', 'id', 'index', 'samples', 'sampleRate'}, {'type': 'done', 'id', 'count'},
// {'type': 'error', 'id', 'message'}

import {setDictionary, textToPhrases} from '../lib/lexicon.js';
import {speak} from '../lib/speak.js';
import {convolve, roomImpulse} from '../lib/room.js';
import {normalize} from '../lib/klatt.js';


// Split text into paragraphs (separated by blank lines), dropping those with nothing to say
export function splitParagraphs(text) {
    return text.split(/\n\s*\n/).map((paragraph) => paragraph.trim()).filter((paragraph) => textToPhrases(paragraph).length !== 0);
}


// Synthesize a passage - the same chain as tools/say.js: speak, then the optional room ambience
export function synthesizeText(text, {voice = 'male', rate = 1, seed = 1, room = false} = {}) {
    const {samples: dry, sampleRate} = speak(text, voice, {seed, rate});
    const samples = room ? normalize(convolve(dry, roomImpulse(sampleRate)), 0.89) : dry;
    return {samples, sampleRate};
}


// Synthesize text paragraph by paragraph, calling onChunk(index, samples, sampleRate) for each - returns the paragraph count
export function synthesizeParagraphs(text, options, onChunk) {
    const paragraphs = splitParagraphs(text);
    paragraphs.forEach((paragraph, index) => {
        const {samples, sampleRate} = synthesizeText(paragraph, options);
        onChunk(index, samples, sampleRate);
    });
    return paragraphs.length;
}


// The worker message loop (only when running as a worker)
if (typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope) {
    // Fetch the dictionary on startup
    const ready = fetch(new URL('../data/cmudict.dict', import.meta.url)).then((response) => {
        if (!response.ok) {
            throw new Error(`Could not load the pronouncing dictionary (${response.status})`);
        }
        return response.text();
    }).then((text) => {
        setDictionary(text);
        self.postMessage({'type': 'ready'});
    });
    ready.catch((error) => {
        self.postMessage({'type': 'error', 'id': null, 'message': error.message});
    });

    // Synthesize each request, posting each paragraph's samples as it is done
    self.onmessage = async (event) => {
        const {id, text, voice, rate, seed, room} = event.data;
        try {
            await ready;
            const count = synthesizeParagraphs(text, {voice, rate, seed, room}, (index, samples, sampleRate) => {
                self.postMessage({'type': 'chunk', id, index, samples, sampleRate}, [samples.buffer]);
            });
            self.postMessage({'type': 'done', id, count});
        } catch (error) {
            self.postMessage({'type': 'error', id, 'message': error.message});
        }
    };
}
