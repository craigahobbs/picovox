// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

/** @module picovox */

import {defaultSampleRate, defaultStyle, defaultVoice, modelLoader, styleTable, textBlocks, voiceTable} from './synthesis.js';
import {encodeWav} from './wav.js';


// The models - loaded on first use, once
const loadModel = modelLoader(() => import('./models/g2pModel.js'), () => import('./models/voiceModel.js'));




// The speaking rate range, the default noise seed (the first sentence's - PicoShelf's), and the sample rate range (Hz)
const minRate = 0.5;
const maxRate = 2;
const defaultSeed = 1000;
const minSampleRate = 8000;
const maxSampleRate = 96000;


// The playback timing (s) - how far ahead of the audio context's time the first block starts, and how far ahead of
// playback synthesis runs
const playbackLead = 0.05;
const playbackAhead = 4;


/**
 * A voice of {@link voices}
 *
 * @typedef {Object} Voice
 * @property {string} key - The voice's key - the "voice" option's value
 * @property {string} label - The voice's display label
 * @property {string} sex - The voice's sex ("female" or "male")
 * @property {string} description - The voice's character, in a sentence
 */


/**
 * The built-in voices - the default ("male") first, then "female", then the designed voices (blends of the model's
 * audiobook readers - Sarah, Helen, Amy, Kate, and Grace; James, Robert, Michael, David, and Thomas)
 *
 * @returns {Voice[]} The voices
 */
export function voices() {
    return voiceTable.map(({key, label, sex, description}) => ({key, label, sex, description}));
}


/**
 * A style of {@link styles}
 *
 * @typedef {Object} Style
 * @property {string} key - The style's key - the "style" option's value
 * @property {string} label - The style's display label
 * @property {string} kind - "emotion" (a feeling - angry, happy, sad, fearful, disgusted) or "mode" (a way of speaking -
 * whisper, soft, shouting)
 * @property {string} description - The style, in a sentence
 */


/**
 * The speaking styles - neutral (the default) first, then the emotions (angry, happy, sad, fearful, disgusted) and the
 * modes (whisper, soft, shouting, dialogue, exclaim); any style goes with any voice
 *
 * @returns {Style[]} The styles
 */
export function styles() {
    return styleTable.map(({key, label, kind, description}) => ({key, label, kind, description}));
}


/**
 * The synthesis options of {@link synthesize}, {@link stream}, and {@link speak}
 *
 * @typedef {Object} SynthesisOptions
 * @property {string} [voice = "male"] - The voice's key (see {@link voices})
 * @property {string} [style = "neutral"] - The speaking style's key (see {@link styles}), or "auto" - the quotations' words
 *     in the "dialogue" style (an exclaimed sentence's in the "exclaim" style), the narration neutral, and each paragraph
 *     read as one arc (its first sentence lifted, its pitch settling, its last sentence slower)
 * @property {number} [rate = 1] - The speaking speed, from 0.5 to 2 (1 is normal, 1.2 is 20% faster)
 * @property {number} [seed = 1000] - The noise seed, an integer - sentence i of the text is rendered with seed + i
 * @property {number} [sampleRate = 24000] - The output sample rate (Hz), an integer from 8000 to 96000
 */


// Validate the text and the synthesis options - the options with their defaults
function synthesisOptions(text, options, allowed = []) {
    if (typeof text !== 'string') {
        throw new TypeError(`The text must be a string, not ${describe(text)}`);
    }
    if (options === null || typeof options !== 'object' || Array.isArray(options)) {
        throw new TypeError(`The options must be an object, not ${describe(options)}`);
    }
    const names = ['voice', 'style', 'rate', 'seed', 'sampleRate', ...allowed];
    for (const name of Object.keys(options)) {
        if (!names.includes(name)) {
            throw new TypeError(`Unknown option "${name}" - the options are ${names.map((item) => `"${item}"`).join(', ')}`);
        }
    }
    const {voice = defaultVoice, style = defaultStyle, rate = 1, seed = defaultSeed, sampleRate = defaultSampleRate} = options;
    if (!voiceTable.some((item) => item.key === voice)) {
        const keys = voiceTable.map((item) => `"${item.key}"`).join(', ');
        throw new RangeError(`Unknown voice ${describe(voice)} - the voices are ${keys}`);
    }
    if (style !== 'auto' && !styleTable.some((item) => item.key === style)) {
        const keys = [...styleTable.map((item) => `"${item.key}"`), '"auto"'].join(', ');
        throw new RangeError(`Unknown style ${describe(style)} - the styles are ${keys}`);
    }
    if (typeof rate !== 'number' || !(rate >= minRate && rate <= maxRate)) {
        throw new RangeError(`The rate must be a number from ${minRate} to ${maxRate}, not ${describe(rate)}`);
    }
    if (!Number.isSafeInteger(seed)) {
        throw new RangeError(`The seed must be an integer, not ${describe(seed)}`);
    }
    if (!Number.isInteger(sampleRate) || sampleRate < minSampleRate || sampleRate > maxSampleRate) {
        throw new RangeError(`The sample rate must be an integer from ${minSampleRate} to ${maxSampleRate}, not ${describe(sampleRate)}`);
    }
    return {voice, style, rate, seed, sampleRate};
}


// A value's description for an error message
function describe(value) {
    if (typeof value === 'string') {
        return JSON.stringify(value);
    }
    if (typeof value === 'number' || typeof value === 'boolean' || value === null || value === undefined) {
        return String(value);
    }
    return Array.isArray(value) ? 'an array' : `a ${typeof value}`;
}


// The next task - a promise resolved by a message to the page (or worker) itself
function nextTask() {
    return new Promise((resolve) => {
        const channel = new MessageChannel();
        channel.port1.onmessage = () => {
            channel.port1.close();
            resolve();
        };
        channel.port2.postMessage(null);
    });
}


// Speak text - an async iterator of the blocks (Float32Array)
async function* speakBlocks(text, {voice, style, rate, seed, sampleRate}) {
    const model = await loadModel();
    yield* textBlocks(model, text, {voice, style, rate, seed, sampleRate});
}


/**
 * Synthesize speech as a stream of sample blocks, made as they're taken - the text a sentence at a time, each sentence
 * a fraction of a second at a time, so the first audio comes quickly. Each sentence starts with a short pause (0.24
 * seconds, 0.4 before a paragraph - the text's first sentence starts one). The blocks, concatenated, are
 * {@link synthesize}'s samples.
 *
 * Synthesis runs on the calling thread while each block is taken - in a browser, take the blocks in a Web Worker to
 * keep the page responsive.
 *
 * @param {string} text - The text to speak (paragraphs are separated by blank lines)
 * @param {module:picovox~SynthesisOptions} [options = {}] - The synthesis options
 * @returns {AsyncIterable<Float32Array>} The blocks of samples (mono, -1 to 1, at the sample rate)
 * @throws {TypeError|RangeError} The text or an option is invalid
 */
export function stream(text, options = {}) {
    return speakBlocks(text, synthesisOptions(text, options));
}


/**
 * A synthesized sound
 *
 * @typedef {Object} Sound
 * @property {Float32Array} samples - The samples (mono, -1 to 1)
 * @property {number} sampleRate - The sample rate (Hz)
 */


/**
 * Synthesize speech
 *
 * @param {string} text - The text to speak (paragraphs are separated by blank lines)
 * @param {module:picovox~SynthesisOptions} [options = {}] - The synthesis options
 * @returns {Promise<module:picovox~Sound>} The speech
 * @throws {TypeError|RangeError} The text or an option is invalid
 */
export async function synthesize(text, options = {}) {
    const settings = synthesisOptions(text, options);
    const blocks = [];
    for await (const block of speakBlocks(text, settings)) {
        blocks.push(block);
    }
    const samples = new Float32Array(blocks.reduce((sum, block) => sum + block.length, 0));
    let offset = 0;
    for (const block of blocks) {
        samples.set(block, offset);
        offset += block.length;
    }
    return {samples, 'sampleRate': settings.sampleRate};
}


/**
 * Encode samples as a WAV file - 16-bit PCM, mono
 *
 * @param {Float32Array|number[]} samples - The samples (-1 to 1 - values beyond are clipped)
 * @param {number} sampleRate - The sample rate (Hz), a positive integer
 * @returns {Uint8Array} The WAV file's bytes
 * @throws {TypeError|RangeError} The samples or the sample rate are invalid
 */
export function toWav(samples, sampleRate) {
    if (!(samples instanceof Float32Array || samples instanceof Float64Array || Array.isArray(samples))) {
        throw new TypeError(`The samples must be a Float32Array (or an array of numbers), not ${describe(samples)}`);
    }
    if (!Number.isInteger(sampleRate) || sampleRate < 1 || sampleRate > 0x7fffffff) {
        throw new RangeError(`The sample rate must be a positive integer, not ${describe(sampleRate)}`);
    }
    return encodeWav(samples, sampleRate);
}


/**
 * The playback options of {@link speak} - the synthesis options, and the audio context
 *
 * @typedef {Object} SpeakOptions
 * @property {string} [voice = "male"] - The voice's key (see {@link voices})
 * @property {string} [style = "neutral"] - The speaking style's key (see {@link styles}), or "auto" - the quotations' words
 *     in the "dialogue" style (an exclaimed sentence's in the "exclaim" style), the narration neutral, and each paragraph
 *     read as one arc (its first sentence lifted, its pitch settling, its last sentence slower)
 * @property {number} [rate = 1] - The speaking speed, from 0.5 to 2
 * @property {number} [seed = 1000] - The noise seed, an integer
 * @property {number} [sampleRate = 24000] - The synthesis sample rate (Hz)
 * @property {AudioContext} [context] - The Web Audio context to play in (default: a new one, closed when playback ends)
 */


/**
 * A playback of {@link speak}
 *
 * @typedef {Object} Playback
 * @property {function(): void} stop - Stop the playback
 * @property {Promise<void>} finished - Resolves when the playback ends or is stopped; rejects if synthesis fails
 */


/**
 * Speak text aloud in a browser, with Web Audio - playback starts with the first sentence's first blocks, while the
 * rest are synthesized (a few seconds ahead of playback). Synthesis runs on the main thread, a block at a time between
 * the page's other work - for long texts, use {@link stream} in a Web Worker and play its blocks.
 *
 * Call it from a user gesture (a click), so the browser allows the audio to start.
 *
 * @param {string} text - The text to speak (paragraphs are separated by blank lines)
 * @param {module:picovox~SpeakOptions} [options = {}] - The playback options
 * @returns {module:picovox~Playback} The playback
 * @throws {Error} There is no Web Audio (as under Node - use {@link synthesize} and {@link toWav} there)
 * @throws {TypeError|RangeError} The text or an option is invalid
 */
export function speak(text, options = {}) {
    const settings = synthesisOptions(text, options, ['context']);
    if (options.context !== undefined && (options.context === null || typeof options.context !== 'object')) {
        throw new TypeError(`The context must be an AudioContext, not ${describe(options.context)}`);
    }
    const AudioContextClass = globalThis.AudioContext ?? globalThis.webkitAudioContext;
    if (options.context === undefined && typeof AudioContextClass !== 'function') {
        throw new Error('speak requires Web Audio (AudioContext), as in a browser - under Node, use synthesize and toWav');
    }
    const ownContext = options.context === undefined;
    const context = ownContext ? new AudioContextClass() : options.context;
    if (context.state === 'suspended') {
        context.resume();
    }

    // The playback's end - stop the scheduled blocks, close the context (if it's the playback's own), and settle
    const sources = new Set();
    const waiting = [];
    const playback = {'done': false};
    let settle = null;
    const finished = new Promise((resolve, reject) => {
        settle = (error) => (error === undefined ? resolve() : reject(error));
    });
    const finish = (error) => {
        if (!playback.done) {
            playback.done = true;
            for (const source of sources) {
                source.onended = null;
                source.stop();
            }
            sources.clear();
            for (const resolve of waiting.splice(0)) {
                resolve();
            }
            if (ownContext) {
                context.close();
            }
            settle(error);
        }
    };

    // Schedule the blocks as they're synthesized, one after another - yielding to the page after each (a message, not a
    // timer: timers are throttled in background tabs), and staying at most a few seconds ahead of playback (waiting for
    // scheduled blocks to end)
    const play = async () => {
        let time = 0;
        let last = null;
        for await (const block of speakBlocks(text, settings)) {
            if (playback.done) {
                return;
            }
            const buffer = context.createBuffer(1, block.length, settings.sampleRate);
            buffer.copyToChannel(block, 0);
            const source = context.createBufferSource();
            source.buffer = buffer;
            source.connect(context.destination);
            source.onended = () => {
                sources.delete(source);
                for (const resolve of waiting.splice(0)) {
                    resolve();
                }
            };
            time = Math.max(time, context.currentTime + playbackLead);
            sources.add(source);
            source.start(time);
            time += block.length / settings.sampleRate;
            last = source;
            await nextTask();
            while (!playback.done && time - context.currentTime > playbackAhead) {
                await new Promise((resolve) => {
                    waiting.push(resolve);
                });
            }
        }
        if (last === null || !sources.has(last)) {
            finish();
        } else {
            last.onended = () => {
                sources.delete(last);
                finish();
            };
        }
    };
    play().catch(finish);
    return {'stop': () => finish(), finished};
}
