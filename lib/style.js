// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// The style network - a quotation's speaking style from its words and the narration around it (lib/models/styleModel.js,
// trained by picovox_train/style.py). Only the high-energy styles: angry, shouting, and fearful - listeners preferred them
// on the quoted words, and the quieter styles made a book worse - so a quotation is neutral unless the network is
// confident (its style's probability at least the style's threshold). The features (picovox_train/style.py's port):
//
// - words - the quotation's, its attribution's (the narration beside it in its sentences), and the neighboring
//   sentences' lowercase words, each hashed (FNV-1a) into the one embedding table
// - cues - its exclamation and question marks, capitals, emphasis (underscores), and length, and the attribution's
//   exclamation marks
//
// The network: each kind of text's mean and max embedding and the cues, a tanh layer, and the styles' logits.

import {decodeBase64} from './package.js';
import {quoteSpans} from './sentences.js';


// The styles the network chooses among - neutral first
export const styleClasses = ['neutral', 'angry', 'shouting', 'fearful'];

// The number of cue features
export const cueCount = 8;


// A text's words for the features - lowercase, without accents, the curly apostrophe straight
export function styleWords(text) {
    const plain = text.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase().replace(/[‘’]/g, '\'');
    return plain.match(/[a-z]+(?:'[a-z]+)*/g) ?? [];
}


// The FNV-1a hash of a string (32 bits, unsigned) - of its UTF-16 code units, as the Python port's
export function fnv1a(text) {
    let hash = 0x811c9dc5;
    for (let ix = 0; ix < text.length; ix++) {
        hash ^= text.charCodeAt(ix);
        hash = Math.imul(hash, 0x01000193) >>> 0;
    }
    return hash;
}


// A text's word buckets - its words hashed into the table
export function wordBuckets(text, buckets) {
    return styleWords(text).map((word) => fnv1a(word) % buckets);
}


// A quotation's cue features
export function styleCues({text, before = '', after = ''}) {
    const letters = text.match(/\p{L}/gu) ?? [];
    const capitals = letters.filter((letter) => letter !== letter.toLowerCase()).length;
    const words = styleWords(text).length;
    const attribution = `${before} ${after}`;
    return [
        Math.min(3, (text.match(/!/g) ?? []).length) / 3,
        (text.match(/\?/g) ?? []).length > 0 ? 1 : 0,
        letters.length > 0 ? capitals / letters.length : 0,
        /_/.test(text) ? 1 : 0,
        Math.log1p(words) / 4,
        words <= 3 ? 1 : 0,
        /!/.test(attribution) ? 1 : 0,
        /[—-]{2,}|—/.test(text) ? 1 : 0
    ];
}


// Load the style network from its generated module - {config, weights (Float32Arrays: embedding, hidden, hiddenBias,
// output, outputBias - int8 in the module, a scale per tensor), thresholds}
export function loadStyleModel(module) {
    const {config, scales, thresholds} = module.styleModel;
    const bytes = decodeBase64(module.styleModel.weights);
    const quantized = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const shapes = {
        'embedding': [config.buckets, config.dim],
        'hidden': [config.hidden, 6 * config.dim + cueCount],
        'hiddenBias': [1, config.hidden],
        'output': [styleClasses.length, config.hidden],
        'outputBias': [1, styleClasses.length]
    };
    const weights = {};
    let offset = 0;
    for (const [name, [rows, columns]] of Object.entries(shapes)) {
        const values = new Float32Array(rows * columns);
        for (let ix = 0; ix < rows * columns; ix++) {
            values[ix] = quantized[offset + ix] * scales[name];
        }
        weights[name] = values;
        offset += rows * columns;
    }
    return {config, weights, thresholds};
}


// The style network's loader - a function returning a promise of loadStyleModel's model, its module imported on first
// use, once (a failed import is tried again on the next call)
export function styleLoader(importModule) {
    let loading = null;
    return () => {
        if (loading === null) {
            loading = importModule().then((module) => loadStyleModel(module));
            loading.catch(() => {
                loading = null;
            });
        }
        return loading;
    };
}


// A quotation's style probabilities - {quotation text, before, after, previous, next} - by styleClasses
export function styleProbabilities(model, quotation) {
    const {config, weights} = model;
    const {dim, hidden, buckets} = config;
    // Each kind of text's mean embedding, then each one's max (an empty text's zero), then the cues
    const input = new Float32Array(6 * dim + cueCount);
    const attribution = `${quotation.before ?? ''} ${quotation.after ?? ''}`;
    const texts = [quotation.text, attribution, `${quotation.previous ?? ''} ${quotation.next ?? ''}`];
    texts.forEach((text, kindIndex) => {
        const indexes = wordBuckets(text, buckets);
        const peak = new Float32Array(dim).fill(-Infinity);
        for (const index of indexes) {
            for (let ix = 0; ix < dim; ix++) {
                const value = weights.embedding[index * dim + ix];
                input[kindIndex * dim + ix] += value / indexes.length;
                peak[ix] = Math.max(peak[ix], value);
            }
        }
        if (indexes.length > 0) {
            input.set(peak, (3 + kindIndex) * dim);
        }
    });
    input.set(styleCues(quotation), 6 * dim);
    const layer = new Float32Array(hidden);
    for (let row = 0; row < hidden; row++) {
        let sum = weights.hiddenBias[row];
        for (let column = 0; column < input.length; column++) {
            sum += weights.hidden[row * input.length + column] * input[column];
        }
        layer[row] = Math.tanh(sum);
    }
    const logits = styleClasses.map((name, row) => {
        let sum = weights.outputBias[row];
        for (let column = 0; column < hidden; column++) {
            sum += weights.output[row * hidden + column] * layer[column];
        }
        return sum;
    });
    const top = Math.max(...logits);
    const exps = logits.map((logit) => Math.exp(logit - top));
    const total = exps.reduce((sum, value) => sum + value, 0);
    return exps.map((value) => value / total);
}


// A quotation's style - the most probable style that reaches its threshold, else neutral
export function quotationStyle(model, quotation) {
    const probabilities = styleProbabilities(model, quotation);
    let best = 'neutral';
    let bestProbability = 0;
    styleClasses.forEach((name, index) => {
        if (index > 0 && probabilities[index] >= model.thresholds[name] && probabilities[index] > bestProbability) {
            best = name;
            bestProbability = probabilities[index];
        }
    });
    return best;
}


// Sentences' quotations - [{quotation, first, last, text, before, after, previous, next}] (quoteSpans's quotations, the
// narration beside each in its first and last sentences, and the sentences around it)
export function textQuotations(sentences) {
    const spans = quoteSpans(sentences);
    const quotations = new Map();
    spans.forEach((sentenceSpans, index) => {
        sentenceSpans.forEach((span, spanIndex) => {
            if (span.quotation === null) {
                return;
            }
            if (!quotations.has(span.quotation)) {
                const before = spanIndex > 0 && !sentenceSpans[spanIndex - 1].quoted ? sentenceSpans[spanIndex - 1].text : '';
                quotations.set(span.quotation, {'quotation': span.quotation, 'first': index, 'last': index, 'texts': [], before,
                    'after': ''});
            }
            const quotation = quotations.get(span.quotation);
            quotation.last = index;
            quotation.texts.push(span.text);
            const following = sentenceSpans[spanIndex + 1];
            quotation.after = following !== undefined && !following.quoted ? following.text : '';
        });
    });
    return [...quotations.values()].map(({texts, ...quotation}) => ({
        ...quotation, 'text': texts.join(' '), 'previous': sentences[quotation.first - 1]?.text ?? '',
        'next': sentences[quotation.last + 1]?.text ?? ''
    }));
}


// A text's automatic styles - {spans(index)}: a sentence's spans ([{text, style}] - its quotations' words in their styles,
// a quotation the network leaves neutral in the dialogue style, its narration neutral, adjacent spans of one style
// merged), or null when the sentence is all narration. The quotations
// are found at once; each is styled when a sentence of it is first asked for.
export function autoStyles(styleModel, sentences) {
    const sentenceSpans = quoteSpans(sentences);
    const quotations = textQuotations(sentences);
    const byIndex = new Map(quotations.map((quotation) => [quotation.quotation, quotation]));
    const styles = new Map();
    const quotationStyleOf = (index) => {
        if (!styles.has(index)) {
            const style = quotationStyle(styleModel, byIndex.get(index));
            styles.set(index, style === 'neutral' ? 'dialogue' : style);
        }
        return styles.get(index);
    };
    return {
        spans(index) {
            const spans = [];
            for (const span of sentenceSpans[index]) {
                const style = span.quotation === null ? 'neutral' : quotationStyleOf(span.quotation);
                const last = spans[spans.length - 1];
                if (last !== undefined && last.style === style) {
                    last.text = `${last.text} ${span.text}`;
                } else {
                    spans.push({'text': span.text, style});
                }
            }
            return spans.some((span) => span.style !== 'neutral') ? spans : null;
        }
    };
}


// The paragraph beat (style "auto") - each paragraph spoken as one arc, as a narrator reads: its first sentence lifted
// (pitch up by lift, level and frication up by level dB), the pitch settling to -lift/2 by its last sentence, which is
// slowed by settle. Tuned by ear against the flat reading (Moby-Dick's first chapter, by James).
export const paragraphBeat = {'lift': 0.08, 'level': 2, 'settle': 0.07};


// Sentences' paragraph beats - [{transforms, rate}] by sentence index (lib/synthesis.js sentenceFrames's beat): each sentence's place in
// its paragraph (lib/sentences.js paragraphStart) as paragraphBeat's arc
export function paragraphBeats(sentences) {
    const paragraphs = [];
    for (const sentence of sentences) {
        if (sentence.paragraphStart || paragraphs.length === 0) {
            paragraphs.push([]);
        }
        paragraphs[paragraphs.length - 1].push(sentence.index);
    }
    const {lift, level, settle} = paragraphBeat;
    const beats = [];
    for (const indexes of paragraphs) {
        for (const [position, index] of indexes.entries()) {
            const along = indexes.length > 1 ? position / (indexes.length - 1) : 0;
            const transforms = [{'pitchScale': 1 + lift - 1.5 * lift * along}];
            if (position === 0) {
                transforms.push({'level': level}, {'frication': level});
            }
            beats[index] = {transforms, 'rate': indexes.length > 1 && position === indexes.length - 1 ? 1 - settle : 1};
        }
    }
    return beats;
}
