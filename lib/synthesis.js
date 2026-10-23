// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// The synthesis pipeline - text to audio, as PicoShelf (the audiobook player) plays it: the text is split into
// sentences (lib/sentences.js), and each, after a short pause, is spoken by the multi-speaker neural model
// (lib/model.js - every word pronounced by the G2P network, lib/g2p.js) a fraction of a second at a time, rendered by
// the source-filter synthesizer (lib/render.js) at 24 kHz, with the output gain and a soft limiter.

import {DB_NAMES, FRAME_RATE, NAMES, PARAMETERS} from './params.js';
import {autoStyles, paragraphBeats} from './style.js';
import {speechText, splitSentences} from './sentences.js';
import {applyTransforms} from './transforms.js';
import {createRenderer} from './render.js';
import {loadG2p} from './g2p.js';
import {loadPackagedModel} from './package.js';
import {setPronouncer} from './lexicon.js';
import {streamFrames} from './model.js';


// The voices - each a speaker (by key) of the multi-speaker model's speaker table - and the default
export const voiceTable = [
    // (the package voices' pulse a little softer - listeners found them more natural, blind, at Rd x1.2 - 2026-10-23)
    {'key': 'male', 'label': 'Male', 'sex': 'male', 'speaker': 'arctic-bdl', 'adjust': {'rdScale': 1.2},
        'description': 'A clear, steady American male voice - one reader of the model\'s speakers.'},
    {'key': 'female', 'label': 'Female', 'sex': 'female', 'speaker': 'arctic-slt', 'adjust': {'rdScale': 1.2},
        'description': 'A clear, steady American female voice - one reader of the model\'s speakers.'},
    // The designed voices (data/nn/voices.json): blends of the model's audiobook readers (by speaker key, the shares
    // summing to 1), with mild adjustments of the parameter frames (lib/transforms.js) and a speaking rate (1 is the
    // model's own)
    {'key': 'sarah', 'label': 'Sarah', 'sex': 'female', 'rate': 0.97,
        'blend': {'libritts-1462': 0.53, 'libritts-5789': 0.47}, 'adjust': {'tractScale': 0.97},
        'description': 'A warm, unhurried storyteller with a mid-low, rounded tone - an easy companion for a long novel.'},
    {'key': 'helen', 'label': 'Helen', 'sex': 'female', 'rate': 0.9,
        'blend': {'libritts-8324': 0.41, 'libritts-4481': 0.3, 'libritts-2182': 0.29}, 'adjust': {'pitchScale': 0.95, 'tractScale': 0.98},
        'description': 'A mature, composed voice, lower and slower, with quiet authority for weighty books.'},
    {'key': 'amy', 'label': 'Amy', 'sex': 'female', 'rate': 1.03,
        'blend': {'libritts-2836': 0.5, 'libritts-226': 0.5}, 'adjust': {'pitchScale': 1.04},
        'description': 'A young, bright, lively voice, higher and quicker, for younger readers and light-hearted stories.'},
    {'key': 'kate', 'label': 'Kate', 'sex': 'female', 'rate': 1.06,
        'blend': {'libritts-5895': 0.43, 'libritts-6078': 0.4, 'libritts-7635': 0.17}, 'adjust': {'tractScale': 0.97},
        'description': 'A brisk, clear, articulate voice with a crisp tone that keeps facts and arguments moving.'},
    {'key': 'grace', 'label': 'Grace', 'sex': 'female', 'rate': 0.9,
        'blend': {'libritts-84': 0.31, 'libritts-6272': 0.24, 'libritts-2910': 0.23, 'libritts-1088': 0.22}, 'adjust': {'effort': -2},
        'description': 'A gentle, soft, slow voice with a dark, quiet tone for winding down.'},
    {'key': 'james', 'label': 'James', 'sex': 'male', 'rate': 0.96,
        'blend': {'libritts-2518': 0.35, 'libritts-1624': 0.25, 'libritts-2911': 0.21, 'libritts-6529': 0.19}, 'adjust': {},
        'description': 'A warm, relaxed storyteller with a mid-low, rounded tone for long-form fiction.'},
    {'key': 'robert', 'label': 'Robert', 'sex': 'male', 'rate': 0.93,
        'blend': {'libritts-4397': 0.44, 'libritts-2803': 0.42, 'libritts-2514': 0.14}, 'adjust': {'pitchScale': 0.95, 'tractScale': 0.96},
        'description': 'A mature, deep, deliberate voice with a grave authority for serious and older books.'},
    {'key': 'michael', 'label': 'Michael', 'sex': 'male', 'rate': 0.98,
        'blend': {'libritts-2893': 0.55, 'libritts-2428': 0.45}, 'adjust': {'pitchScale': 1.04, 'tractScale': 1.04},
        'description': 'A young, bright, energetic voice, higher and quicker, for younger readers and adventure.'},
    {'key': 'david', 'label': 'David', 'sex': 'male', 'rate': 0.99,
        'blend': {'arctic-bdl': 0.37, 'libritts-4441': 0.34, 'libritts-6241': 0.29}, 'adjust': {},
        'description': 'A brisk, clear, matter-of-fact voice with a crisp tone for explanation and reportage.'},
    {'key': 'thomas', 'label': 'Thomas', 'sex': 'male', 'rate': 0.91,
        'blend': {'libritts-4018': 0.45, 'libritts-4898': 0.33, 'libritts-3436': 0.22}, 'adjust': {'effort': -2, 'tractScale': 1.03},
        'description': 'A gentle, soft, slow voice with a dark, calm tone for poetry and quiet evenings.'}
];
export const defaultVoice = 'male';


// The styles (data/nn/voices.json's, tuned by ear) - for some, the model's emotion embedding at an intensity (the voice
// model's style table: emotions learned from CREMA-D's actors); frame transforms (lib/transforms.js - press, pitch,
// whisper, level, and frication: a louder style is level, not effort, so its consonants keep pace with its vowels, and
// its frication moves with its level, its fricatives sharper or softer); and a speaking rate (multiplying the voice's).
// Neutral is the voice as it is.
export const styleTable = [
    {'key': 'neutral', 'label': 'Neutral', 'kind': 'emotion',
        'description': 'The voice as designed - plain narration.'},
    {'key': 'angry', 'label': 'Angry', 'kind': 'emotion',
        'transforms': [{'press': 6}, {'pitchScale': 1.05}, {'level': 3}, {'frication': 3}],
        'description': 'A pressed, raised, louder voice - angry without the angry embedding, which sounded muffled.'},
    {'key': 'happy', 'label': 'Happy', 'kind': 'emotion',
        'emotion': {'emotion': 'happy', 'intensity': 0.7},
        'transforms': [{'press': 2}, {'pitchScale': 1.04}, {'level': 2}, {'frication': 2}],
        'description': 'The happy style embedding, a little brighter, higher, and louder.'},
    {'key': 'sad', 'label': 'Sad', 'kind': 'emotion',
        'emotion': {'emotion': 'sad', 'intensity': 0.6}, 'transforms': [{'level': 2.5}, {'frication': 2.5}],
        'description': 'The sad style embedding (lower, slower, and darker).'},
    {'key': 'fearful', 'label': 'Fearful', 'kind': 'emotion',
        'emotion': {'emotion': 'fearful', 'intensity': 0.6},
        'transforms': [{'whisper': 0.15}, {'pitchScale': 1.1}, {'level': 2}, {'frication': 2}], 'rate': 1.08,
        'description': 'The fearful style embedding, a little breathy, higher, and quicker.'},
    {'key': 'disgusted', 'label': 'Disgusted / contemptuous', 'kind': 'emotion',
        'emotion': {'emotion': 'disgusted', 'intensity': 0.5},
        'transforms': [{'press': 2}, {'pitchScale': 0.95}, {'level': 1}, {'frication': 1}], 'rate': 0.9,
        'description': 'The disgusted style embedding, a little pressed, lower, and slower.'},
    {'key': 'whisper', 'label': 'Whisper', 'kind': 'mode',
        'transforms': [{'whisper': 1}, {'level': 5}, {'frication': 5}],
        'description': 'No voicing - the breath (aspiration) only.'},
    {'key': 'soft', 'label': 'Soft', 'kind': 'mode',
        'transforms': [{'level': -5}, {'frication': -5}], 'rate': 0.95,
        'description': 'A quieter voice, a little slower.'},
    {'key': 'shouting', 'label': 'Shouting', 'kind': 'mode',
        'transforms': [{'press': 10}, {'pitchScale': 1.15}, {'level': 1}, {'frication': 1}],
        'description': 'A pressed, bright, raised voice, a little louder - its consonants with it.'},
    {'key': 'dialogue', 'label': 'Dialogue', 'kind': 'mode',
        'transforms': [{'press': 1}, {'pitchScale': 1.07}, {'level': 1}, {'frication': 1}], 'rate': 1.05,
        'description': 'Neutral dialogue - a little higher, brighter, and quicker than the narration (style auto\'s quotations).'},
    {'key': 'exclaim', 'label': 'Exclaim', 'kind': 'mode',
        'transforms': [{'pitchRange': 1.2}, {'press': 3}, {'pitchScale': 1.09}, {'level': 3}, {'frication': 3}], 'rate': 1.06,
        'description': 'An exclamation - the dialogue style livelier: louder, wider in pitch, and quicker (style auto\'s ' +
            'exclaimed quotations).'}
];
export const defaultStyle = 'neutral';


// The pauses before a sentence and a paragraph (s, at normal speed - the model's own edge silences are short)
export const sentencePause = 0.24;
export const paragraphPause = 0.4;


// No paragraph beat (sentenceFrames's beat)
const noBeat = {'transforms': [], 'rate': 1};


// The output sample rate (Hz) - the voices' training rate (their content stops below 10.5 kHz), at about half 44.1
// kHz's rendering cost
export const defaultSampleRate = 24000;


// The output gain (the multi-speaker model's speech runs about 2 dB below the earlier single-speaker voices', whose
// gain was 1.05), and the rendered block size (samples)
export const outputGain = 1.3;
export const blockSize = 2048;


// A soft limiter - linear below 0.8, approaching 1 above
export function softLimit(value) {
    const magnitude = Math.abs(value);
    if (magnitude <= 0.8) {
        return value;
    }
    return Math.sign(value) * (0.8 + 0.2 * Math.tanh((magnitude - 0.8) / 0.2));
}


// A model loader - a function returning a Promise of the voice model, loaded once: the G2P network (importG2p's
// module, lib/models/g2pModel.js) set as the text front end's pronouncer, and the multi-speaker model (importVoice's
// module, lib/models/voiceModel.js) unpacked. A failed load is retried on the next call.
export function modelLoader(importG2p, importVoice) {
    let loading = null;
    return () => {
        if (loading === null) {
            const pronouncer = importG2p().then((module) => loadG2p(module.g2pModel));
            const model = importVoice().then((module) => loadPackagedModel(module));
            loading = Promise.all([pronouncer, model]).then(([pronounce, voiceModel]) => {
                setPronouncer(pronounce);
                return voiceModel;
            });
            loading.catch(() => {
                loading = null;
            });
        }
        return loading;
    };
}


// A voice's speaker in the model's speaker table - its id, or a designed voice's blend ({id: share})
export function voiceSpeaker(model, voice) {
    const entry = voiceTable.find((item) => item.key === voice);
    const ids = new Map(model.manifest.speakers.map((speaker) => [speaker.key, speaker.id]));
    const speakers = entry.blend ?? {[entry.speaker]: 1};
    const blend = {};
    for (const [key, share] of Object.entries(speakers)) {
        if (!ids.has(key)) {
            throw new Error(`The ${voice} voice's speaker (${key}) isn't in the model`);
        }
        blend[ids.get(key)] = share;
    }
    return entry.blend ? blend : ids.get(entry.speaker);
}


// The source shelf correction (dB) - every voice's source this much darker than the network predicts it: listeners heard
// the predicted source as tinny (blind A/B in the voice lab, 2026-10-23 - less tinny in 6 of the 12 voices, the same in
// the rest, never less natural; UTMOS, WER, and speaker similarity on the panel texts unchanged)
export const sourceShelf = -5;


// A voice's settings - {speaker (voiceSpeaker's), rate (the voice's speaking speed), transforms (its adjustments as
// lib/transforms.js applyTransforms's - identity values omitted - then the source shelf correction)}
export function voiceSettings(model, voice) {
    const entry = voiceTable.find((item) => item.key === voice);
    if (entry === undefined) {
        throw new Error(`Unknown voice ${JSON.stringify(voice)}`);
    }
    const {pitchScale = 1, tractScale = 1, effort = 0, rdScale = 1} = entry.adjust;
    const transforms = [];
    if (pitchScale !== 1) {
        transforms.push({pitchScale});
    }
    if (tractScale !== 1) {
        transforms.push({tractScale});
    }
    if (effort !== 0) {
        transforms.push({effort});
    }
    if (rdScale !== 1) {
        transforms.push({rdScale});
    }
    transforms.push({'shelf': sourceShelf});
    return {'speaker': voiceSpeaker(model, voice), 'rate': entry.rate ?? 1, transforms};
}


// A style's settings - {emotion (the model's style - {emotion, intensity} - or undefined), rate, transforms}
export function styleSettings(style = defaultStyle) {
    const entry = styleTable.find((item) => item.key === style);
    if (entry === undefined) {
        throw new Error(`Unknown style ${JSON.stringify(style)}`);
    }
    return {'emotion': entry.emotion, 'rate': entry.rate ?? 1, 'transforms': entry.transforms ?? []};
}


// A sentence's frames from the model - {chunks (an iterator of the frames, a fraction of a second at a time - the network
// runs as they're taken), eq (the voice's channel equalizer)}. options: voice (a voiceTable key - its speaker, speaking
// rate, and adjustments) or speaker (the model's speaker id, or a blend {id: share}), style (a styleTable key - its
// emotion, adjustments after the voice's, and rate), rate (the speaking speed - 1 is normal), and beat (lib/style.js
// paragraphBeats's - its transforms after the style's, and its rate; default none). With createSentenceRenderer, the
// network and the renderer can run apart (in two workers, say); sentenceBlocks is the two together.
export function sentenceFrames(model, text, {voice, speaker, style, rate = 1, beat = noBeat} = {}) {
    const voiced = voice === undefined ? {speaker, 'rate': 1, 'transforms': []} : voiceSettings(model, voice);
    const styled = styleSettings(style);
    const transforms = [...voiced.transforms, ...styled.transforms, ...beat.transforms];
    const options = {'rate': 1 / (rate * voiced.rate * styled.rate * beat.rate), 'speaker': voiced.speaker, 'style': styled.emotion};
    const stream = streamFrames(model, text, options);
    if (transforms.length === 0) {
        return {'chunks': stream.chunks, 'eq': stream.eq};
    }
    // A pitch range is the whole text's (around its mean pitch) - its frames gathered, then transformed at once
    if (transforms.some((transform) => 'pitchRange' in transform)) {
        const frames = [...stream.chunks].flat();
        return {'chunks': (frames.length === 0 ? [] : [applyTransforms(frames, transforms)])[Symbol.iterator](), 'eq': stream.eq};
    }
    function* chunks() {
        for (const chunk of stream.chunks) {
            yield applyTransforms(chunk, transforms);
        }
    }
    return {'chunks': chunks(), 'eq': stream.eq};
}


// The silence between a sentence's spans (s, at normal speed) - a quotation's words in a style, its narration neutral
export const spanPause = 0.06;


// A silent frame like a frame - its sources at their floors (the tract unchanged, so the spans join smoothly)
const silentIndexes = DB_NAMES.map((name) => [NAMES.indexOf(name), PARAMETERS.find((parameter) => parameter.name === name).low]);
function silentFrame(frame) {
    const out = Float64Array.from(frame);
    for (const [index, low] of silentIndexes) {
        out[index] = low;
    }
    return out;
}


// A sentence's frames in spans - each span ({text, style}) spoken in its style (sentenceFrames's), spanPause of silence
// between them - as one stream: {chunks, eq (the voice's)}. options: voice or speaker, rate, and beat, as
// sentenceFrames's. Each span's network runs as its frames are taken.
export function spanFrames(model, spans, {voice, speaker, rate = 1, beat = noBeat} = {}) {
    const first = sentenceFrames(model, spans[0].text, {voice, speaker, 'style': spans[0].style, rate, beat});
    const silentCount = Math.round(spanPause * FRAME_RATE / rate);
    function* chunks() {
        let last = null;
        for (const [index, span] of spans.entries()) {
            const stream = index === 0 ? first : sentenceFrames(model, span.text, {voice, speaker, 'style': span.style, rate, beat});
            if (last !== null) {
                const silent = silentFrame(last);
                yield Array.from({'length': silentCount}, () => Float64Array.from(silent));
            }
            for (const chunk of stream.chunks) {
                last = chunk[chunk.length - 1];
                yield chunk;
            }
        }
    }
    return {'chunks': chunks(), 'eq': first.eq};
}


// A sentence's renderer - onBlock(block) gets the sentence's audio (Float32Array blocks, with the output gain and the
// limiter): its pause (s, at normal speed) at once, then each block as it's final. push(frames) the sentence's frames (a
// chunk of sentenceFrames's) as they come, then end(). options: eq (sentenceFrames's), rate, seed (the noise seed),
// pause, and sampleRate.
export function createSentenceRenderer({onBlock, eq = null, rate = 1, seed = 1, pause = 0, sampleRate = defaultSampleRate}) {
    const pauseLength = Math.round(sampleRate * pause / rate);
    if (pauseLength > 0) {
        onBlock(new Float32Array(pauseLength));
    }
    return createRenderer({sampleRate, seed, eq, blockSize, 'onBlock': (block) => {
        const samples = new Float32Array(block.length);
        for (let ix = 0; ix < block.length; ix++) {
            samples[ix] = softLimit(block[ix] * outputGain);
        }
        onBlock(samples);
    }});
}


// Speak a sentence with the model - a pause (s, at normal speed), then the sentence's audio - as an iterator of blocks
// (Float32Array) made as they're taken. options: voice or speaker, and style (sentenceFrames's) - or spans (spanFrames's,
// in place of the text) - rate (the speaking speed - 1 is normal), beat (sentenceFrames's), seed (the renderer's noise
// seed), pause, and sampleRate.
export function* sentenceBlocks(model, text, options = {}) {
    const {voice, speaker, style, spans, rate = 1, beat, seed = 1, pause = 0, sampleRate = defaultSampleRate} = options;

    // The network streams its frames a fraction of a second at a time, each chunk rendered as it comes
    const frames = spans ? spanFrames(model, spans, {voice, speaker, rate, beat}) :
        sentenceFrames(model, text, {voice, speaker, style, rate, beat});
    const {chunks, eq} = frames;
    const ready = [];
    const renderer = createSentenceRenderer({'onBlock': (block) => ready.push(block), eq, rate, seed, pause, sampleRate});
    yield* ready.splice(0);
    for (const chunk of chunks) {
        renderer.push(chunk);
        yield* ready.splice(0);
    }
    renderer.end();
    yield* ready.splice(0);
}


// Speak text with the model - each sentence (lib/sentences.js) after its pause (a longer one before a paragraph), the
// sentence at index i with the noise seed seed + i - as an iterator of blocks (Float32Array). options: voice or
// speaker, style, rate, seed, and sampleRate, as sentenceBlocks's; style "auto" styles the quotations (lib/style.js
// autoStyles - in the dialogue style, an exclaimed sentence's in the exclaim style), their narration neutral, and each
// paragraph shaped as one arc (paragraphBeats).
export function* textBlocks(model, text, {voice, speaker, style, rate = 1, seed = 1, sampleRate = defaultSampleRate} = {}) {
    const sentences = splitSentences(text);
    const auto = style === 'auto' ? autoStyles(sentences) : null;
    const beats = auto === null ? null : paragraphBeats(sentences);
    for (const sentence of sentences) {
        const pause = sentence.paragraphStart ? paragraphPause : sentencePause;
        const spans = auto?.spans(sentence.index)?.map((span) => ({'text': speechText(span.text), 'style': span.style}));
        const beat = beats?.[sentence.index];
        const options = {
            voice, speaker, 'style': auto === null ? style : undefined, spans, rate, beat, 'seed': seed + sentence.index, pause
        };
        yield* sentenceBlocks(model, speechText(sentence.text), {...options, sampleRate});
    }
}
