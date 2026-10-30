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


// The voices - each a speaker of the multi-speaker model's speaker table (the voice's key is its speaker's key: one
// voice of the local TTS model the voice model was trained from - train/data/train.json's tts voices) - and the
// default
export const voiceTable = [
    {'key': 'male', 'label': 'Male', 'sex': 'male',
        'description': 'A warm, buoyant male voice over a resonant mid-low core - the default.'},
    {'key': 'female', 'label': 'Female', 'sex': 'female',
        'description': 'A modern, confident, and friendly female voice.'},
    {'key': 'sarah', 'label': 'Sarah', 'sex': 'female',
        'description': 'A cheerful female voice with real resonance under it, personable and decisive.'},
    {'key': 'helen', 'label': 'Helen', 'sex': 'female',
        'description': 'A low, grounded female voice, quick to the point, with a sincere tone.'},
    {'key': 'amy', 'label': 'Amy', 'sex': 'female',
        'description': 'A young, playful, and upbeat female voice.'},
    {'key': 'kate', 'label': 'Kate', 'sex': 'female',
        'description': 'A settled, even-toned, and kind female voice with crisp consonants.'},
    {'key': 'grace', 'label': 'Grace', 'sex': 'female',
        'description': 'A rounded, soft-edged female voice with an easy, relaxed delivery.'},
    {'key': 'james', 'label': 'James', 'sex': 'male',
        'description': 'A resonant, decisive male voice with a wide dynamic range.'},
    {'key': 'robert', 'label': 'Robert', 'sex': 'male',
        'description': 'A trim, focused male voice with a steady, low-set core.'},
    {'key': 'michael', 'label': 'Michael', 'sex': 'male',
        'description': 'A buoyant, energetic male voice with a playful lift at the ends of phrases.'},
    {'key': 'david', 'label': 'David', 'sex': 'male',
        'description': 'A punchy, close male voice with crisp consonants and a pitch that lifts across each line.'},
    {'key': 'thomas', 'label': 'Thomas', 'sex': 'male',
        'description': 'A high-energy, encouraging male voice.'}
];
export const defaultVoice = 'male';


// The styles (test/voices.json's) - frame transforms (lib/transforms.js - press, pitch, pitch range, whisper, level, and
// frication: a louder style is level, not effort, so its consonants keep pace with its vowels, and its frication moves
// with its level, its fricatives sharper or softer) and a speaking rate. The emotions' transforms (happy, sad, fearful,
// disgusted) are the measured effects of the emotion embeddings the models trained on CREMA-D's actors had (before
// 1.7 - their F0, range, Rd, voicing, frication, and duration on the panel texts in every voice, embedding and
// transforms together, less the breathiness the embeddings added). Neutral is the voice as it is.
export const styleTable = [
    {'key': 'neutral', 'label': 'Neutral', 'kind': 'emotion',
        'description': 'The voice as designed - plain narration.'},
    {'key': 'angry', 'label': 'Angry', 'kind': 'emotion',
        'transforms': [{'press': 6}, {'pitchScale': 1.05}, {'level': 3}, {'frication': 3}],
        'description': 'A pressed, raised, louder voice.'},
    {'key': 'happy', 'label': 'Happy', 'kind': 'emotion',
        'transforms': [{'press': 5}, {'pitchScale': 1.23}, {'level': -0.8}],
        'description': 'A brighter, much higher voice.'},
    {'key': 'sad', 'label': 'Sad', 'kind': 'emotion',
        'transforms': [{'pitchRange': 0.93}, {'press': 2}, {'pitchScale': 0.97}, {'level': -3.3}, {'frication': 2.9}], 'rate': 0.97,
        'description': 'A quieter, lower, flatter, and slower voice.'},
    {'key': 'fearful', 'label': 'Fearful', 'kind': 'emotion',
        'transforms': [{'whisper': 0.15}, {'pitchScale': 1.23}, {'level': -2.4}, {'frication': 2.9}], 'rate': 1.1,
        'description': 'A breathy, quieter, much higher, and quicker voice.'},
    {'key': 'disgusted', 'label': 'Disgusted / contemptuous', 'kind': 'emotion',
        'transforms': [{'press': 4.8}, {'pitchScale': 0.95}, {'level': -2.4}, {'frication': 1.9}], 'rate': 0.83,
        'description': 'A pressed, quieter, lower, and much slower voice.'},
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


// The output gain, and the rendered block size (samples)
export const outputGain = 1.3;
const blockSize = 2048;


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


// A voice's speaker in the model's speaker table - its id (the speaker of the voice's key)
export function voiceSpeaker(model, voice) {
    const speaker = model.manifest.speakers.find((item) => item.key === voice);
    if (speaker === undefined) {
        throw new Error(`The ${voice} voice's speaker isn't in the model`);
    }
    return speaker.id;
}


// The source shelf correction (dB) - every voice's source this much darker than the network predicts it: listeners heard
// the predicted source as tinny (blind A/B in the voice lab, 2026-10-23 - less tinny in 6 of the 12 voices, the same in
// the rest, never less natural; UTMOS, WER, and speaker similarity on the panel texts unchanged)
export const sourceShelf = -5;


// A voice's settings - {speaker (voiceSpeaker's), transforms (lib/transforms.js applyTransforms's - the source shelf
// correction)}
export function voiceSettings(model, voice) {
    if (!voiceTable.some((item) => item.key === voice)) {
        throw new Error(`Unknown voice ${JSON.stringify(voice)}`);
    }
    return {'speaker': voiceSpeaker(model, voice), 'transforms': [{'shelf': sourceShelf}]};
}


// A style's settings - {rate, transforms}
export function styleSettings(style = defaultStyle) {
    const entry = styleTable.find((item) => item.key === style);
    if (entry === undefined) {
        throw new Error(`Unknown style ${JSON.stringify(style)}`);
    }
    return {'rate': entry.rate ?? 1, 'transforms': entry.transforms ?? []};
}


// A sentence's frames from the model - {chunks (an iterator of the frames, a fraction of a second at a time - the network
// runs as they're taken), eq (the voice's channel equalizer)}. options: voice (a voiceTable key - its speaker and
// transforms) or speaker (the model's speaker id - its frames as the model speaks them), style (a styleTable key - its
// transforms after the voice's, and rate), rate (the speaking speed - 1 is normal), and beat (lib/style.js
// paragraphBeats's - its transforms after the style's, and its rate; default none). With createSentenceRenderer, the
// network and the renderer can run apart (in two workers, say); sentenceBlocks is the two together.
export function sentenceFrames(model, text, {voice, speaker, style, rate = 1, beat = noBeat} = {}) {
    const voiced = voice === undefined ? {speaker, 'transforms': []} : voiceSettings(model, voice);
    const styled = styleSettings(style);
    const transforms = [...voiced.transforms, ...styled.transforms, ...beat.transforms];
    const options = {'rate': 1 / (rate * styled.rate * beat.rate), 'speaker': voiced.speaker};
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
const spanPause = 0.06;


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
