// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// The designed voices and styles of the multi-speaker, multi-style model (data/nn/voices.json) -
// browser and Node, no dependencies beyond nn/js. A voice is a blend of the speaker table's readers (by speaker key)
// with mild interpretable adjustments (pitch, tract, and effort transforms of the parameter frames) and a default
// speaking rate; a style is the model's style embedding at an intensity, parameter transforms (whisper, soft,
// shouting), a rate, or a mix of them.
//
//     import {speakVoice} from './nn/js/voices.js';
//     const voices = JSON.parse(voicesText);                // data/nn/voices.json
//     const {frames, eq} = speakVoice(model, voices, 'Hello, world.', {'voice': 'narrator-f', 'style': 'happy'});
//     const samples = render(frames, {'sampleRate': 44100, eq});
//
// voiceOptions resolves a voice and a style to the model's options (speakFrames/streamFrames) and the frame
// transforms (lib/transforms.js applyTransforms) to apply to the frames; streamVoice transforms a stream's chunks.

import {speakFrames, streamFrames} from '../../lib/model.js';
import {applyTransforms} from '../../lib/transforms.js';


// A voice of the voices document by id
export function findVoice(voices, id) {
    const voice = voices.voices.find((item) => item.id === id);
    if (!voice) {
        throw new Error(`unknown voice ${JSON.stringify(id)}`);
    }
    return voice;
}


// A style of the voices document by id (null or undefined is neutral)
export function findStyle(voices, id) {
    if (id === null || id === undefined) {
        return voices.styles.find((item) => item.id === 'neutral') ?? {'id': 'neutral'};
    }
    const style = voices.styles.find((item) => item.id === id);
    if (!style) {
        throw new Error(`unknown style ${JSON.stringify(id)}`);
    }
    return style;
}


// A voice's speaker blend as the model's {id: weight} - the blend's speaker keys looked up in the manifest's speakers
export function voiceBlend(model, voice) {
    const ids = new Map(model.manifest.speakers.map((speaker) => [speaker.key, speaker.id]));
    const blend = {};
    for (const [key, share] of Object.entries(voice.blend)) {
        if (!ids.has(key)) {
            throw new Error(`the ${voice.id} voice's speaker ${key} isn't in the model`);
        }
        blend[ids.get(key)] = share;
    }
    return blend;
}


// The voice's adjustments as frame transforms - [{'pitchScale'}, {'tractScale'}, {'effort'}] (identity values omitted)
export function voiceTransforms(voice) {
    const {pitchScale = 1, tractScale = 1, effort = 0} = voice.adjust ?? {};
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
    return transforms;
}


// The model options and frame transforms of a voice in a style - {options (speakFrames's: speaker, style, rate - a
// duration multiplier), transforms}. options.rate is the listener's speaking speed (1 is the voice's own); other
// options (pauseThreshold, bandwidthScale, ...) pass through.
export function voiceOptions(model, voices, {voice: voiceId, style: styleId, rate = 1, ...rest} = {}) {
    const voice = findVoice(voices, voiceId);
    const style = findStyle(voices, styleId);
    const speed = rate * (voice.rate ?? 1) * (style.rate ?? 1);
    if (!(speed > 0)) {
        throw new Error(`invalid rate ${rate}`);
    }
    const options = {...rest, 'speaker': voiceBlend(model, voice), 'rate': 1 / speed};
    if (style.model) {
        options.style = style.model;
    }
    return {options, 'transforms': [...voiceTransforms(voice), ...(style.transform ?? [])]};
}


// Speak text in a voice and style - speakFrames's {frames, eq, sentences}, the frames transformed
export function speakVoice(model, voices, text, voiceStyle = {}) {
    const {options, transforms} = voiceOptions(model, voices, voiceStyle);
    const spoken = speakFrames(model, text, options);
    return {...spoken, 'frames': applyTransforms(spoken.frames, transforms)};
}


// Speak text in a voice and style as a stream - streamFrames's {eq, sentences, chunks}, each chunk transformed (the
// transforms are per frame, so the stream's frames equal speakVoice's)
export function streamVoice(model, voices, text, voiceStyle = {}) {
    const {options, transforms} = voiceOptions(model, voices, voiceStyle);
    const stream = streamFrames(model, text, options);
    function* chunks() {
        for (const chunk of stream.chunks) {
            yield applyTransforms(chunk, transforms);
        }
    }
    return {...stream, 'chunks': chunks()};
}
