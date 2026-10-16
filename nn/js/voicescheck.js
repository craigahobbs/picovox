// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

// Check the designed voices (data/nn/voices.json) with nn/js/voices.js under Node: every voice resolves (its speakers
// are in the model, its blend sums to 1, its sex matches its speakers'), speaks in every style, and its stream
// (streamVoice) equals its whole-text frames (speakVoice); voiceOptions composes the rates and transforms; bad ids
// throw; and the single-speaker voices (data/nn/slt, data/nn/bdl) still load and speak.
//
//     node nn/js/voicescheck.js [data/nn/multi] [data/nn/voices.json]
//
// It prints one JSON line - the checks run - and exits non-zero on the first failure.

/* eslint-disable no-console */

import {findStyle, findVoice, speakVoice, streamVoice, voiceBlend, voiceOptions} from './voices.js';
import {loadModel, speakFrames} from './model.js';
import assert from 'node:assert/strict';
import process from 'node:process';
import {readFileSync} from 'node:fs';


// Load an exported model (data/nn/<name>.json and .bin)
function load(path) {
    const buffer = readFileSync(`${path}.bin`);
    const weights = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
    return loadModel(JSON.parse(readFileSync(`${path}.json`, 'utf8')), weights);
}


function main() {
    const [modelPath = 'data/nn/multi', voicesPath = 'data/nn/voices.json'] = process.argv.slice(2);
    const model = load(modelPath);
    const voices = JSON.parse(readFileSync(voicesPath, 'utf8'));
    const speakers = new Map(model.manifest.speakers.map((speaker) => [speaker.key, speaker]));
    const text = 'It was the best of times. "Was it?" she asked.';
    let checks = 0;

    // The document
    assert.equal(new Set(voices.voices.map((voice) => voice.id)).size, voices.voices.length, 'voice ids are unique');
    assert.equal(new Set(voices.styles.map((style) => style.id)).size, voices.styles.length, 'style ids are unique');
    for (const sex of ['female', 'male']) {
        assert.equal(voices.voices.filter((voice) => voice.sex === sex).length, 5, `five ${sex} voices`);
    }
    for (const style of voices.styles) {
        assert.ok(style.label, `${style.id} has a label`);
        if (style.model) {
            const names = 'emotion' in style.model ? [style.model.emotion] : Object.keys(style.model);
            const known = names.every((name) => model.manifest.styles.includes(name) || name === 'neutral');
            assert.ok(known, `${style.id}'s model style exists`);
        }
        checks += 1;
    }

    // Every voice
    for (const voice of voices.voices) {
        assert.ok(voice.name && voice.description && voice.genres.length > 0, `${voice.id} has a name, description, and genres`);
        const total = Object.values(voice.blend).reduce((sum, share) => sum + share, 0);
        assert.ok(Math.abs(total - 1) < 1e-6, `${voice.id}'s blend sums to 1`);
        assert.ok(Object.keys(voice.blend).length >= 2 && Object.keys(voice.blend).length <= 4, `${voice.id} blends 2-4 speakers`);
        for (const key of Object.keys(voice.blend)) {
            assert.equal(speakers.get(key)?.role, 'reader', `${voice.id}'s speaker ${key} is a reader`);
            assert.equal(speakers.get(key).sex, voice.sex, `${voice.id}'s speaker ${key} is ${voice.sex}`);
        }
        const blend = voiceBlend(model, voice);
        assert.deepEqual(Object.keys(blend).map(Number).sort(), Object.keys(voice.blend).map((key) => speakers.get(key).id).sort());
        assert.equal(findVoice(voices, voice.id), voice);
        for (const style of voices.styles) {
            const spoken = speakVoice(model, voices, text, {'voice': voice.id, 'style': style.id});
            const finite = spoken.frames.every((frame) => frame.every(Number.isFinite));
            assert.ok(spoken.frames.length > 100 && finite, `${voice.id} ${style.id} speaks`);
            checks += 1;
        }
        // The stream equals the whole-text frames
        const whole = speakVoice(model, voices, text, {'voice': voice.id, 'style': 'whisper'});
        const stream = streamVoice(model, voices, text, {'voice': voice.id, 'style': 'whisper'});
        const streamed = [];
        for (const chunk of stream.chunks) {
            streamed.push(...chunk);
        }
        assert.deepEqual(streamed.map((frame) => Array.from(frame)), whole.frames.map((frame) => Array.from(frame)), `${voice.id} streams`);
        assert.deepEqual(stream.eq, whole.eq);
        checks += 1;
    }

    // The options - the rates multiply (listener x voice x style) into the duration multiplier, the transforms are the
    // voice's then the style's, and the model style comes from the style
    const [first] = voices.voices;
    const shout = voices.styles.find((style) => style.transform && !style.model) ?? findStyle(voices, 'neutral');
    const {options, transforms} = voiceOptions(model, voices,
        {'voice': first.id, 'style': shout.id, 'rate': 1.25, 'pauseThreshold': 0.6});
    assert.ok(Math.abs(options.rate - 1 / (1.25 * (first.rate ?? 1) * (shout.rate ?? 1))) < 1e-12, 'rates compose');
    assert.equal(options.pauseThreshold, 0.6);
    const adjustCount = Object.keys(first.adjust ?? {}).filter((name) => name !== 'rate').length;
    assert.equal(transforms.length, adjustCount + (shout.transform ?? []).length, 'the voice\'s transforms, then the style\'s');
    const happy = findStyle(voices, 'happy');
    assert.deepEqual(voiceOptions(model, voices, {'voice': first.id, 'style': 'happy'}).options.style, happy.model);
    assert.equal(voiceOptions(model, voices, {'voice': first.id}).options.style, undefined, 'neutral has no model style');
    assert.throws(() => findVoice(voices, 'nobody'), /unknown voice/u);
    assert.throws(() => findStyle(voices, 'bored'), /unknown style/u);
    assert.throws(() => voiceBlend(model, {'id': 'bad', 'blend': {'libritts-0': 1}}), /isn't in the model/u);
    assert.throws(() => voiceOptions(model, voices, {'voice': first.id, 'rate': 0}), /invalid rate/u);
    // A voice equals its blend spoken directly with its transforms
    const direct = speakFrames(model, text, {'speaker': voiceBlend(model, first), 'rate': 1 / (first.rate ?? 1)});
    assert.equal(speakVoice(model, voices, text, {'voice': first.id}).frames.length, direct.frames.length, 'the voice is its blend');
    checks += 9;

    // The single-speaker voices still load and speak
    for (const legacy of ['data/nn/slt', 'data/nn/bdl']) {
        const legacyModel = load(legacy);
        const {frames} = speakFrames(legacyModel, text);
        assert.ok(frames.length > 100 && frames.every((frame) => frame.every(Number.isFinite)), `${legacy} speaks`);
        checks += 1;
    }
    console.log(JSON.stringify({'ok': true, checks, 'voices': voices.voices.length, 'styles': voices.styles.length}));
}

main();
