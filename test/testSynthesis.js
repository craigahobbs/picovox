// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {autoStyles, paragraphBeats} from '../lib/style.js';
import {
    blockSize, createSentenceRenderer, defaultSampleRate, defaultVoice, modelLoader, outputGain, paragraphPause, sentenceBlocks,
    sentenceFrames, sentencePause, softLimit, sourceShelf, spanFrames, styleSettings, styleTable, textBlocks, voiceSettings,
    voiceSpeaker, voiceTable
} from '../lib/synthesis.js';
import {concatBlocks, render} from '../lib/render.js';
import {speechText, splitSentences} from '../lib/sentences.js';
import {Buffer} from 'node:buffer';
import {applyTransforms} from '../lib/transforms.js';
import {strict as assert} from 'node:assert';
import {readFileSync} from 'node:fs';
import {speakFrames} from '../lib/model.js';
import test from 'node:test';


// The package's models
const importG2p = () => import('../lib/models/g2pModel.js');
const importVoice = () => import('../lib/models/voiceModel.js');
const model = await modelLoader(importG2p, importVoice)();


// Whether two sample arrays are the same, sample for sample (their bytes - a failed comparison costs nothing, where
// assert.deepEqual's diff of every sample can exhaust memory)
function sameSamples(actual, expected) {
    const bytes = (array) => Buffer.from(array.buffer, array.byteOffset, array.byteLength);
    return bytes(actual).equals(bytes(expected));
}


// A sentence spoken by the pipeline directly - the pause, then the frames rendered whole, with the gain and the limiter
function directSentence(text, {speaker, rate = 1, seed = 1, pause = 0, sampleRate = defaultSampleRate}) {
    const {frames, eq} = speakFrames(model, text, {'rate': 1 / rate, speaker});
    const rendered = render(frames, {sampleRate, seed, eq});
    const pauseLength = Math.round(sampleRate * pause / rate);
    const samples = new Float32Array(pauseLength + rendered.length);
    for (const [index, value] of rendered.entries()) {
        samples[pauseLength + index] = softLimit(value * outputGain);
    }
    return samples;
}


test('synthesis, constants', () => {
    assert.equal(defaultSampleRate, 24000);
    assert.equal(blockSize, 2048);
    assert.equal(outputGain, 1.3);
    assert.equal(sentencePause, 0.24);
    assert.equal(paragraphPause, 0.4);
    assert.equal(defaultVoice, 'male');
    assert.deepEqual(voiceTable.map((voice) => voice.key), [
        'male', 'female', 'sarah', 'helen', 'amy', 'kate', 'grace', 'james', 'robert', 'michael', 'david', 'thomas'
    ]);

    // Each voice's speakers are readers of its sex in the model's speaker table
    const speakers = new Map(model.manifest.speakers.map((speaker) => [speaker.key, speaker]));
    for (const voice of voiceTable) {
        assert(voice.label && voice.description, voice.key);
        for (const key of Object.keys(voice.blend ?? {[voice.speaker]: 1})) {
            assert.equal(speakers.get(key).sex, voice.sex, `${voice.key}'s ${key}`);
            assert.equal(speakers.get(key).role, 'reader', `${voice.key}'s ${key}`);
        }
    }

    // The designed voices are data/nn/voices.json's - its names, sexes, descriptions, blends, adjustments, and rates
    const designed = JSON.parse(readFileSync('data/nn/voices.json', 'utf8')).voices;
    assert.deepEqual(voiceTable.slice(2), designed.map((voice) => ({
        'key': voice.name.toLowerCase(), 'label': voice.name, 'sex': voice.sex, 'rate': voice.rate, 'blend': voice.blend,
        'adjust': voice.adjust, 'description': voice.description
    })));
});


test('softLimit', () => {
    assert.equal(softLimit(0), 0);
    assert.equal(softLimit(0.5), 0.5);
    assert.equal(softLimit(-0.8), -0.8);
    assert.equal(softLimit(0.9), 0.8 + 0.2 * Math.tanh(0.5));
    assert.equal(softLimit(-0.9), -(0.8 + 0.2 * Math.tanh(0.5)));
    assert(softLimit(1.5) < 1);
    assert(softLimit(-1.5) > -1);
    assert.equal(softLimit(100), 1);
});


test('modelLoader', async () => {
    // The model is loaded once
    let imports = 0;
    const load = modelLoader(importG2p, () => {
        imports += 1;
        return importVoice();
    });
    const first = load();
    assert.equal(load(), first);
    const loaded = await first;
    assert.equal(loaded.manifest.kind, 'multi');
    assert.equal(imports, 1);
    assert.equal(await load(), loaded);
    assert.equal(imports, 1);
});


test('modelLoader, failure', async () => {
    // A failed load is retried on the next call
    let attempts = 0;
    const load = modelLoader(importG2p, () => {
        attempts += 1;
        return attempts === 1 ? Promise.reject(new Error('offline')) : importVoice();
    });
    await assert.rejects(load(), {'message': 'offline'});
    await new Promise((resolve) => {
        setTimeout(resolve, 0);
    });
    const loaded = await load();
    assert.equal(loaded.manifest.kind, 'multi');
    assert.equal(attempts, 2);
});


test('voiceSpeaker', () => {
    assert.equal(voiceSpeaker(model, 'female'), 0);
    assert.equal(voiceSpeaker(model, 'male'), 1);

    // A designed voice's speaker is its blend, by speaker id
    const ids = new Map(model.manifest.speakers.map((speaker) => [speaker.key, speaker.id]));
    const sarah = voiceTable.find((voice) => voice.key === 'sarah');
    const blend = Object.fromEntries(Object.entries(sarah.blend).map(([key, share]) => [ids.get(key), share]));
    assert.deepEqual(voiceSpeaker(model, 'sarah'), blend);

    // A model without the voice's speaker
    const other = {'manifest': {'speakers': [{'id': 0, 'key': 'arctic-slt'}]}};
    assert.throws(() => voiceSpeaker(other, 'male'), {'message': 'The male voice\'s speaker (arctic-bdl) isn\'t in the model'});
    assert.throws(() => voiceSpeaker(other, 'sarah'), {'message': 'The sarah voice\'s speaker (libritts-1462) isn\'t in the model'});
});


test('voiceSettings', () => {
    // A package voice - its speaker, no rate, its softer pulse, and the source shelf correction
    assert.deepEqual(voiceSettings(model, 'male'), {'speaker': 1, 'rate': 1, 'transforms': [{'rdScale': 1.2}, {'shelf': sourceShelf}]});
    assert.equal(sourceShelf, -5);

    // A designed voice - its blend, rate, and adjustments as transforms (identity values omitted), then the correction
    const helen = voiceSettings(model, 'helen');
    assert.deepEqual(helen.speaker, voiceSpeaker(model, 'helen'));
    assert.equal(helen.rate, 0.9);
    assert.deepEqual(helen.transforms, [{'pitchScale': 0.95}, {'tractScale': 0.98}, {'shelf': -5}]);
    assert.deepEqual(voiceSettings(model, 'grace').transforms, [{'effort': -2}, {'shelf': -5}]);
    assert.deepEqual(voiceSettings(model, 'david').transforms, [{'shelf': -5}]);
    assert.throws(() => voiceSettings(model, 'nobody'), {'message': 'Unknown voice "nobody"'});
});


test('styles', () => {
    // The styles are data/nn/voices.json's - their keys, labels, kinds, emotions, transforms, rates, and descriptions
    const designed = JSON.parse(readFileSync('data/nn/voices.json', 'utf8')).styles;
    assert.deepEqual(styleTable.map((style) => style.key), designed.map((style) => style.id));
    for (const [style, entry] of styleTable.map((item, index) => [item, designed[index]])) {
        assert.deepEqual([style.label, style.kind, style.description], [entry.label, entry.kind, entry.description]);
        assert.deepEqual([style.emotion, style.transforms, style.rate], [entry.model, entry.transform, entry.rate]);
    }

    // A style's settings - neutral is nothing; an emotion, transforms, and a rate
    assert.deepEqual(styleSettings(), {'emotion': undefined, 'rate': 1, 'transforms': []});
    assert.deepEqual(styleSettings('fearful'), {
        'emotion': {'emotion': 'fearful', 'intensity': 0.6}, 'rate': 1.08,
        'transforms': [{'whisper': 0.15}, {'pitchScale': 1.1}, {'level': 2}, {'frication': 2}]
    });
    assert.deepEqual(styleSettings('whisper'), {
        'emotion': undefined, 'rate': 1, 'transforms': [{'whisper': 1}, {'level': 5}, {'frication': 5}]
    });
    assert.throws(() => styleSettings('bored'), {'message': 'Unknown style "bored"'});
});


test('sentenceFrames, styles', () => {
    // A styled voice's frames are its emotion's, at the voice's and the style's rates, with the voice's transforms and
    // then the style's
    const text = 'Get out of my house!';
    for (const [voice, style] of [['helen', 'fearful'], ['male', 'whisper'], ['david', 'angry']]) {
        const {chunks, eq} = sentenceFrames(model, text, {voice, style, 'rate': 1.2});
        const voiced = voiceSettings(model, voice);
        const styled = styleSettings(style);
        const options = {'speaker': voiced.speaker, 'style': styled.emotion, 'rate': 1 / (1.2 * voiced.rate * styled.rate)};
        const expected = speakFrames(model, text, options);
        const frames = applyTransforms(expected.frames, [...voiced.transforms, ...styled.transforms]);
        const spoken = [...chunks].flat().map((frame) => Array.from(frame));
        assert.deepEqual(spoken, frames.map((frame) => Array.from(frame)), `${voice} ${style}`);
        assert.deepEqual(eq, expected.eq);
    }

    // A speaker with a style, and neutral as no style at all
    const speaker = voiceSpeaker(model, 'female');
    const shouted = sentenceFrames(model, text, {speaker, 'style': 'shouting'});
    const loud = applyTransforms(speakFrames(model, text, {speaker}).frames, styleSettings('shouting').transforms);
    assert.deepEqual([...shouted.chunks].flat(), loud);
    const neutral = sentenceFrames(model, text, {speaker, 'style': 'neutral'});
    assert.deepEqual([...neutral.chunks].flat(), [...sentenceFrames(model, text, {speaker}).chunks].flat());

    // sentenceBlocks and textBlocks take the style too
    assert.deepEqual(concatBlocks([...sentenceBlocks(model, text, {'voice': 'amy', 'style': 'sad', 'seed': 2})]),
        concatBlocks([...sentenceBlocks(model, text, {'voice': 'amy', 'style': 'sad', 'seed': 2})]));
    assert.notDeepEqual(concatBlocks([...textBlocks(model, 'Hello there.', {'voice': 'amy', 'style': 'happy'})]),
        concatBlocks([...textBlocks(model, 'Hello there.', {'voice': 'amy'})]));
});


test('sentenceFrames, pitch range', () => {
    // A style with a pitch range - the text's frames gathered, then every transform applied to them at once
    const text = 'Down with the door!';
    const voiced = voiceSettings(model, 'james');
    const styled = styleSettings('exclaim');
    const {chunks} = sentenceFrames(model, text, {'voice': 'james', 'style': 'exclaim'});
    const spoken = [...chunks];
    assert.equal(spoken.length, 1);
    const expected = speakFrames(model, text, {'speaker': voiced.speaker, 'rate': 1 / (voiced.rate * styled.rate)});
    const frames = applyTransforms(expected.frames, [...voiced.transforms, ...styled.transforms]);
    assert.deepEqual(spoken[0].map((frame) => Array.from(frame)), frames.map((frame) => Array.from(frame)));

    // No speech - no chunks
    assert.deepEqual([...sentenceFrames(model, '--', {'voice': 'james', 'style': 'exclaim'}).chunks], []);
});


test('sentenceFrames, beat', () => {
    // A beat's transforms follow the voice's and the style's, its rate times theirs
    const text = 'The birch canoe slid on the smooth planks.';
    const beat = {'transforms': [{'pitchScale': 1.08}, {'level': 2}], 'rate': 0.93};
    const {chunks} = sentenceFrames(model, text, {'voice': 'helen', 'style': 'soft', 'rate': 1.1, beat});
    const voiced = voiceSettings(model, 'helen');
    const styled = styleSettings('soft');
    const expected = speakFrames(model, text, {'speaker': voiced.speaker, 'rate': 1 / (1.1 * voiced.rate * styled.rate * beat.rate)});
    const frames = applyTransforms(expected.frames, [...voiced.transforms, ...styled.transforms, ...beat.transforms]);
    assert.deepEqual([...chunks].flat().map((frame) => Array.from(frame)), frames.map((frame) => Array.from(frame)));

    // Spans take the beat in every span; sentenceBlocks passes it on
    const spans = [{'text': 'She said,', 'style': 'neutral'}, {'text': 'hello.', 'style': 'dialogue'}];
    const spanned = spanFrames(model, spans, {'voice': 'helen', beat});
    const first = sentenceFrames(model, 'She said,', {'voice': 'helen', beat});
    assert.deepEqual([...spanned.chunks][0], [...first.chunks][0]);
    assert.deepEqual(concatBlocks([...sentenceBlocks(model, text, {'voice': 'helen', beat})]),
        concatBlocks([...sentenceBlocks(model, text, {'voice': 'helen', beat})]));
    assert.notDeepEqual(concatBlocks([...sentenceBlocks(model, text, {'voice': 'helen', beat})]),
        concatBlocks([...sentenceBlocks(model, text, {'voice': 'helen'})]));
});


test('textBlocks, auto', () => {
    // Style "auto" - each sentence in its auto spans (the quotations in the dialogue style) with its paragraph beat; the
    // narration's sentences unspanned
    const text = 'It was late. "Who is there?" she asked.\n\nNobody answered.';
    const sentences = splitSentences(text);
    const auto = autoStyles(sentences);
    const beats = paragraphBeats(sentences);
    assert.deepEqual([auto.spans(0), auto.spans(1).map((span) => span.style)], [null, ['dialogue']]);
    const expected = sentences.map((sentence) => {
        const spans = auto.spans(sentence.index)?.map((span) => ({'text': speechText(span.text), 'style': span.style}));
        return concatBlocks([...sentenceBlocks(model, speechText(sentence.text), {
            'voice': 'kate', spans, 'beat': beats[sentence.index], 'seed': 3 + sentence.index,
            'pause': sentence.paragraphStart ? paragraphPause : sentencePause
        })]);
    });
    assert(sameSamples(concatBlocks([...textBlocks(model, text, {'voice': 'kate', 'style': 'auto', 'seed': 3})]), concatBlocks(expected)));
});


test('sentenceFrames, voices', () => {
    // A voice's frames are its blend spoken at its rate times the listener's, with its adjustments
    const text = 'The birch canoe slid on the smooth planks.';
    for (const key of ['helen', 'david']) {
        const {chunks, eq} = sentenceFrames(model, text, {'voice': key, 'rate': 1.1});
        const {speaker, rate, transforms} = voiceSettings(model, key);
        const expected = speakFrames(model, text, {speaker, 'rate': 1 / (1.1 * rate)});
        assert.deepEqual([...chunks].flat().map((frame) => Array.from(frame)),
            applyTransforms(expected.frames, transforms).map((frame) => Array.from(frame)), key);
        assert.deepEqual(eq, expected.eq);
    }

    // A package voice is its speaker with its adjustments and the source shelf correction, exactly
    const {transforms} = voiceSettings(model, 'female');
    assert.deepEqual(transforms, [{'rdScale': 1.2}, {'shelf': sourceShelf}]);
    const voiced = sentenceFrames(model, text, {'voice': 'female'});
    const spoken = sentenceFrames(model, text, {'speaker': voiceSpeaker(model, 'female')});
    assert.deepEqual([...voiced.chunks].flat().map((frame) => Array.from(frame)),
        applyTransforms([...spoken.chunks].flat(), transforms).map((frame) => Array.from(frame)));

    // sentenceBlocks and textBlocks take the voice too
    const correction = {transforms, 'rate': 1};
    assert(sameSamples(concatBlocks([...sentenceBlocks(model, text, {'voice': 'female', 'seed': 2})]),
        concatBlocks([...sentenceBlocks(model, text, {'speaker': voiceSpeaker(model, 'female'), 'seed': 2, 'beat': correction})])));
    assert(concatBlocks([...textBlocks(model, 'Hello there.', {'voice': 'amy'})]).some((value) => value !== 0));
});


test('sentenceBlocks', () => {
    // The pause, then the sentence in blocks - exactly the pipeline run directly
    const text = 'The birch canoe slid on the smooth planks.';
    const options = {'speaker': voiceSpeaker(model, 'female'), 'rate': 1.2, 'seed': 3, 'pause': paragraphPause};
    const blocks = [...sentenceBlocks(model, text, options)];
    assert.equal(blocks[0].length, Math.round(defaultSampleRate * paragraphPause / 1.2));
    assert(blocks[0].every((value) => value === 0));
    assert(blocks.slice(1, -1).every((block) => block.length === blockSize));
    assert(blocks[blocks.length - 1].length <= blockSize);
    assert.deepEqual(concatBlocks(blocks), directSentence(text, options));

    // Another sample rate
    const fast = {'speaker': voiceSpeaker(model, 'male'), 'seed': 5, 'sampleRate': 16000};
    assert.deepEqual(concatBlocks([...sentenceBlocks(model, 'Call me Ishmael.', fast)]), directSentence('Call me Ishmael.', fast));
});


test('sentenceBlocks, defaults', () => {
    // No pause, the model's default speaker, seed 1, normal speed
    const blocks = [...sentenceBlocks(model, 'Hello.')];
    assert.deepEqual(concatBlocks(blocks), directSentence('Hello.', {'speaker': model.manifest.defaultSpeaker}));
    assert.notEqual(blocks[0][0], 0);
});


test('sentenceBlocks, streaming', () => {
    // The blocks are made as they're taken - the network's first chunks come before the rest of the sentence is spoken
    const text = 'It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.';
    const options = {'speaker': voiceSpeaker(model, 'male'), 'pause': sentencePause};
    const iterator = sentenceBlocks(model, text, options);
    const startFirst = performance.now();
    const first = iterator.next();
    const second = iterator.next();
    const firstMs = performance.now() - startFirst;
    assert.equal(first.value.length, Math.round(defaultSampleRate * sentencePause));
    assert.equal(second.value.length, blockSize);
    const rest = [...iterator];
    const startWhole = performance.now();
    const whole = directSentence(text, options);
    const wholeMs = performance.now() - startWhole;
    assert(sameSamples(concatBlocks([first.value, second.value, ...rest]), whole));
    assert(firstMs < wholeMs, `first block ${firstMs} ms, whole sentence ${wholeMs} ms`);

    // An empty sentence is only its pause
    assert.deepEqual([...sentenceBlocks(model, '', options)].map((block) => block.length), [Math.round(defaultSampleRate * sentencePause)]);
});


test('sentenceFrames and createSentenceRenderer', () => {
    // The network and the renderer apart - the frames' chunks pushed to the sentence's renderer - are sentenceBlocks
    const text = 'The birch canoe slid on the smooth planks.';
    const options = {'speaker': voiceSpeaker(model, 'female'), 'rate': 0.9, 'seed': 4, 'pause': sentencePause};
    const {chunks, eq} = sentenceFrames(model, text, options);
    assert.equal(eq.length, 24);
    const blocks = [];
    const renderer = createSentenceRenderer({'onBlock': (block) => blocks.push(block), eq, ...options});
    assert.equal(blocks.length, 1);
    assert.equal(blocks[0].length, Math.round(defaultSampleRate * sentencePause / 0.9));
    for (const chunk of chunks) {
        renderer.push(chunk);
    }
    renderer.end();
    assert.deepEqual(concatBlocks(blocks), concatBlocks([...sentenceBlocks(model, text, options)]));

    // The defaults - no pause, seed 1, normal speed, no equalizer
    const plain = [];
    const plainRenderer = createSentenceRenderer({'onBlock': (block) => plain.push(block)});
    assert.equal(plain.length, 0);
    plainRenderer.push(speakFrames(model, 'Hello.').frames);
    plainRenderer.end();
    assert(plain.length > 0);
    assert.equal([...sentenceFrames(model, 'Hello.').chunks].flat().length, speakFrames(model, 'Hello.').frames.length);
});


test('textBlocks', () => {
    // Each sentence after its pause - the paragraph's longer - spoken with the seed plus its index
    const text = 'Chapter IV\n\nIt was late \u2014 very late. "Who is there?" she asked.\n\nNobody answered.';
    const speaker = voiceSpeaker(model, 'female');
    const expected = splitSentences(text).map((sentence) => directSentence(speechText(sentence.text), {
        speaker, 'rate': 0.8, 'seed': 1000 + sentence.index, 'pause': sentence.paragraphStart ? paragraphPause : sentencePause
    }));
    assert.equal(expected.length, 5);
    assert(sameSamples(concatBlocks([...textBlocks(model, text, {speaker, 'rate': 0.8, 'seed': 1000})]), concatBlocks(expected)));

    // Defaults - seed 1, normal speed, 24 kHz
    const defaults = concatBlocks([...textBlocks(model, 'One. Two.')]);
    const speakerDefault = model.manifest.defaultSpeaker;
    assert.deepEqual(defaults, concatBlocks([
        directSentence('One.', {'speaker': speakerDefault, 'seed': 1, 'pause': paragraphPause}),
        directSentence('Two.', {'speaker': speakerDefault, 'seed': 2, 'pause': sentencePause})
    ]));

    // No sentences
    assert.deepEqual([...textBlocks(model, ' -- ', {speaker})], []);
});
