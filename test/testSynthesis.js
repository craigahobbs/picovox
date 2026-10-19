// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {
    blockSize, createSentenceRenderer, defaultSampleRate, defaultVoice, modelLoader, outputGain, paragraphPause, sentenceBlocks,
    sentenceFrames, sentencePause, softLimit, textBlocks, voiceSpeaker, voiceTable
} from '../lib/synthesis.js';
import {speechText, splitSentences} from '../lib/sentences.js';
import {strict as assert} from 'node:assert';
import {render} from '../lib/render.js';
import {speakFrames} from '../lib/model.js';
import test from 'node:test';


// The package's models
const importG2p = () => import('../lib/models/g2pModel.js');
const importVoice = () => import('../lib/models/voiceModel.js');
const model = await modelLoader(importG2p, importVoice)();


// Blocks concatenated
function concat(blocks) {
    const samples = new Float32Array(blocks.reduce((sum, block) => sum + block.length, 0));
    let offset = 0;
    for (const block of blocks) {
        samples.set(block, offset);
        offset += block.length;
    }
    return samples;
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
    assert.deepEqual(voiceTable.map((voice) => voice.key), ['male', 'female']);

    // Each voice's speaker is a reader of its sex in the model's speaker table
    for (const voice of voiceTable) {
        const speaker = model.manifest.speakers[voiceSpeaker(model, voice.key)];
        assert.equal(speaker.key, voice.speaker);
        assert.equal(speaker.sex, voice.sex);
        assert.equal(speaker.role, 'reader');
    }
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

    // A model without the voice's speaker
    const other = {'manifest': {'speakers': [{'id': 0, 'key': 'arctic-slt'}]}};
    assert.throws(() => voiceSpeaker(other, 'male'), {'message': 'The male voice\'s speaker (arctic-bdl) isn\'t in the model'});
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
    assert.deepEqual(concat(blocks), directSentence(text, options));

    // Another sample rate
    const fast = {'speaker': voiceSpeaker(model, 'male'), 'seed': 5, 'sampleRate': 16000};
    assert.deepEqual(concat([...sentenceBlocks(model, 'Call me Ishmael.', fast)]), directSentence('Call me Ishmael.', fast));
});


test('sentenceBlocks, defaults', () => {
    // No pause, the model's default speaker, seed 1, normal speed
    const blocks = [...sentenceBlocks(model, 'Hello.')];
    assert.deepEqual(concat(blocks), directSentence('Hello.', {'speaker': model.manifest.defaultSpeaker}));
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
    assert.deepEqual(concat([first.value, second.value, ...rest]), whole);
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
    assert.deepEqual(concat(blocks), concat([...sentenceBlocks(model, text, options)]));

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
    assert.deepEqual(concat([...textBlocks(model, text, {speaker, 'rate': 0.8, 'seed': 1000})]), concat(expected));

    // Defaults - seed 1, normal speed, 24 kHz
    const defaults = concat([...textBlocks(model, 'One. Two.')]);
    const speakerDefault = model.manifest.defaultSpeaker;
    assert.deepEqual(defaults, concat([
        directSentence('One.', {'speaker': speakerDefault, 'seed': 1, 'pause': paragraphPause}),
        directSentence('Two.', {'speaker': speakerDefault, 'seed': 2, 'pause': sentencePause})
    ]));

    // No sentences
    assert.deepEqual([...textBlocks(model, ' -- ', {speaker})], []);
});
