// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {
    autoStyles, cueCount, fnv1a, loadStyleModel, paragraphBeat, paragraphBeats, quotationStyle, styleClasses, styleCues, styleLoader,
    styleProbabilities, styleWords, textQuotations, wordBuckets
} from '../lib/style.js';
import {strict as assert} from 'node:assert';
import {splitSentences} from '../lib/sentences.js';
import test from 'node:test';


// The package's style network, and one that styles every quotation (its thresholds zero)
const styleModule = await import('../lib/models/styleModel.js');
const model = loadStyleModel(styleModule);
const eager = {...model, 'thresholds': {'angry': 0, 'shouting': 0, 'fearful': 0}};


test('styleWords', () => {
    // Lowercase words, accents off, the curly apostrophe straight
    assert.deepEqual(styleWords('Señor, it’s DONE!'), ['senor', 'it\'s', 'done']);
    assert.deepEqual(styleWords('... --'), []);
});


test('fnv1a', () => {
    // FNV-1a over UTF-16 code units - as picovox_train/style.py's word_hash
    assert.equal(fnv1a(''), 2166136261);
    assert.equal(fnv1a('a'), 3826002220);
    assert.equal(fnv1a('hello'), 0x4f9f2cab);
    assert.deepEqual(wordBuckets('Hello, a!', 4096), [0x4f9f2cab % 4096, 3826002220 % 4096]);
});


test('styleCues', () => {
    // Exclamations (to three), a question, capitals, emphasis, length, a short quotation, the attribution's exclamation, and
    // a dash
    assert.deepEqual(styleCues({'text': '“STOP! Stop—now!”', 'before': 'he cried!'}),
        [2 / 3, 0, 5 / 11, 0, Math.log1p(3) / 4, 1, 1, 1]);
    assert.deepEqual(styleCues({'text': '"Why? _Why?_ Tell me why, why, why!!!!"', 'after': 'she said.'}),
        [1, 1, 3 / 21, 1, Math.log1p(7) / 4, 0, 0, 0]);

    // No letters, and no attribution
    assert.deepEqual(styleCues({'text': '...'}), [0, 0, 0, 0, 0, 1, 0, 0]);
    assert.equal(styleCues({'text': 'x'}).length, cueCount);
});


test('loadStyleModel', () => {
    // The int8 tensors at their shapes, scaled; the module's config and thresholds
    const {buckets, dim, hidden} = styleModule.styleModel.config;
    assert.deepEqual(model.config, styleModule.styleModel.config);
    assert.deepEqual(model.thresholds, styleModule.styleModel.thresholds);
    assert.deepEqual(Object.fromEntries(Object.entries(model.weights).map(([name, values]) => [name, values.length])), {
        'embedding': buckets * dim, 'hidden': hidden * (6 * dim + cueCount), 'hiddenBias': hidden,
        'output': styleClasses.length * hidden, 'outputBias': styleClasses.length
    });
    assert(model.weights.embedding.every((value) => Math.abs(value) <= 128 * styleModule.styleModel.scales.embedding));
});


test('styleProbabilities', () => {
    // A distribution over the styles, its words and narration counted
    const quotation = {'text': '“Get out!”', 'before': '', 'after': 'he shouted.', 'previous': 'He was furious.', 'next': ''};
    const probabilities = styleProbabilities(model, quotation);
    assert.equal(probabilities.length, styleClasses.length);
    assert(Math.abs(probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-9);
    assert(probabilities.every((value) => value > 0 && value < 1));
    assert.notDeepEqual(styleProbabilities(model, {...quotation, 'after': 'he whispered.'}), probabilities);

    // A quotation alone - no attribution or context
    const alone = styleProbabilities(model, {'text': 'Hm.'});
    assert.deepEqual(alone, styleProbabilities(model, {'text': 'Hm.', 'before': '', 'after': '', 'previous': '', 'next': ''}));
});


test('quotationStyle', () => {
    // Neutral unless a style reaches its threshold - then the most probable of those that do
    const quotation = {'text': '“Get out!”', 'after': 'he shouted.', 'previous': 'He was furious.'};
    const [, angry, shouting, fearful] = styleProbabilities(model, quotation);
    assert.equal(quotationStyle(model, quotation), 'neutral');
    const best = [['angry', angry], ['shouting', shouting], ['fearful', fearful]].sort((first, second) => second[1] - first[1]);
    assert.equal(quotationStyle(eager, quotation), best[0][0]);
    const without = {...eager, 'thresholds': {...eager.thresholds, [best[0][0]]: 1.01}};
    assert.equal(quotationStyle(without, quotation), best[1][0]);
});


test('textQuotations', () => {
    // Each quotation's text, its attribution in its sentences, and the sentences around it
    const sentences = splitSentences('He stood. “Go home,” she said, “and stay there.”\n\n“No!”\n\nShe sighed.');
    assert.deepEqual(textQuotations(sentences), [
        {'quotation': 0, 'first': 1, 'last': 1, 'before': '', 'after': 'she said,', 'text': '“Go home,”',
            'previous': 'He stood.', 'next': '“No!”'},
        {'quotation': 1, 'first': 1, 'last': 1, 'before': 'she said,', 'after': '', 'text': '“and stay there.”',
            'previous': 'He stood.', 'next': '“No!”'},
        {'quotation': 2, 'first': 2, 'last': 2, 'before': '', 'after': '', 'text': '“No!”',
            'previous': '“Go home,” she said, “and stay there.”', 'next': 'She sighed.'}
    ]);

    // A quotation across sentences - its texts joined, from its first sentence to its last
    const across = textQuotations(splitSentences('“Wait. Listen.”'));
    assert.deepEqual(across.map(({text, first, last}) => [text, first, last]), [['“Wait. Listen.”', 0, 1]]);
    assert.deepEqual(textQuotations([]), []);
});


test('autoStyles', () => {
    // A quotation the network leaves neutral is dialogue, the narration neutral; adjacent spans of one style merge; a
    // sentence without a quotation has no spans
    const sentences = splitSentences('It was late. “Hello,” “there,” she said.');
    const auto = autoStyles(model, sentences);
    assert.equal(auto.spans(0), null);
    assert.deepEqual(auto.spans(1), [
        {'text': '“Hello,” “there,”', 'style': 'dialogue'}, {'text': 'she said.', 'style': 'neutral'}
    ]);

    // A confident quotation in its style - asked for again, the same
    const styled = autoStyles(eager, sentences);
    const [quotation] = textQuotations(sentences);
    assert.equal(styled.spans(1)[0].style, quotationStyle(eager, quotation));
    assert.deepEqual(styled.spans(1), styled.spans(1));
});


test('paragraphBeats', () => {
    // Each paragraph as one arc - its first sentence lifted and louder, the pitch settling to -lift/2 by its last
    // sentence, which is slowed; a one-sentence paragraph is only lifted
    const {lift, level, settle} = paragraphBeat;
    const beats = paragraphBeats(splitSentences('One. Two. Three.\n\nFour.'));
    assert.deepEqual(beats, [
        {'transforms': [{'pitchScale': 1 + lift}, {level}, {'frication': level}], 'rate': 1},
        {'transforms': [{'pitchScale': 1 + lift - 1.5 * lift * 0.5}], 'rate': 1},
        {'transforms': [{'pitchScale': 1 + lift - 1.5 * lift}], 'rate': 1 - settle},
        {'transforms': [{'pitchScale': 1 + lift}, {level}, {'frication': level}], 'rate': 1}
    ]);

    // Sentences without a paragraph start begin one
    assert.deepEqual(paragraphBeats([{'index': 0, 'paragraphStart': false}]), [beats[3]]);
    assert.deepEqual(paragraphBeats([]), []);
});


test('styleLoader', async () => {
    // Imported once, on first use
    let imports = 0;
    const loader = styleLoader(() => {
        imports += 1;
        return Promise.resolve(styleModule);
    });
    const first = await loader();
    assert.equal(await loader(), first);
    assert.equal(imports, 1);
    assert.deepEqual(first.thresholds, model.thresholds);

    // A failed import is tried again
    let attempts = 0;
    const flaky = styleLoader(() => {
        attempts += 1;
        return attempts === 1 ? Promise.reject(new Error('offline')) : Promise.resolve(styleModule);
    });
    await assert.rejects(flaky(), {'message': 'offline'});
    assert.deepEqual((await flaky()).thresholds, model.thresholds);
    assert.equal(attempts, 2);
});
