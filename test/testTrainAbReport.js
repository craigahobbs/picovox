// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {
    abReport, alignedDistance, chooseTrials, formatReport, frameFeatures, makeQuestions, seededRandom, signTest, wilson
} from '../tools/trainabreport.js';
import {NAMES} from '../lib/params.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


// Three voices' trials - each a sentence and a passage
const trials = ['male', 'female', 'sarah'].flatMap((voice, index) => [
    {voice, 'text': index === 1 ? 'Two.' : 'One.', 'difference': 0.1 * index},
    {voice, 'text': 'A passage.', 'difference': 0.2}
]);


// A round's trials, with two repeats
const questions = makeQuestions({trials, 'round': 1, 'repeats': 2});


// Answers ({id: answer}) choosing a version for each question in turn (null - Same)
function answer(versions) {
    return Object.fromEntries(versions.map((version, index) => [questions[index].id, {version, 'choice': version ?? 'Same'}]));
}


test('trainabreport.seededRandom', () => {
    const first = seededRandom(7);
    const second = seededRandom(7);
    const values = [first(), first(), first()];
    assert.deepEqual(values, [second(), second(), second()]);
    assert(values.every((value) => value >= 0 && value < 1));
    assert.notEqual(values[0], seededRandom(8)());
});


// A frame - every parameter at its default, with some set
function frame(values = {}) {
    return Float64Array.from(NAMES.map((name) => values[name] ?? {'f0': 100, 'f1': 500, 'f2': 1500, 'f3': 2500, 'f4': 3500,
        'b1': 80, 'b2': 90, 'b3': 150, 'rd': 1}[name] ?? 0));
}


test('trainabreport.frameFeatures', () => {
    // Every step-th frame, each feature scaled to its range - a log parameter's log range
    const frames = Array.from({'length': 9}, (_unused, index) => frame({'f0': 40 * (index + 1), 'av': -39}));
    const features = frameFeatures(frames);
    assert.equal(features.length, 3);
    assert.equal(features[0].length, 13);
    assert.equal(features[0][0], 0);
    assert(Math.abs(features[1][0] - Math.log(5) / Math.log(15)) < 1e-12);
    assert(Math.abs(features[0][1] - 0.5) < 1e-12);
    assert.equal(frameFeatures(frames, 1).length, 9);

    // A log parameter below its low is its low
    assert.equal(frameFeatures([frame({'f0': 10})])[0][0], 0);
});


test('trainabreport.alignedDistance', () => {
    // The same frames - no distance; frames slower in time - aligned, only the duration's log ratio
    const features = frameFeatures(Array.from({'length': 40}, (_unused, index) => frame({'f0': 100 + 5 * index})));
    assert.equal(alignedDistance(features, features), 0);
    const slower = features.flatMap((feature) => [feature, feature]);
    assert(Math.abs(alignedDistance(features, slower) - Math.log(2)) < 1e-12);

    // Different frames - their mean distance along the alignment
    const higher = features.map((feature) => feature.map((value, index) => (index === 1 ? value + 0.1 : value)));
    assert(Math.abs(alignedDistance(features, higher) - 0.1) < 1e-12);

    // Empty sequences
    assert.equal(alignedDistance([], []), 0);
    assert.equal(alignedDistance(features, []), Infinity);
});


test('trainabreport.chooseTrials', () => {
    // Each voice's most different texts, in the voices' order
    const candidates = [
        {'voice': 'male', 'text': 'a', 'difference': 0.1}, {'voice': 'male', 'text': 'b', 'difference': 0.3},
        {'voice': 'female', 'text': 'a', 'difference': 0.5}, {'voice': 'male', 'text': 'c', 'difference': 0.2},
        {'voice': 'female', 'text': 'b', 'difference': 0.4}, {'voice': 'female', 'text': 'c', 'difference': 0.6}
    ];
    const chosen = (perVoice) => chooseTrials(candidates, perVoice).map((trial) => `${trial.voice} ${trial.text}`);
    assert.deepEqual(chosen(), ['male b', 'female c']);
    assert.deepEqual(chosen(2), ['male b', 'male c', 'female c', 'female a']);
});


test('trainabreport.makeQuestions', () => {
    // Each voice's sentence and passage, then the repeats
    assert.equal(questions.length, 8);
    const originals = questions.filter((question) => !question.repeatOf);
    assert.deepEqual(originals.map((question) => `${question.voice}: ${question.text}`).sort(), [
        'female: A passage.', 'female: Two.', 'male: A passage.', 'male: One.', 'sarah: A passage.', 'sarah: One.'
    ]);
    assert.deepEqual(originals.map((question) => question.id).sort(), ['r1-1', 'r1-2', 'r1-3', 'r1-4', 'r1-5', 'r1-6']);

    // A repeat is its original's voice and text, at least three trials later
    const repeats = questions.filter((question) => question.repeatOf);
    assert.equal(repeats.length, 2);
    for (const repeat of repeats) {
        const original = questions.find((question) => question.id === repeat.repeatOf);
        assert.equal(repeat.id, `${original.id}-repeat`);
        assert.deepEqual([repeat.voice, repeat.text], [original.voice, original.text]);
        assert(questions.indexOf(repeat) - questions.indexOf(original) >= 3);
    }

    // Each a blind A/B of the shipped voice and the trained one, ordered within its round, with its trial's difference
    for (const [index, question] of questions.entries()) {
        const trial = trials.find((item) => item.voice === question.voice && item.text === question.text);
        assert.equal(question.difference, trial.difference);
        assert.equal(question.order, 1000 + index);
        assert.equal(question.round, 1);
        assert.equal(question.type, 'ab');
        assert.equal(question.allowSame, true);
        assert.deepEqual(question.versions, [{'name': 'shipped', 'model': 'shipped'}, {'name': 'trained', 'model': 'trained'}]);
    }

    // The same round is the same trials (the trials unchanged); another round, its own ids and order
    assert.deepEqual(makeQuestions({trials, 'round': 1, 'repeats': 2}), questions);
    assert.deepEqual(trials.map((trial) => trial.voice), ['male', 'male', 'female', 'female', 'sarah', 'sarah']);
    const second = makeQuestions({'trials': trials.slice(0, 2), 'round': 2, 'repeats': 0});
    assert.deepEqual(second.map((question) => [question.id, question.order]), [['r2-1', 2000], ['r2-2', 2001]]);
});


test('trainabreport.wilson', () => {
    assert.deepEqual(wilson(0, 0), [0, 1]);
    const [low, high] = wilson(15, 20);
    assert(Math.abs(low - 0.531) < 0.001 && Math.abs(high - 0.888) < 0.001);
    assert.equal(wilson(0, 10)[0], 0);
    assert.equal(wilson(10, 10)[1], 1);
});


test('trainabreport.signTest', () => {
    assert.equal(signTest(0, 0), 1);
    assert.equal(signTest(5, 10), 1);
    assert(Math.abs(signTest(0, 10) - 2 / 1024) < 1e-12);
    assert(Math.abs(signTest(10, 10) - 2 / 1024) < 1e-12);
    assert(Math.abs(signTest(15, 20) - 0.04139) < 0.0001);
});


test('trainabreport.abReport', () => {
    // The trained voice preferred, not significantly - install; the repeats' agreement; each voice's counts
    const versions = ['trained', 'trained', 'shipped', null, 'trained', 'trained', 'trained', 'shipped'];
    const answers = answer(versions);
    const report = abReport(questions, answers);
    assert.equal(report.trials, 8);
    assert.equal(report.answered, 8);
    assert.deepEqual([report.trained, report.shipped, report.same], [5, 2, 1]);
    assert.equal(report.winRate, 5 / 7);
    assert.deepEqual(report.interval, wilson(5, 7));
    assert.equal(report.pValue, signTest(5, 7));
    const agreeing = questions.filter((question) => question.repeatOf)
        .filter((question) => answers[question.id].version === answers[question.repeatOf].version).length;
    assert.deepEqual(report.repeats, {'pairs': 2, 'agree': agreeing});
    const voiceCounts = Object.values(report.byVoice).reduce(
        (sum, counts) => [sum[0] + counts.trained, sum[1] + counts.shipped, sum[2] + counts.same], [0, 0, 0]
    );
    assert.deepEqual(voiceCounts, [5, 2, 1]);
    assert.equal(report.recommendation.action, 'install');
    assert.match(report.recommendation.text,
        /^Install the trained voice \(make train-install-voice\) - you preferred it in 5 of 7 decided trials \(not/);

    // Significantly preferred - install
    const manyTrials = Array.from({'length': 20}, (_unused, index) => ({'voice': `voice${index}`, 'text': 'One.', 'difference': 0}));
    const many = makeQuestions({'trials': manyTrials, 'round': 1, 'repeats': 0});
    const winning = Object.fromEntries(many.map((question, index) => [question.id, {'version': index < 16 ? 'trained' : 'shipped'}]));
    assert.match(abReport(many, winning).recommendation.text, /in 16 of 20 decided trials, significantly$/);

    // The shipped voice significantly preferred - keep it
    const losing = Object.fromEntries(many.map((question, index) => [question.id, {'version': index < 16 ? 'shipped' : 'trained'}]));
    assert.deepEqual(abReport(many, losing).recommendation, {
        'action': 'keep', 'text': 'Keep the shipped voice - you preferred it in 16 of 20 decided trials'
    });

    // The shipped voice preferred, not significantly - too close to call
    const close = abReport(questions, answer(['shipped', 'shipped', 'trained']));
    assert.equal(close.answered, 3);
    assert.equal(close.recommendation.action, 'unclear');
    assert.match(close.recommendation.text, /^Too close to call - the trained voice won 1 of 3 decided trials/);

    // Every answer Same, or at most a fifth a voice - no audible difference
    assert.deepEqual(abReport(questions, answer([null, null])).recommendation, {
        'action': 'same',
        'text': 'No audible difference - you heard the voices as the same in 2 of 2 trials, where they differ most: keep the shipped ' +
            'voice (installing the trained one changes nothing you can hear)'
    });
    assert.equal(abReport(questions, answer([null, null, null, null, 'trained'])).recommendation.action, 'same');
    assert.equal(abReport(questions, answer([null, null, null, 'trained', 'trained'])).recommendation.action, 'install');

    // No answers
    const none = abReport(questions, {});
    assert.deepEqual([none.answered, none.winRate, none.repeats.pairs], [0, null, 0]);
    assert.deepEqual(none.recommendation, {'action': 'unclear', 'text': 'No answers'});
});


test('trainabreport.formatReport', () => {
    // The counts, the win rate, the repeats, each voice, the panel, and the recommendation
    const report = {
        'trials': 30, 'answered': 30, 'trained': 15, 'shipped': 5, 'same': 10, 'winRate': 0.75, 'interval': [0.531, 0.888],
        'pValue': 0.0414, 'repeats': {'pairs': 6, 'agree': 2}, 'byVoice': {'male': {'trained': 2, 'shipped': 0, 'same': 1},
            'female': {'trained': 1, 'shipped': 1, 'same': 0}},
        'recommendation': {'action': 'install', 'text': 'Install it'}
    };
    const panel = {'student': {'mos': 4.131, 'wer': 1.36}, 'shipped': {'mos': 4.146, 'wer': 1.01}};
    assert.deepEqual(formatReport(report, panel), [
        'The blind A/B - 30 of 30 trials answered',
        '  preferred: the trained voice 15, the shipped voice 5, Same 10',
        '  the trained voice\'s win rate: 75% of 20 decided (95% interval 53%-89%; sign test p = 0.041)',
        '  the repeats: the same choice in 2 of 6 - the voices may be hard to tell apart',
        '  by voice (trained-shipped): male 2-0 (1 Same), female 1-1',
        '  the listener panel: UTMOS 4.131 (shipped 4.146), WER 1.36% (shipped 1.01%)',
        'Recommendation: Install it'
    ]);

    // Nothing decided, consistent repeats, no voices, no panel
    assert.deepEqual(formatReport({...report, 'trained': 0, 'shipped': 0, 'repeats': {'pairs': 2, 'agree': 2}, 'byVoice': {}}), [
        'The blind A/B - 30 of 30 trials answered',
        '  preferred: the trained voice 0, the shipped voice 0, Same 10',
        '  the repeats: the same choice in 2 of 2',
        'Recommendation: Install it'
    ]);
    assert(!formatReport({...report, 'repeats': {'pairs': 0, 'agree': 0}}).some((line) => line.includes('repeats')));
});
