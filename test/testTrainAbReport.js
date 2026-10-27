// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {abReport, formatReport, makeQuestions, seededRandom, signTest, wilson} from '../tools/trainabreport.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


// A round's trials - three voices, each a sentence and a passage, with two repeats
const questions = makeQuestions({
    'voices': ['male', 'female', 'sarah'], 'sentences': ['One.', 'Two.'], 'passages': ['A passage.'], 'round': 1, 'repeats': 2
});


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

    // Each a blind A/B of the shipped voice and the trained one, ordered within its round
    for (const [index, question] of questions.entries()) {
        assert.equal(question.order, 1000 + index);
        assert.equal(question.round, 1);
        assert.equal(question.type, 'ab');
        assert.equal(question.allowSame, true);
        assert.deepEqual(question.versions, [{'name': 'shipped', 'model': 'shipped'}, {'name': 'trained', 'model': 'trained'}]);
    }

    // The same round is the same trials; another round, its own ids and order
    assert.deepEqual(makeQuestions({'voices': ['male', 'female', 'sarah'], 'sentences': ['One.', 'Two.'], 'passages': ['A passage.'],
        'round': 1, 'repeats': 2}), questions);
    const second = makeQuestions({'voices': ['male'], 'sentences': ['One.'], 'passages': ['A passage.'], 'round': 2, 'repeats': 0});
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
    const many = makeQuestions({'voices': Array.from({'length': 10}, (_unused, index) => `voice${index}`), 'sentences': ['One.'],
        'passages': ['Two.'], 'round': 1, 'repeats': 0});
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

    // Every answer Same, or none
    assert.deepEqual(abReport(questions, answer([null, null])).recommendation,
        {'action': 'unclear', 'text': 'No decided trials - every answer was Same'});
    const none = abReport(questions, {});
    assert.deepEqual([none.answered, none.winRate, none.repeats.pairs], [0, null, 0]);
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
