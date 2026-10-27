// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// make train-check's blind A/B (tools/trainab.js) - its trials, and the report of the answers with a recommendation. A
// trial is a voice lab question (tools/voicelab/voicelab.js): the shipped voice against make train's trained voice (both
// with the shipped G2P), played in a shuffled order, chosen or "Same". An answer's version is the chosen version's name
// (shipped or trained), null for Same. The trials are where the two voices differ most - each voice's texts ranked by
// the distance of their parameter frames (aligned) - so an audible difference is heard if there is one.

import {PARAMETERS} from '../lib/params.js';


// A seeded random number generator (mulberry32) - numbers in [0, 1), the same sequence for a seed
export function seededRandom(seed) {
    let state = seed >>> 0;
    return () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let mixed = Math.imul(state ^ (state >>> 15), state | 1);
        mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
        return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
    };
}


// Shuffle an array in place (Fisher-Yates) with a random number generator
function shuffle(items, random) {
    for (let position = items.length - 1; position > 0; position--) {
        const other = Math.floor(random() * (position + 1));
        [items[position], items[other]] = [items[other], items[position]];
    }
    return items;
}


// The audible parameters a frame's distance compares - the source (pitch, voicing, pulse shape, tilt, the noises) and the
// tract's lower formants and bandwidths - each scaled to its range (a log parameter's log range)
const FEATURES = ['f0', 'av', 'rd', 'shelf', 'ah', 'af', 'f1', 'f2', 'f3', 'f4', 'b1', 'b2', 'b3'].map((name) => {
    const index = PARAMETERS.findIndex((parameter) => parameter.name === name);
    const {scale, low, high} = PARAMETERS[index];
    return scale === 'log' ?
        {index, 'transform': (value) => Math.log(Math.max(value, low) / low) / Math.log(high / low)} :
        {index, 'transform': (value) => (value - low) / (high - low)};
});


// A text's parameter frames (200 a second) as feature vectors (FEATURES), one every step frames
export function frameFeatures(frames, step = 4) {
    const features = [];
    for (let index = 0; index < frames.length; index += step) {
        features.push(FEATURES.map(({index: parameter, transform}) => transform(frames[index][parameter])));
    }
    return features;
}


// The distance of two feature sequences - the mean Euclidean distance along their dynamic time warping alignment (the
// voices' timings differ), plus the log ratio of their durations
export function alignedDistance(first, second) {
    if (first.length === 0 || second.length === 0) {
        return first.length === second.length ? 0 : Infinity;
    }
    const distance = (one, two) => Math.sqrt(one.reduce((sum, value, index) => sum + (value - two[index]) ** 2, 0));
    let previous = new Float64Array(second.length + 1).fill(Infinity);
    let previousSteps = new Float64Array(second.length + 1);
    previous[0] = 0;
    for (let row = 1; row <= first.length; row++) {
        const current = new Float64Array(second.length + 1).fill(Infinity);
        const currentSteps = new Float64Array(second.length + 1);
        for (let column = 1; column <= second.length; column++) {
            const options = [[previous[column - 1], previousSteps[column - 1]], [previous[column], previousSteps[column]],
                [current[column - 1], currentSteps[column - 1]]];
            const [best, steps] = options.reduce((least, option) => (option[0] < least[0] ? option : least));
            current[column] = best + distance(first[row - 1], second[column - 1]);
            currentSteps[column] = steps + 1;
        }
        previous = current;
        previousSteps = currentSteps;
    }
    return previous[second.length] / previousSteps[second.length] + Math.abs(Math.log(first.length / second.length));
}


// The trials - each voice's texts most different between the two voices (candidates: [{voice, text, difference}]), the
// perVoice of each voice's, in the candidates' voice order
export function chooseTrials(candidates, perVoice = 2) {
    const voices = [...new Set(candidates.map((candidate) => candidate.voice))];
    return voices.flatMap((voice) => candidates.filter((candidate) => candidate.voice === voice)
        .sort((one, two) => two.difference - one.difference).slice(0, perVoice));
}


// The trials of a round ([{voice, text, difference}] - chooseTrials's), in a shuffled order (seeded by the round), and
// repeats of the first trials (repeatOf), each placed at least three trials after its original, for the listener's
// consistency
export function makeQuestions({trials, round, repeats = 6}) {
    const random = seededRandom(round);
    const questions = shuffle([...trials], random).map((trial, index) => ({...trial, 'id': `r${round}-${index + 1}`}));
    for (const original of questions.slice(0, repeats)) {
        const after = questions.findIndex((question) => question.id === original.id) + 3;
        const position = after + Math.floor(random() * (questions.length - after + 1));
        questions.splice(position, 0, {...original, 'id': `${original.id}-repeat`, 'repeatOf': original.id});
    }
    return questions.map((question, index) => ({
        'id': question.id, 'order': round * 1000 + index, round, 'type': 'ab',
        'prompt': 'Which sounds better - more natural, and clearer?',
        'hint': 'Play each as often as you like, then choose - Same if you can\'t tell them apart.',
        'text': question.text, 'voice': question.voice, 'difference': question.difference, 'allowSame': true,
        'versions': [{'name': 'shipped', 'model': 'shipped'}, {'name': 'trained', 'model': 'trained'}],
        ...(question.repeatOf ? {'repeatOf': question.repeatOf} : {})
    }));
}


// The Wilson score interval of a proportion (95%) - [low, high]
export function wilson(successes, total, zScore = 1.96) {
    if (total === 0) {
        return [0, 1];
    }
    const rate = successes / total;
    const denominator = 1 + zScore * zScore / total;
    const center = (rate + zScore * zScore / (2 * total)) / denominator;
    const half = zScore * Math.sqrt(rate * (1 - rate) / total + zScore * zScore / (4 * total * total)) / denominator;
    return [Math.max(0, center - half), Math.min(1, center + half)];
}


// The sign test's two-sided p-value - successes of total against an even split (the exact binomial)
export function signTest(successes, total) {
    let probability = 0.5 ** total;
    const terms = [probability];
    for (let count = 1; count <= total; count++) {
        probability *= (total - count + 1) / count;
        terms.push(probability);
    }
    const extreme = Math.min(successes, total - successes);
    const tail = terms.slice(0, extreme + 1).reduce((sum, term) => sum + term, 0);
    return Math.min(1, 2 * tail);
}


// The report of the answers ({question id: answer}) - the counts, the trained voice's win rate over the decided trials
// with its interval and sign test, the repeats' agreement, each voice's counts, and the recommendation
export function abReport(questions, answers) {
    const count = (items, version) => items.filter((answer) => answer.version === version).length;
    const answered = questions.filter((question) => question.id in answers);
    const chosen = answered.map((question) => answers[question.id]);
    const trained = count(chosen, 'trained');
    const shipped = count(chosen, 'shipped');
    const decided = trained + shipped;
    const pValue = signTest(trained, decided);

    // The repeats - whether each repeat's choice is its original's
    const pairs = answered.filter((question) => question.repeatOf && question.repeatOf in answers);
    const agree = pairs.filter((question) => answers[question.id].version === answers[question.repeatOf].version).length;

    // Each voice's counts
    const byVoice = {};
    for (const question of answered) {
        byVoice[question.voice] ??= {'trained': 0, 'shipped': 0, 'same': 0};
        byVoice[question.voice][answers[question.id].version ?? 'same'] += 1;
    }

    // The recommendation - no audible difference when at most a fifth of the answers chose a voice (keep the shipped
    // voice - installing changes nothing heard), keep the shipped voice when it's significantly preferred, install the
    // trained voice when it's preferred at least as often, else too close to call
    let recommendation;
    if (answered.length === 0) {
        recommendation = {'action': 'unclear', 'text': 'No answers'};
    } else if (decided * 5 <= answered.length) {
        recommendation = {
            'action': 'same',
            'text': `No audible difference - you heard the voices as the same in ${answered.length - decided} of ${answered.length} ` +
                'trials, where they differ most: keep the shipped voice (installing the trained one changes nothing you can hear)'
        };
    } else if (shipped > trained && pValue < 0.05) {
        recommendation = {'action': 'keep', 'text': `Keep the shipped voice - you preferred it in ${shipped} of ${decided} decided trials`};
    } else if (trained >= shipped) {
        const significance = pValue < 0.05 ? ', significantly' : ' (not significantly better, but no worse)';
        recommendation = {
            'action': 'install',
            'text': `Install the trained voice (make train-install-voice) - you preferred it in ${trained} of ${decided} decided ` +
                `trials${significance}`
        };
    } else {
        recommendation = {
            'action': 'unclear',
            'text': `Too close to call - the trained voice won ${trained} of ${decided} decided trials: another round (make ` +
                'train-check), or keep the shipped voice'
        };
    }
    return {
        'trials': questions.length, 'answered': answered.length, trained, shipped, 'same': answered.length - decided,
        'winRate': decided === 0 ? null : trained / decided, 'interval': wilson(trained, decided), pValue,
        'repeats': {'pairs': pairs.length, agree}, byVoice, recommendation
    };
}


// The report as text lines - with the listener panel's scores (distill pick's summary rows: {student, shipped}) if given
export function formatReport(report, panel = null) {
    const percent = (value) => `${Math.round(value * 100)}%`;
    const lines = [`The blind A/B - ${report.answered} of ${report.trials} trials answered`];
    const decided = report.trained + report.shipped;
    lines.push(`  preferred: the trained voice ${report.trained}, the shipped voice ${report.shipped}, Same ${report.same}`);
    if (decided !== 0) {
        lines.push(`  the trained voice's win rate: ${percent(report.winRate)} of ${decided} decided (95% interval ` +
            `${percent(report.interval[0])}-${percent(report.interval[1])}; sign test p = ${report.pValue.toFixed(3)})`);
    }
    if (report.repeats.pairs !== 0) {
        const hard = report.repeats.agree * 2 < report.repeats.pairs ? ' - the voices may be hard to tell apart' : '';
        lines.push(`  the repeats: the same choice in ${report.repeats.agree} of ${report.repeats.pairs}${hard}`);
    }
    const voices = Object.entries(report.byVoice).map(
        ([voice, counts]) => `${voice} ${counts.trained}-${counts.shipped}${counts.same ? ` (${counts.same} Same)` : ''}`
    );
    if (voices.length !== 0) {
        lines.push(`  by voice (trained-shipped): ${voices.join(', ')}`);
    }
    if (panel !== null) {
        lines.push(`  the listener panel: UTMOS ${panel.student.mos.toFixed(3)} (shipped ${panel.shipped.mos.toFixed(3)}), ` +
            `WER ${panel.student.wer.toFixed(2)}% (shipped ${panel.shipped.wer.toFixed(2)}%)`);
    }
    lines.push(`Recommendation: ${report.recommendation.text}`);
    return lines;
}
