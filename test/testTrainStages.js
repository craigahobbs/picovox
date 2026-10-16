// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

import {strict as assert} from 'node:assert';
import {readFileSync} from 'node:fs';
import test from 'node:test';


// The Makefile's make train stages - {stage: {needs, after, recipe}} - from its $(TRAIN_DIR)/<stage>.done rules: the
// stamps a stage needs (normal prerequisites), the stamps it runs after (order-only), and its recipe lines
function makefileStages(text) {
    const stages = {};
    const joined = text.replace(/\\\n/g, ' ');
    for (const match of joined.matchAll(/^\$\(TRAIN_DIR\)\/([\w-]+)\.done:([^\n]*)\n((?:\t[^\n]*\n)*)/gm)) {
        const [, stage, prerequisites, recipe] = match;
        const [normal, orderOnly = ''] = prerequisites.split('|');
        const stamps = (part) => [...part.matchAll(/\$\(TRAIN_DIR\)\/([\w-]+)\.done/g)].map((stamp) => stamp[1]);
        const lines = recipe.trim().split('\n').map((line) => line.trim());
        stages[stage] = {'needs': stamps(normal), 'after': stamps(orderOnly), 'recipe': lines};
    }
    return stages;
}


test('trainStages', () => {
    // data/nn/train-stages.json (the training dashboard's stage graph) has the Makefile's stages, dependencies, and
    // ordering, each stage a known kind with an expected time
    const makefile = readFileSync(new URL('../Makefile', import.meta.url), 'utf8');
    const manifest = JSON.parse(readFileSync(new URL('../data/nn/train-stages.json', import.meta.url), 'utf8'));
    const stages = makefileStages(makefile);
    assert.deepEqual(manifest.stages.map((stage) => stage.stage).sort(), Object.keys(stages).sort());
    for (const stage of manifest.stages) {
        assert.deepEqual([...stage.needs].sort(), [...stages[stage.stage].needs].sort(), stage.stage);
        assert.deepEqual([...stage.after || []].sort(), [...stages[stage.stage].after].sort(), stage.stage);
        assert(['cpu', 'gpu', 'network'].includes(stage.on), stage.stage);
        assert(stage.minutes > 0, stage.stage);
        assert.equal(typeof stage.group, 'string');
        assert.equal(typeof stage.title, 'string');
    }

    // The manifest's runs are the training runs its stages write
    assert.deepEqual(manifest.stages.filter((stage) => stage.run).map((stage) => stage.run),
        ['teacher-a', 'teacher-b', 'distill-student', 'distill-student-polish']);

    // make train's targets reach every stage
    const [, trainTargets] = makefile.match(/^train: ([^\n]*)$/m);
    const reached = new Set();
    const visit = (stage) => {
        reached.add(stage);
        stages[stage].needs.forEach(visit);
    };
    [...trainTargets.matchAll(/\$\(TRAIN_DIR\)\/([\w-]+)\.done/g)].forEach((match) => visit(match[1]));
    assert.deepEqual([...reached].sort(), Object.keys(stages).sort());

    // Each stage's recipe records its start, wraps every command (its output in its log), records its end, then stamps
    for (const [stage, {recipe}] of Object.entries(stages)) {
        assert.equal(recipe[0], '@$(STAGE) start', stage);
        assert.equal(recipe[recipe.length - 2], '@$(STAGE) end', stage);
        assert.match(recipe[recipe.length - 1], /touch \$@$/, stage);
        for (const line of recipe.slice(1, -2)) {
            assert.match(line, /^\$\(STAGE\) (?:--log \S+ )?-- /, stage);
        }
    }
});
