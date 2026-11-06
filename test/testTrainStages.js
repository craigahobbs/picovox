// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

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
    // train/data/train-stages.json (the training dashboard's stage graph) has the Makefile's stages, dependencies, and
    // ordering, each stage a known kind with an expected time
    const makefile = readFileSync(new URL('../Makefile', import.meta.url), 'utf8');
    const manifest = JSON.parse(readFileSync(new URL('../train/data/train-stages.json', import.meta.url), 'utf8'));
    const stages = makefileStages(makefile);
    assert.deepEqual(manifest.stages.map((stage) => stage.stage).sort(), Object.keys(stages).sort());
    for (const stage of manifest.stages) {
        assert.deepEqual([...stage.needs].sort(), [...stages[stage.stage].needs].sort(), stage.stage);
        assert.deepEqual([...stage.after || []].sort(), [...stages[stage.stage].after].sort(), stage.stage);
        assert(['cpu', 'gpu'].includes(stage.on), stage.stage);
        assert(stage.minutes > 0, stage.stage);
        assert.equal(typeof stage.group, 'string');
        assert.equal(typeof stage.title, 'string');
    }

    // A stage's progress counts its log's finished parts - the context network's trainings: its teachers and itself,
    // and each fold's - in a log the stage starts anew
    for (const stage of manifest.stages.filter((item) => item.progress && !item.progress.phases)) {
        assert.equal(typeof stage.progress.log, 'string', stage.stage);
        assert.equal(stages[stage.stage].recipe[1], `$(STAGE) -- rm -f $(TRAIN_DIR)/${stage.progress.log}`, stage.stage);
        assert(new RegExp(stage.progress.pattern).test('train-context: step 10000/10000 (quantized) loss 0.0180'), stage.stage);
        assert(stage.progress.total > 0, stage.stage);
    }
    const trainRecipe = JSON.parse(readFileSync(new URL('../train/data/train.json', import.meta.url), 'utf8'));
    const contextStage = manifest.stages.find((stage) => stage.stage === 'context-network');
    assert.equal(contextStage.progress.total, (trainRecipe.context.network.teachers + 1) * (trainRecipe.context.gate.folds + 1));

    // The training speech's stages - their progress the phases of their commands' log lines (the Python sources' log
    // lines, rendered), their first run's time longer, and the first run known by the corpus stage's generate line
    const source = (name) => readFileSync(new URL(`../train/src/picovox_train/${name}`, import.meta.url), 'utf8');
    const ttsSource = source('tts.py');
    const measureSource = source('measure.py');
    const logLines = {
        'generate': [ttsSource, 'f\'tts generate: {len(clips)} clips in {directory}, {len(missing)} to generate\'',
            'tts generate: 29475 clips in /downloads/tts/phonon, 120 to generate'],
        'generateStep': [ttsSource,
            'f\'  {number}/{len(missing)} clips, {total / 24000 / 3600:.2f} hours, {time.time() - start:.0f} s\'',
            '  500/29475 clips, 1.07 hours, 52 s'],
        'align': [ttsSource, 'f\'tts align: {len(clips)} clips, {len(todo)} to align\'', 'tts align: 29475 clips, 120 to align'],
        'alignStep': [ttsSource, 'f\'  {offset + len(group)}/{len(todo)} clips, {time.time() - start:.0f} s\'', '  256/29475 clips, 15 s'],
        'measure': [measureSource, 'f\'measure: {len(clips)} clips, {sum(not task.get("control") for _, task in plan)} measured tokens, \'',
            'measure: 28000 clips, 120 measured tokens, 900 control tokens'],
        'measureCurrent': [measureSource, 'f\'measure: {out} is current\'', 'measure: /downloads/tts/phonon/readings is current'],
        'build': [ttsSource, 'f\'tts build: {len(clips)} clips to build\'', 'tts build: 120 clips to build'],
        'buildCurrent': [ttsSource, 'f\'tts build: {cached} is current\'', 'tts build: /downloads/tts/phonon/build-0123 is current'],
        'buildStep': [ttsSource,
            'f\'  {count}/{len(tasks)} clips, {frame_offset / 100 / 3600:.1f} h, {time.time() - start_time:.0f} s\'',
            '  2000/29475 clips, 3.2 h, 61 s'],
        'measureStep': [measureSource,
            'f\'  ctc {min(len(voted), offset + args.batch)}/{len(voted)} tokens, {time.time() - start_time:.0f} s\'',
            '  ctc 64/120 tokens, 9 s']
    };
    for (const [name, [text, format]] of Object.entries(logLines)) {
        assert(text.includes(format), name);
    }
    const phased = (name, lines) => {
        const stage = manifest.stages.find((item) => item.stage === name);
        assert(stage.firstMinutes > stage.minutes, name);
        return stage.progress.phases.map((phase, index) => {
            const [header, step] = lines[index];
            assert.equal(logLines[header][2].match(new RegExp(phase.header))[1], '120', `${name} ${phase.name}`);
            assert.deepEqual(logLines[step][2].match(new RegExp(phase.step)).slice(1).length, 3, `${name} ${phase.name}`);
            assert(phase.secondsEach > 0, `${name} ${phase.name}`);
            return phase.name;
        });
    };
    assert.deepEqual(phased('tts-corpus', [['generate', 'generateStep'], ['align', 'alignStep']]), ['generate', 'align']);
    assert.deepEqual(phased('tts-readings', [['measure', 'measureStep'], ['align', 'alignStep']]), ['measure', 'align']);
    assert.deepEqual(phased('data-tts', [['build', 'buildStep']]), ['build']);
    const build = manifest.stages.find((item) => item.stage === 'data-tts');
    assert.equal(logLines.buildCurrent[2].match(new RegExp(build.progress.phases[0].header))[1], undefined);
    const readings = manifest.stages.find((item) => item.stage === 'tts-readings');
    assert.equal(logLines.measureCurrent[2].match(new RegExp(readings.progress.phases[0].header))[1], undefined);
    assert.deepEqual(stages['tts-corpus'].recipe.filter((line) => line.includes('picovox_train.')).
        map((line) => line.match(/picovox_train\.(\w+ \w+)/)[1]), ['tts generate', 'tts align']);
    assert.equal(manifest.firstRun.stage, 'tts-corpus');
    assert.equal(logLines.generate[2].match(new RegExp(manifest.firstRun.pattern))[1], '120');

    // The manifest's runs are the training runs its stages write
    assert.deepEqual(manifest.stages.filter((stage) => stage.run).map((stage) => stage.run),
        ['voice-a', 'voice-b']);

    // The listening samples stages each render their stage's model (tools/trainsamples.js --stage), the reference the
    // shipped model
    const samplesStages = manifest.stages.filter((stage) => stage.samples);
    assert.deepEqual(samplesStages.map((stage) => stage.samples), ['reference', 'voice-a', 'voice-b']);
    for (const stage of samplesStages) {
        assert(stages[stage.stage].recipe.some((line) => line.includes(`--stage ${stage.samples} `)), stage.stage);
        assert(stage.samples === 'reference' || stage.needs.includes(stage.samples), stage.stage);
    }

    // The manifest's targets are the Makefile's - make train (both models), make train-voice and train-g2p (each model) -
    // each recording its run, making its stages, then reporting
    const joinedMakefile = makefile.replace(/\\\n/g, ' ');
    const stampsOf = (text) => [...text.matchAll(/\$\(TRAIN_DIR\)\/([\w-]+)\.done/g)].map((match) => match[1]);
    const variable = (name) => joinedMakefile.match(new RegExp(`^${name} := ([^\\n]*)$`, 'm'))[1];
    const targetStamps = {
        'train': '$(TRAIN_VOICE) $(TRAIN_G2P)',
        'train-voice': '$(TRAIN_VOICE)',
        'train-g2p': '$(TRAIN_G2P)'
    };
    const expand = (stamps) => stamps.replace(/\$\((TRAIN_[A-Z0-9_]+)\)/g, (match, name) => variable(name));
    const expected = Object.entries(targetStamps).map(([target, stamps]) => [target, stampsOf(expand(stamps))]);
    assert.deepEqual(manifest.targets, Object.fromEntries(expected));
    for (const [target, stamps] of Object.entries(targetStamps)) {
        const [, body] = joinedMakefile.match(new RegExp(`^${target}:\\n((?:\\t[^\\n]*\\n)*)`, 'm'));
        assert(body.startsWith(`\t$(call TRAIN_TARGET,${target})\n`), target);
        const [make, report] = body.trimEnd().split('\n').slice(-2);
        assert(make.endsWith(` --no-print-directory ${stamps}`) || make.endsWith(` BESIDE= ${stamps}`), target);
        assert.equal(report, '\t@$(TRAIN_REPORT)', target);
    }

    // make train reaches every stage
    const trainTargets = expand(targetStamps.train);
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
