// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

import {applyStyle, baseVoice, resolveVoice, validateVoice, voiceMetadata, voiceParameters, voices} from '../lib/voices.js';
import {schwa, vowels} from '../lib/phonemes.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


test('voices.voiceParameters', () => {
    const groups = new Set(['pitch', 'source', 'tract', 'articulation', 'timing']);
    for (const [key, parameter] of Object.entries(voiceParameters)) {
        assert.deepEqual(Object.keys(parameter).sort(), ['default', 'doc', 'group', 'range', 'units'], key);
        assert(groups.has(parameter.group), key);
        assert(parameter.doc.length > 10, key);
        assert(parameter.units.length > 0, key);
        if (parameter.range === null) {
            assert.equal(typeof parameter.default, 'boolean', key);
        } else {
            const [low, high] = parameter.range;
            assert(low < high, key);
        }
        assert(!voiceMetadata.includes(key), key);
    }
});


test('voices.baseVoice', () => {
    // Every parameter, at its default, within its typical range
    assert.deepEqual(Object.keys(baseVoice), Object.keys(voiceParameters));
    for (const [key, parameter] of Object.entries(voiceParameters)) {
        assert.deepEqual(baseVoice[key], parameter.default, key);
    }
    assert.deepEqual(validateVoice(baseVoice), []);
});


test('voices.voices', () => {
    for (const [name, spec] of Object.entries(voices)) {
        assert.equal(spec.name, name);
        assert.equal(typeof spec.label, 'string');
        assert.equal(typeof spec.description, 'string');
        assert.deepEqual(validateVoice(spec), [], name);
    }

    // Both voices define the same parameters
    assert.deepEqual(Object.keys(voices.female).sort(), Object.keys(voices.male).sort());

    // The female voice is higher pitched, with higher formants and the women's vowel space
    assert(voices.female.f0 > voices.male.f0 * 1.4);
    assert(voices.female.f4 > voices.male.f4);
    for (const scale of voices.female.formantScale) {
        assert(scale > 1);
    }
    assert.deepEqual(voices.male.formantScale, [1, 1, 1]);
    assert.equal(voices.male.vowelBlend, 0);
    assert.equal(voices.female.vowelBlend, 1);
});


test('voices.resolveVoice', () => {
    // A name - every parameter, the spec over the root voice
    const male = resolveVoice('male');
    for (const key of Object.keys(voiceParameters)) {
        assert(key in male, key);
    }
    assert.equal(male.name, 'male');
    assert.equal(male.commaPause, baseVoice.commaPause);
    const female = resolveVoice('female');
    assert.equal(female.f0, voices.female.f0);
    assert.deepEqual(female.higherFormants, voices.female.higherFormants);

    // A spec object - the same as its name
    assert.deepEqual(resolveVoice(voices.female), female);

    // A base chain - a spec object on a named spec on a spec object
    const alto = {'name': 'alto', 'base': 'female', 'f0': 160, 'tractScale': 0.96};
    const narrator = resolveVoice({'name': 'narrator', 'base': alto, 'rate': 1.05});
    assert.equal(narrator.name, 'narrator');
    assert.equal(narrator.f0, 160);
    assert.equal(narrator.tractScale, 0.96);
    assert.equal(narrator.rate, 1.05);
    assert.equal(narrator.breathDb, voices.female.breathDb);
    assert.equal(narrator.glottalAttack, true);

    // No base - the root voice
    assert.equal(resolveVoice({'f0': 120}).rd, baseVoice.rd);
    assert.equal(resolveVoice({'f0': 120, 'base': null}).rd, baseVoice.rd);

    // The overrides, then the rate option and pitchScale
    const overridden = resolveVoice('male', {'overrides': {'rd': 1.5, 'pitchScale': 1.5}, 'rate': 2});
    assert.equal(overridden.rd, 1.5);
    assert.equal(overridden.f0, voices.male.f0 * 1.5);
    assert.equal(overridden.rate, voices.male.rate * 2);

    // A style, after the overrides
    const styled = resolveVoice('male', {'overrides': {'breathDb': -30}, 'style': {'add': {'breathDb': 10}}});
    assert.equal(styled.breathDb, -20);

    // Errors
    assert.throws(() => resolveVoice('nobody'), /Unknown voice "nobody"/);
    assert.throws(() => resolveVoice({'base': 'nobody'}), /Unknown voice "nobody"/);
    const first = {'name': 'first'};
    const second = {'name': 'second', 'base': first};
    first.base = second;
    assert.throws(() => resolveVoice(second), /Voice base cycle at "second"/);

    // The specs are unchanged
    assert.equal(voices.male.f0, 101);
    assert.equal(alto.base, 'female');
});


test('voices.resolveVoice, vowel blend', () => {
    // The end points are the tables
    const male = resolveVoice('male');
    assert.equal(male.vowelTargets.AA, vowels.AA.men);
    assert.equal(male.schwaTargets, schwa.men);
    const female = resolveVoice('female');
    assert.equal(female.vowelTargets.AY, vowels.AY.women);
    assert.equal(female.schwaTargets, schwa.women);
    assert.deepEqual(Object.keys(female.vowelTargets).sort(), Object.keys(vowels).sort());

    // Between, a blend of every target
    const blended = resolveVoice('male', {'overrides': {'vowelBlend': 0.25}});
    const [[menF1, menF2], [menEndF1]] = vowels.AY.men;
    const [[womenF1, womenF2], [womenEndF1]] = vowels.AY.women;
    assert.equal(blended.vowelTargets.AY.length, 2);
    assert(Math.abs(blended.vowelTargets.AY[0][0] - (menF1 + 0.25 * (womenF1 - menF1))) < 1e-9);
    assert(Math.abs(blended.vowelTargets.AY[0][1] - (menF2 + 0.25 * (womenF2 - menF2))) < 1e-9);
    assert(Math.abs(blended.vowelTargets.AY[1][0] - (menEndF1 + 0.25 * (womenEndF1 - menEndF1))) < 1e-9);
    assert(Math.abs(blended.schwaTargets[2] - (schwa.men[2] + 0.25 * (schwa.women[2] - schwa.men[2]))) < 1e-9);
});


test('voices.applyStyle', () => {
    const params = resolveVoice('male');

    // Full intensity (the default) - scale, add, and set
    const full = applyStyle(params, {
        'scale': {'f0': 1.2, 'accent': 2}, 'add': {'breathDb': 12}, 'set': {'rd': 1.5, 'glottalAttack': false}
    });
    assert(Math.abs(full.f0 - params.f0 * 1.2) < 1e-9);
    assert(Math.abs(full.accent - params.accent * 2) < 1e-12);
    assert.equal(full.breathDb, params.breathDb + 12);
    assert.equal(full.rd, 1.5);
    assert.equal(full.glottalAttack, false);

    // Half intensity - geometric scaling, linear adding and setting, booleans switched at one half
    const half = applyStyle(params, {
        'intensity': 0.5, 'scale': {'accent': 4}, 'add': {'breathDb': 12}, 'set': {'rd': 1.5, 'glottalAttack': false}
    });
    assert(Math.abs(half.accent - params.accent * 2) < 1e-12);
    assert.equal(half.breathDb, params.breathDb + 6);
    assert(Math.abs(half.rd - (params.rd + 1.5) / 2) < 1e-12);
    assert.equal(half.glottalAttack, false);
    const less = applyStyle(params, {'intensity': 0.4, 'set': {'glottalAttack': false}});
    assert.equal(less.glottalAttack, true);

    // Zero intensity changes nothing; intensity is limited to 0-1
    assert.deepEqual(applyStyle(params, {'intensity': 0, 'scale': {'f0': 1.2}, 'add': {'breathDb': 12}, 'set': {'rd': 2}}), params);
    assert.deepEqual(applyStyle(params, {'intensity': -1, 'add': {'breathDb': 12}}), params);
    assert.equal(applyStyle(params, {'intensity': 3, 'add': {'breathDb': 12}}).breathDb, params.breathDb + 12);

    // Array parameters - a number operand for every element, or a like array
    const arrays = applyStyle(params, {'scale': {'higherFormants': 2, 'formantScale': [1, 2, 3]}, 'add': {'fricVoiceBar': [10, 20]}});
    assert.deepEqual(arrays.higherFormants, params.higherFormants.map(([frequency, bandwidth]) => [frequency * 2, bandwidth * 2]));
    assert.deepEqual(arrays.formantScale, [1, 2, 3]);
    assert.deepEqual(arrays.fricVoiceBar, [310, 520]);

    // The order - scale, then add, then set
    const ordered = applyStyle(params, {'set': {'commaPause': 100}, 'add': {'commaPause': 50}, 'scale': {'commaPause': 2}});
    assert.equal(ordered.commaPause, 100);
    assert.equal(applyStyle(params, {'add': {'commaPause': 50}, 'scale': {'commaPause': 2}}).commaPause, params.commaPause * 2 + 50);

    // Styles in order
    const composed = applyStyle(params, [{'add': {'commaPause': 50}}, {'scale': {'commaPause': 2}}]);
    assert.equal(composed.commaPause, (params.commaPause + 50) * 2);

    // The parameters are unchanged
    assert.deepEqual(params, resolveVoice('male'));

    // Errors
    assert.throws(() => applyStyle(params, {'scale': {'loudness': 2}}), /Unknown style parameter "loudness"/);
    assert.throws(() => applyStyle(params, {'scale': {'glottalAttack': 2}}), /Invalid style scale of "glottalAttack"/);
    assert.throws(() => applyStyle(params, {'add': {'f0': 'high'}}), /Invalid style add of "f0"/);
    assert.throws(() => applyStyle(params, {'set': {'glottalAttack': 1}}), /Invalid style set of "glottalAttack"/);
});


test('voices.validateVoice', () => {
    assert.deepEqual(validateVoice({'name': 'tenor', 'label': 'Tenor', 'description': 'A tenor', 'base': 'male', 'f0': 130}), []);
    assert.deepEqual(validateVoice({'base': voices.male}), []);
    assert.deepEqual(validateVoice({
        'name': 1, 'base': 2, 'loudness': 3, 'f0': 'high', 'formantScale': 1, 'rd': [1], 'glottalAttack': 1,
        'accent': 2, 'higherFormants': [[6500, 500], [20000, 500]], 'scaleNasalF1': true
    }), [
        'Invalid name',
        'Invalid base',
        'Unknown parameter "loudness"',
        'Invalid type of "f0"',
        'Invalid type of "formantScale"',
        'Invalid type of "rd"',
        'Invalid type of "glottalAttack"',
        '"accent" outside its typical range [0.1, 0.7]',
        '"higherFormants" outside its typical range [300, 13000]'
    ]);
    assert.deepEqual(validateVoice({'base': null}), []);
});
