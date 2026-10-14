import {lfPulse, lfTiming} from '../lib/glottal.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


// The voice quality range, pressed to breathy
const rdValues = [0.3, 0.5, 0.8, 1, 1.3, 1.7, 2.2, 2.7];


// Integrate a pulse over a period by the midpoint rule
function integrate(pulse, period, steps = 20000) {
    let sum = 0;
    for (let ix = 0; ix < steps; ix++) {
        sum += pulse((ix + 0.5) * period / steps);
    }
    return sum * period / steps;
}


test('lfTiming, modal voice', () => {
    const timing = lfTiming(1);
    assert.ok(Math.abs(timing.tp - 0.48436) < 1e-4);
    assert.ok(Math.abs(timing.te - 0.65001) < 1e-4);
    assert.ok(Math.abs(timing.ta - 0.038) < 1e-12);
});


test('lfTiming, ordering across Rd', () => {
    let previousTa = 0;
    for (const rd of rdValues) {
        const timing = lfTiming(rd);
        assert.ok(timing.tp > 0, `rd ${rd}`);
        assert.ok(timing.tp < timing.te, `rd ${rd}`);
        assert.ok(timing.te <= 0.98, `rd ${rd}`);
        // The return phase lengthens as the voice gets breathier
        assert.ok(timing.ta > previousTa, `rd ${rd}`);
        previousTa = timing.ta;
    }
});


test('lfTiming, clamped Rd', () => {
    assert.deepEqual(lfTiming(0.1), lfTiming(0.3));
    assert.deepEqual(lfTiming(-5), lfTiming(0.3));
    assert.deepEqual(lfTiming(3), lfTiming(2.7));
    assert.deepEqual(lfTiming(100), lfTiming(2.7));
});


test('lfPulse, closure timing and excitation peak', () => {
    const period = 0.01;
    for (const rd of rdValues) {
        const timing = lfTiming(rd);
        const pulse = lfPulse(period, rd);
        const te = timing.te * period;

        // The derivative starts at zero, crosses zero at peak flow, and returns to zero at the period's end
        assert.ok(Math.abs(pulse(0)) < 1e-12, `rd ${rd}`);
        assert.ok(Math.abs(pulse(timing.tp * period)) < 1e-9, `rd ${rd}`);
        assert.ok(Math.abs(pulse(period)) < 1e-9, `rd ${rd}`);

        // The excitation (negative peak) is -1 at closure, approached continuously from both sides
        assert.ok(Math.abs(pulse(te) + 1) < 1e-9, `rd ${rd}`);
        assert.ok(Math.abs(pulse(te - 1e-9) + 1) < 1e-4, `rd ${rd}`);
        assert.ok(Math.abs(pulse(te + 1e-9) + 1) < 1e-4, `rd ${rd}`);

        // The flow rises (positive derivative) before peak flow and falls after it
        assert.ok(pulse(0.5 * timing.tp * period) > 0, `rd ${rd}`);
        assert.ok(pulse(0.5 * (timing.tp + timing.te) * period) < 0, `rd ${rd}`);
        assert.ok(pulse(0.5 * (te + period)) < 0, `rd ${rd}`);
    }
});


test('lfPulse, zero net flow', () => {
    for (const period of [1 / 80, 1 / 120, 1 / 250]) {
        for (const rd of rdValues) {
            const pulse = lfPulse(period, rd);

            // The net flow (the derivative's integral) is zero - compared with the open phase's flow
            const te = lfTiming(rd).te * period;
            const openFlow = integrate(pulse, lfTiming(rd).tp * period);
            const netFlow = integrate(pulse, period);
            assert.ok(openFlow > 0, `period ${period}, rd ${rd}`);
            assert.ok(Math.abs(netFlow) < 1e-3 * openFlow, `period ${period}, rd ${rd}, net ${netFlow}, open ${openFlow}`);
            assert.ok(te < period);
        }
    }
});


test('lfPulse, return phase floor', () => {
    // A period so short that the return phase hits its 1 microsecond floor - the growth constant goes negative
    const period = 3e-6;
    const timing = lfTiming(0.3);
    const pulse = lfPulse(period, 0.3);
    assert.ok(Math.abs(pulse(timing.te * period) + 1) < 1e-6);
    assert.ok(Math.abs(integrate(pulse, period)) < 1e-6 * integrate(pulse, timing.tp * period));
    assert.ok(pulse(0.5 * timing.tp * period) > 0);
});


test('lfPulse, period scaling', () => {
    // The pulse's shape scales with the period
    const short = lfPulse(0.005, 1);
    const long = lfPulse(0.01, 1);
    for (const fraction of [0.1, 0.3, 0.6, 0.7, 0.9]) {
        assert.ok(Math.abs(short(fraction * 0.005) - long(fraction * 0.01)) < 1e-6, `fraction ${fraction}`);
    }
});
