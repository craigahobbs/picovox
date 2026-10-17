// The Liljencrants-Fant (LF) glottal flow derivative pulse, parameterized by Rd, and its harmonic table - the same as
// nn/lf.py (a port of the former lib/glottal.js, kept separate so the renderer doesn't change when lib/ is recalibrated). The
// pulse's shape in normalized time depends only on Rd, normalized so its negative peak (Ee) is -1, so its Fourier
// coefficients are independent of F0. The table rows (one per Rd grid point) are computed on first use.

import {LF_HARMONICS, LF_POINTS, LF_RD_MAX, LF_RD_MIN, LF_RD_STEP} from './params.js';
import {fft} from './fft.js';


// The LF timing parameters (fractions of the period) for an Rd - tp, te, ta
export function lfTiming(rdIn) {
    const rd = Math.max(0.3, Math.min(2.7, rdIn));
    const ra = (-1 + 4.8 * rd) / 100;
    const rk = (22.4 + 11.8 * rd) / 100;
    const rg = rk / (4 * ((0.11 * rd) / (0.5 + 1.2 * rk) - ra));
    const tp = 1 / (2 * rg);
    const te = tp * (1 + rk);
    return {tp, 'te': Math.min(te, 0.98), 'ta': ra};
}


// One period (of length 1) of the LF flow derivative - a function of the time in [0, 1)
export function lfPulse(rd) {
    const timing = lfTiming(rd);
    const {tp, te} = timing;
    const ta = Math.max(timing.ta, 1e-6);
    const tc = 1;
    const wg = Math.PI / tp;

    // The return phase's decay constant: epsilon * ta = 1 - exp(-epsilon * (tc - te))
    let epsilon = 1 / ta;
    for (let iteration = 0; iteration < 50; iteration++) {
        const next = (1 - Math.exp(-epsilon * (tc - te))) / ta;
        if (Math.abs(next - epsilon) < 1e-9 * epsilon) {
            epsilon = next;
            break;
        }
        epsilon = next;
    }
    const returnArea = -(1 / (epsilon * ta)) *
        ((1 - Math.exp(-epsilon * (tc - te))) / epsilon - (tc - te) * Math.exp(-epsilon * (tc - te)));
    const openArea = (alpha) => {
        const e0 = -1 / (Math.exp(alpha * te) * Math.sin(wg * te));
        return e0 * (Math.exp(alpha * te) * (alpha * Math.sin(wg * te) - wg * Math.cos(wg * te)) + wg) / (alpha * alpha + wg * wg);
    };
    const netFlow = (alpha) => openArea(alpha) + returnArea;

    // Solve for the growth constant so the net flow is zero (scan up for the sign change, then bisect - the net flow at
    // zero growth is positive for every Rd of the clamped range)
    const step = 0.5;
    let low = 0;
    let high = 0;
    while (netFlow(high) > 0 && high < 200) {
        low = high;
        high += step;
    }
    for (let iteration = 0; iteration < 60; iteration++) {
        const mid = (low + high) / 2;
        if (netFlow(mid) > 0) {
            low = mid;
        } else {
            high = mid;
        }
    }
    const alpha = (low + high) / 2;
    const e0 = -1 / (Math.exp(alpha * te) * Math.sin(wg * te));
    const tail = Math.exp(-epsilon * (tc - te));
    return (time) => {
        if (time < te) {
            return e0 * Math.exp(alpha * time) * Math.sin(wg * time);
        }
        return -(1 / (epsilon * ta)) * (Math.exp(-epsilon * (time - te)) - tail);
    };
}


// The Rd grid size
export const LF_RD_COUNT = Math.round((LF_RD_MAX - LF_RD_MIN) / LF_RD_STEP) + 1;


// The table rows computed so far - {re, im} Float64Arrays of LF_HARMONICS coefficients
const tableRows = new Map();


// The harmonic coefficients for an Rd grid index - {re, im}
export function lfHarmonics(gridIndex) {
    if (!tableRows.has(gridIndex)) {
        const pulse = lfPulse(LF_RD_MIN + LF_RD_STEP * gridIndex);
        const re = new Float64Array(LF_POINTS);
        const im = new Float64Array(LF_POINTS);
        for (let ix = 0; ix < LF_POINTS; ix++) {
            re[ix] = pulse(ix / LF_POINTS);
        }
        fft(re, im);
        const rowRe = new Float64Array(LF_HARMONICS);
        const rowIm = new Float64Array(LF_HARMONICS);
        for (let kx = 0; kx < LF_HARMONICS; kx++) {
            rowRe[kx] = re[kx] / LF_POINTS;
            rowIm[kx] = im[kx] / LF_POINTS;
        }
        tableRows.set(gridIndex, {'re': rowRe, 'im': rowIm});
    }
    return tableRows.get(gridIndex);
}


// The glottis's open fraction of the period (the former rule synthesizer's openFraction)
export function openFraction(rd) {
    return Math.min(0.95, 0.45 + 0.17 * rd);
}
