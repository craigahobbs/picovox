// The Liljencrants-Fant (LF) glottal flow derivative model, parameterized by Fant's Rd voice quality parameter
//
// Rd runs from about 0.3 (tense, pressed) through 1.0 (modal) to 2.7 (lax, breathy). The model gives one period
// of the glottal flow derivative - the excitation of the vocal tract, including the lip radiation's emphasis.


// The LF timing parameters (fractions of the period) for an Rd - tp (peak flow), te (closure), ta (return phase)
export function lfTiming(rd) {
    const rdClamped = Math.max(0.3, Math.min(2.7, rd));
    const ra = (-1 + 4.8 * rdClamped) / 100;
    const rk = (22.4 + 11.8 * rdClamped) / 100;
    const rg = rk / (4 * ((0.11 * rdClamped) / (0.5 + 1.2 * rk) - ra));
    const tp = 1 / (2 * rg);
    const te = tp * (1 + rk);
    return {tp, 'te': Math.min(te, 0.98), 'ta': ra};
}


// Solve the LF model's shape constants for a period (in seconds) - the returned function gives the flow
// derivative at a time within the period, normalized so the negative peak (excitation) is -1
export function lfPulse(period, rd) {
    const timing = lfTiming(rd);
    const tp = timing.tp * period;
    const te = timing.te * period;
    const ta = Math.max(timing.ta * period, 1e-6);
    const tc = period;
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

    // The return phase's area, for Ee = 1
    const returnArea = -(1 / (epsilon * ta)) *
        ((1 - Math.exp(-epsilon * (tc - te))) / epsilon - (tc - te) * Math.exp(-epsilon * (tc - te)));

    // The open phase's area for a growth constant alpha, with E0 set so the derivative is -1 at te
    const openArea = (alpha) => {
        const e0 = -1 / (Math.exp(alpha * te) * Math.sin(wg * te));
        return e0 * (Math.exp(alpha * te) * (alpha * Math.sin(wg * te) - wg * Math.cos(wg * te)) + wg) / (alpha * alpha + wg * wg);
    };

    // Solve for alpha so the net flow over the period is zero. The net flow falls as alpha rises near the
    // physical solution, so scan out from zero for the first sign change, then bisect.
    const netFlow = (alpha) => openArea(alpha) + returnArea;
    const step = 0.5 / period;
    let low = 0;
    let high = 0;
    if (netFlow(0) > 0) {
        while (netFlow(high) > 0 && high < 200 / period) {
            low = high;
            high += step;
        }
    } else {
        while (netFlow(low) <= 0 && low > -200 / period) {
            high = low;
            low -= step;
        }
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
