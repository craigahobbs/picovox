# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The Liljencrants-Fant (LF) glottal flow derivative pulse, parameterized by Rd - a port of the former lib/glottal.js (kept here
so nn/ doesn't change when lib/ is recalibrated) - and its harmonic table.

The pulse's shape in normalized time (a fraction of the period) depends only on Rd, and it's normalized so its
negative peak (Ee) is -1, so its Fourier series coefficients c_k = integral over one period of E(u) exp(-2 pi i k u) du
are independent of F0. The table holds them for Rd from LF_RD_MIN to LF_RD_MAX (steps of LF_RD_STEP) and
k = 0 ... LF_HARMONICS - 1, from an FFT of LF_POINTS samples of a period. lib/lf.js computes the same table.
"""

import functools
import math

import numpy

from . import params


# The LF timing parameters (fractions of the period) for an Rd - tp, te, ta
def lf_timing(rd):
    rd = max(0.3, min(2.7, rd))
    ra = (-1 + 4.8 * rd) / 100
    rk = (22.4 + 11.8 * rd) / 100
    rg = rk / (4 * ((0.11 * rd) / (0.5 + 1.2 * rk) - ra))
    tp = 1 / (2 * rg)
    te = tp * (1 + rk)
    return tp, min(te, 0.98), ra


# One period (of length 1) of the LF flow derivative at times u (a numpy array in [0, 1))
def lf_pulse(u, rd):
    tp, te, ta = lf_timing(rd)
    ta = max(ta, 1e-6)
    tc = 1.0
    wg = math.pi / tp

    # The return phase's decay constant: epsilon * ta = 1 - exp(-epsilon * (tc - te))
    epsilon = 1 / ta
    for _ in range(50):
        following = (1 - math.exp(-epsilon * (tc - te))) / ta
        if abs(following - epsilon) < 1e-9 * epsilon:
            epsilon = following
            break
        epsilon = following
    return_area = -(1 / (epsilon * ta)) * \
        ((1 - math.exp(-epsilon * (tc - te))) / epsilon - (tc - te) * math.exp(-epsilon * (tc - te)))

    def open_area(alpha):
        e0 = -1 / (math.exp(alpha * te) * math.sin(wg * te))
        return e0 * (math.exp(alpha * te) * (alpha * math.sin(wg * te) - wg * math.cos(wg * te)) + wg) / (alpha * alpha + wg * wg)

    def net_flow(alpha):
        return open_area(alpha) + return_area

    # Solve for the growth constant alpha so the net flow is zero (scan for the sign change, then bisect)
    step = 0.5
    low = 0.0
    high = 0.0
    if net_flow(0) > 0:
        while net_flow(high) > 0 and high < 200:
            low = high
            high += step
    else:
        while net_flow(low) <= 0 and low > -200:
            high = low
            low -= step
    for _ in range(60):
        mid = (low + high) / 2
        if net_flow(mid) > 0:
            low = mid
        else:
            high = mid
    alpha = (low + high) / 2
    e0 = -1 / (math.exp(alpha * te) * math.sin(wg * te))
    tail = math.exp(-epsilon * (tc - te))
    opening = e0 * numpy.exp(alpha * u) * numpy.sin(wg * u)
    closing = -(1 / (epsilon * ta)) * (numpy.exp(-epsilon * (u - te)) - tail)
    return numpy.where(u < te, opening, closing)


# The Rd grid of the harmonic table
def rd_grid():
    count = int(round((params.LF_RD_MAX - params.LF_RD_MIN) / params.LF_RD_STEP)) + 1
    return params.LF_RD_MIN + params.LF_RD_STEP * numpy.arange(count)


# The harmonic table - a complex128 array [len(rd_grid()), LF_HARMONICS]
@functools.lru_cache(maxsize=1)
def harmonic_table():
    u = numpy.arange(params.LF_POINTS) / params.LF_POINTS
    rows = []
    for rd in rd_grid():
        spectrum = numpy.fft.fft(lf_pulse(u, float(rd))) / params.LF_POINTS
        rows.append(spectrum[:params.LF_HARMONICS])
    return numpy.array(rows)


# The glottis's open fraction of the period, for the aspiration's modulation (the former lib/klatt.js's openFraction)
def open_fraction(rd):
    return numpy.minimum(0.95, 0.45 + 0.17 * rd)
