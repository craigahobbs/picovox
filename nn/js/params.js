// The synthesizer parameter set - GENERATED from nn/params.py (python -m nn.params --write-js); do not edit.
// See nn/params.py for the parameters' meanings and the signal model.

export const FRAME_RATE = 200;
export const REF_RATE = 44100;
export const FLOOR_DB = -90.0;
export const SHELF_HZ = 2000.0;
export const NOISE_MODULATION = 0.7;
export const HIGHER_POLES = 6;
export const HIGHER_POLE_SPACING = 0.18181818181818182;
export const HIGHER_POLE_BANDWIDTH = 500.0;
export const HIGHER_POLE_BANDWIDTH_STEP = 100.0;
export const HARMONIC_TAPER_START = 0.44;
export const HARMONIC_TAPER_END = 0.475;
export const F0_MIN = 40.0;
export const LF_RD_MIN = 0.3;
export const LF_RD_MAX = 2.7;
export const LF_RD_STEP = 0.01;
export const LF_HARMONICS = 1024;
export const LF_POINTS = 8192;
export const EQ_BAND_COUNT = 24;
export const EQ_LOW_HZ = 60.0;
export const EQ_HIGH_HZ = 16000.0;
export const EQ_RANGE_DB = 30.0;
export const CORRECTION_LOW_HZ = 100.0;
export const CORRECTION_HIGH_HZ = 12000.0;

// The parameters - {name, group, units, scale, low, high, default, doc}
export const PARAMETERS = [
    {'name': 'f0', 'group': 'source', 'units': 'Hz', 'scale': 'log', 'low': 40.0, 'high': 600.0, 'default': 100.0,
        'doc': 'The fundamental frequency'},
    {'name': 'av', 'group': 'source', 'units': 'dB', 'scale': 'linear', 'low': -90.0, 'high': 12.0, 'default': -90.0,
        'doc': 'The voicing amplitude (0 dB - an LF pulse train with Ee = 1)'},
    {'name': 'rd', 'group': 'source', 'units': 'Rd', 'scale': 'linear', 'low': 0.3, 'high': 2.7, 'default': 1.0,
        'doc': 'The LF pulse shape - 0.3 pressed, ~1 modal, 2.7 lax and breathy'},
    {'name': 'shelf', 'group': 'source', 'units': 'dB', 'scale': 'linear', 'low': -12.0, 'high': 24.0, 'default': 0.0,
        'doc': 'The voicing\'s high-shelf gain at 2000 Hz - spectral tilt correction (brightness, effort)'},
    {'name': 'jitter', 'group': 'source', 'units': 'fraction', 'scale': 'linear', 'low': 0.0, 'high': 0.1, 'default': 0.0,
        'doc': 'Random cycle-to-cycle period variation (standard deviation)'},
    {'name': 'shimmer', 'group': 'source', 'units': 'fraction', 'scale': 'linear', 'low': 0.0, 'high': 0.5, 'default': 0.0,
        'doc': 'Random cycle-to-cycle amplitude variation (standard deviation)'},
    {'name': 'ah', 'group': 'source', 'units': 'dB', 'scale': 'linear', 'low': -90.0, 'high': 12.0, 'default': -90.0,
        'doc': 'The aspiration noise amplitude (0 dB - unit-variance white noise at 44.1 kHz), through the tract'},
    {'name': 'af', 'group': 'source', 'units': 'dB', 'scale': 'linear', 'low': -90.0, 'high': 12.0, 'default': -90.0,
        'doc': 'The frication noise amplitude (0 dB - unit-variance white noise at 44.1 kHz), through the frication filter'},
    {'name': 'f1', 'group': 'tract', 'units': 'Hz', 'scale': 'log', 'low': 120.0, 'high': 1400.0, 'default': 500.0,
        'doc': 'The first formant frequency'},
    {'name': 'f2', 'group': 'tract', 'units': 'Hz', 'scale': 'log', 'low': 400.0, 'high': 3400.0, 'default': 1500.0,
        'doc': 'The second formant frequency'},
    {'name': 'f3', 'group': 'tract', 'units': 'Hz', 'scale': 'log', 'low': 1000.0, 'high': 4500.0, 'default': 2500.0,
        'doc': 'The third formant frequency'},
    {'name': 'f4', 'group': 'tract', 'units': 'Hz', 'scale': 'log', 'low': 2000.0, 'high': 6000.0, 'default': 3500.0,
        'doc': 'The fourth formant frequency'},
    {'name': 'f5', 'group': 'tract', 'units': 'Hz', 'scale': 'log', 'low': 2500.0, 'high': 7000.0, 'default': 4500.0,
        'doc': 'The fifth formant frequency'},
    {'name': 'f6', 'group': 'tract', 'units': 'Hz', 'scale': 'log', 'low': 3000.0, 'high': 8500.0, 'default': 5500.0,
        'doc': 'The sixth formant frequency (the higher poles follow it)'},
    {'name': 'b1', 'group': 'tract', 'units': 'Hz', 'scale': 'log', 'low': 20.0, 'high': 1000.0, 'default': 80.0,
        'doc': 'The first formant bandwidth'},
    {'name': 'b2', 'group': 'tract', 'units': 'Hz', 'scale': 'log', 'low': 20.0, 'high': 1500.0, 'default': 90.0,
        'doc': 'The second formant bandwidth'},
    {'name': 'b3', 'group': 'tract', 'units': 'Hz', 'scale': 'log', 'low': 30.0, 'high': 2000.0, 'default': 150.0,
        'doc': 'The third formant bandwidth'},
    {'name': 'b4', 'group': 'tract', 'units': 'Hz', 'scale': 'log', 'low': 40.0, 'high': 2500.0, 'default': 250.0,
        'doc': 'The fourth formant bandwidth'},
    {'name': 'b5', 'group': 'tract', 'units': 'Hz', 'scale': 'log', 'low': 50.0, 'high': 3000.0, 'default': 300.0,
        'doc': 'The fifth formant bandwidth'},
    {'name': 'b6', 'group': 'tract', 'units': 'Hz', 'scale': 'log', 'low': 60.0, 'high': 4000.0, 'default': 1000.0,
        'doc': 'The sixth formant bandwidth'},
    {'name': 'fnp', 'group': 'tract', 'units': 'Hz', 'scale': 'log', 'low': 150.0, 'high': 4000.0, 'default': 270.0,
        'doc': 'The nasal (or lateral) pole frequency'},
    {'name': 'bnp', 'group': 'tract', 'units': 'Hz', 'scale': 'log', 'low': 30.0, 'high': 2000.0, 'default': 100.0,
        'doc': 'The nasal pole bandwidth'},
    {'name': 'fnz', 'group': 'tract', 'units': 'Hz', 'scale': 'log', 'low': 150.0, 'high': 4000.0, 'default': 270.0,
        'doc': 'The nasal zero frequency - equal to the pole, they cancel'},
    {'name': 'bnz', 'group': 'tract', 'units': 'Hz', 'scale': 'log', 'low': 30.0, 'high': 2000.0, 'default': 100.0,
        'doc': 'The nasal zero bandwidth'},
    {'name': 'fp1', 'group': 'frication', 'units': 'Hz', 'scale': 'log', 'low': 300.0, 'high': 12000.0, 'default': 3000.0,
        'doc': 'The frication noise\'s first resonance frequency'},
    {'name': 'bp1', 'group': 'frication', 'units': 'Hz', 'scale': 'log', 'low': 50.0, 'high': 8000.0, 'default': 1000.0,
        'doc': 'The first resonance bandwidth'},
    {'name': 'ap1', 'group': 'frication', 'units': 'dB', 'scale': 'linear', 'low': -60.0, 'high': 12.0, 'default': -60.0,
        'doc': 'The first resonance gain (fourth-order bandpass, 0 dB peak)'},
    {'name': 'fp2', 'group': 'frication', 'units': 'Hz', 'scale': 'log', 'low': 300.0, 'high': 16000.0, 'default': 6000.0,
        'doc': 'The frication noise\'s second resonance frequency'},
    {'name': 'bp2', 'group': 'frication', 'units': 'Hz', 'scale': 'log', 'low': 50.0, 'high': 10000.0, 'default': 2000.0,
        'doc': 'The second resonance bandwidth'},
    {'name': 'ap2', 'group': 'frication', 'units': 'dB', 'scale': 'linear', 'low': -60.0, 'high': 12.0, 'default': -60.0,
        'doc': 'The second resonance gain'},
    {'name': 'ab', 'group': 'frication', 'units': 'dB', 'scale': 'linear', 'low': -60.0, 'high': 12.0, 'default': -60.0,
        'doc': 'The flat (bypass) gain'},
    {'name': 'fhp', 'group': 'frication', 'units': 'Hz', 'scale': 'log', 'low': 50.0, 'high': 10000.0, 'default': 500.0,
        'doc': 'The high-pass cutoff (24 dB/octave below it)'}
];

// The parameter names, in order
export const NAMES = PARAMETERS.map((parameter) => parameter.name);

// The default (silent) frame
export const DEFAULTS = Object.fromEntries(PARAMETERS.map((parameter) => [parameter.name, parameter.default]));

// The level parameters (dB), converted to linear amplitudes before interpolation
export const DB_NAMES = ['av', 'ah', 'af', 'ap1', 'ap2', 'ab'];

// The frequency parameters the vocal tract length scales
export const TRACT_FREQUENCIES = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'fnp', 'fnz', 'fp1', 'fp2', 'fhp'];
