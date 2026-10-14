// ARPAbet phonemes - classes, Klatt's inherent and minimum durations (ms), and articulatory targets
//
// Vowel formants (F1, F2, F3 in Hz) are per voice ('male', 'female') - a diphthong has a start and an end target, as
// do the lax vowels IH, EH, AE, and UH, which glide slightly toward center (vowel-inherent spectral change).
// Consonant formants are male loci (scaled for other voices), pulled toward the adjacent vowel by the coupling k.


// The vowels - male targets blend Hillenbrand et al. (1995) and Peterson and Barney (1952) averages. Tense IY is
// longer than lax IH and UH (Peterson and Lehiste, 1960) - the length difference is a key tense-lax cue.
export const vowels = {
    'IY': {'dur': [180, 60], 'male': [[300, 2300, 3000]], 'female': [[340, 2750, 3250]]},
    'IH': {'dur': [120, 40], 'male': [[420, 2000, 2650], [450, 1880, 2600]], 'female': [[500, 2320, 3000], [530, 2160, 2950]]},
    'EY': {'dur': [180, 100], 'male': [[480, 2050, 2650], [340, 2250, 2850]], 'female': [[530, 2450, 3000], [420, 2650, 3200]]},
    'EH': {'dur': [150, 70], 'male': [[570, 1800, 2550], [600, 1700, 2500]], 'female': [[710, 2080, 2950], [740, 1950, 2950]]},
    'AE': {'dur': [230, 80], 'male': [[680, 1780, 2450], [690, 1640, 2450]], 'female': [[840, 2080, 2900], [860, 1880, 2900]]},
    'AA': {'dur': [240, 100], 'male': [[740, 1280, 2450]], 'female': [[900, 1400, 2800]]},
    'AO': {'dur': [240, 100], 'male': [[620, 950, 2500]], 'female': [[760, 1100, 2800]]},
    'OW': {'dur': [220, 80], 'male': [[520, 1020, 2450], [410, 930, 2350]], 'female': [[590, 1170, 2800], [460, 1080, 2750]]},
    'UH': {'dur': [140, 50], 'male': [[460, 1100, 2400], [480, 1230, 2400]], 'female': [[520, 1230, 2800], [540, 1400, 2800]]},
    'UW': {'dur': [210, 70], 'male': [[340, 1100, 2300]], 'female': [[420, 1250, 2700]]},
    'AH': {'dur': [140, 60], 'male': [[620, 1220, 2500]], 'female': [[750, 1450, 2900]]},
    'ER': {'dur': [180, 80], 'male': [[470, 1350, 1700]], 'female': [[520, 1550, 1900]]},
    'AY': {'dur': [250, 150], 'male': [[730, 1250, 2450], [420, 1950, 2600]], 'female': [[880, 1450, 2800], [500, 2300, 2950]]},
    'AW': {'dur': [260, 100], 'male': [[730, 1250, 2450], [450, 950, 2350]], 'female': [[880, 1450, 2800], [520, 1100, 2750]]},
    'OY': {'dur': [280, 150], 'male': [[560, 900, 2450], [420, 1850, 2600]], 'female': [[640, 1050, 2800], [500, 2200, 2950]]}
};


// The reduced vowel (schwa) targets
export const schwa = {'male': [500, 1450, 2450], 'female': [580, 1700, 2850]};


// The consonants - type, voicing, place, durations, male formant loci, and the F2 (k) and F3 (k3) couplings
export const consonants = {
    'P': {'type': 'stop', 'voiced': false, 'place': 'labial', 'dur': [90, 50], 'f': [250, 900, 2100], 'k': 0.55},
    'B': {'type': 'stop', 'voiced': true, 'place': 'labial', 'dur': [85, 60], 'f': [250, 900, 2100], 'k': 0.55},
    'T': {'type': 'stop', 'voiced': false, 'place': 'alveolar', 'dur': [75, 50], 'f': [250, 1700, 2800], 'k': 0.4, 'k3': 0.2},
    'D': {'type': 'stop', 'voiced': true, 'place': 'alveolar', 'dur': [75, 50], 'f': [250, 1700, 2800], 'k': 0.4, 'k3': 0.2},
    'K': {'type': 'stop', 'voiced': false, 'place': 'velar', 'dur': [80, 60], 'f': [250, 1900, 2300], 'k': 0.5, 'k3': 0.3},
    'G': {'type': 'stop', 'voiced': true, 'place': 'velar', 'dur': [80, 60], 'f': [250, 1900, 2300], 'k': 0.5, 'k3': 0.3},
    'CH': {'type': 'affricate', 'voiced': false, 'place': 'postalveolar', 'dur': [130, 80], 'f': [300, 1840, 2750], 'k': 0.3},
    'JH': {'type': 'affricate', 'voiced': true, 'place': 'postalveolar', 'dur': [110, 70], 'f': [300, 1840, 2750], 'k': 0.3},
    'F': {'type': 'fricative', 'voiced': false, 'place': 'labiodental', 'dur': [100, 80], 'f': [340, 1100, 2080], 'k': 0.5},
    'V': {'type': 'fricative', 'voiced': true, 'place': 'labiodental', 'dur': [60, 40], 'f': [300, 1100, 2080], 'k': 0.5},
    'TH': {'type': 'fricative', 'voiced': false, 'place': 'dental', 'dur': [90, 60], 'f': [320, 1400, 2540], 'k': 0.45},
    'DH': {'type': 'fricative', 'voiced': true, 'place': 'dental', 'dur': [50, 30], 'f': [270, 1400, 2540], 'k': 0.45},
    'S': {'type': 'fricative', 'voiced': false, 'place': 'alveolar', 'dur': [105, 60], 'f': [320, 1600, 2600], 'k': 0.3},
    'Z': {'type': 'fricative', 'voiced': true, 'place': 'alveolar', 'dur': [75, 40], 'f': [280, 1600, 2600], 'k': 0.3},
    'SH': {'type': 'fricative', 'voiced': false, 'place': 'postalveolar', 'dur': [105, 80], 'f': [300, 1840, 2750], 'k': 0.3},
    'ZH': {'type': 'fricative', 'voiced': true, 'place': 'postalveolar', 'dur': [70, 40], 'f': [300, 1840, 2750], 'k': 0.3},
    'HH': {'type': 'aspirate', 'voiced': false, 'place': 'glottal', 'dur': [80, 20], 'f': [500, 1500, 2500], 'k': 1},
    'M': {'type': 'nasal', 'voiced': true, 'place': 'labial', 'dur': [70, 60], 'f': [480, 1000, 2200], 'k': 0.45},
    'N': {'type': 'nasal', 'voiced': true, 'place': 'alveolar', 'dur': [60, 50], 'f': [480, 1650, 2800], 'k': 0.3, 'k3': 0.2},
    'NG': {'type': 'nasal', 'voiced': true, 'place': 'velar', 'dur': [95, 60], 'f': [480, 1900, 2300], 'k': 0.5, 'k3': 0.3},
    'L': {'type': 'liquid', 'voiced': true, 'place': 'alveolar', 'dur': [80, 40], 'f': [300, 1050, 2800], 'k': 0.25},
    'R': {'type': 'liquid', 'voiced': true, 'place': 'postalveolar', 'dur': [80, 30], 'f': [320, 1050, 1450], 'k': 0.12, 'k3': 0},
    'W': {'type': 'glide', 'voiced': true, 'place': 'labial', 'dur': [80, 60], 'f': [290, 650, 2200], 'k': 0.1},
    'Y': {'type': 'glide', 'voiced': true, 'place': 'palatal', 'dur': [80, 40], 'f': [250, 2250, 3050], 'k': 0.12}
};


// Noise spectra (male) - two resonances (frequency, bandwidth, amplitude), a flat bypass amplitude, a high-pass
// cutoff, and the level (dB relative to a stressed vowel). Fricatives use their own; stop bursts use their place's.
export const noiseSpectra = {
    'S': {'fp1': 5000, 'bp1': 2500, 'ap1': 1, 'fp2': 7500, 'bp2': 3000, 'ap2': 0.7, 'ab': 0.03, 'fhp': 3200, 'db': -10},
    'Z': {'fp1': 5000, 'bp1': 2500, 'ap1': 1, 'fp2': 7500, 'bp2': 3000, 'ap2': 0.7, 'ab': 0.03, 'fhp': 3200, 'db': -16},
    'SH': {'fp1': 2700, 'bp1': 1000, 'ap1': 1, 'fp2': 4200, 'bp2': 2500, 'ap2': 0.6, 'ab': 0.02, 'fhp': 1700, 'db': -8},
    'ZH': {'fp1': 2700, 'bp1': 1000, 'ap1': 1, 'fp2': 4200, 'bp2': 2500, 'ap2': 0.6, 'ab': 0.02, 'fhp': 1700, 'db': -14},
    'CH': {'fp1': 2700, 'bp1': 1000, 'ap1': 1, 'fp2': 4200, 'bp2': 2500, 'ap2': 0.6, 'ab': 0.02, 'fhp': 1700, 'db': -8},
    'JH': {'fp1': 2700, 'bp1': 1000, 'ap1': 1, 'fp2': 4200, 'bp2': 2500, 'ap2': 0.6, 'ab': 0.02, 'fhp': 1700, 'db': -14},
    'F': {'fp1': 7000, 'bp1': 6000, 'ap1': 0.4, 'fp2': 3000, 'bp2': 3000, 'ap2': 0.2, 'ab': 0.6, 'fhp': 1200, 'db': -22},
    'V': {'fp1': 7000, 'bp1': 6000, 'ap1': 0.4, 'fp2': 3000, 'bp2': 3000, 'ap2': 0.2, 'ab': 0.6, 'fhp': 1200, 'db': -31},
    'TH': {'fp1': 6000, 'bp1': 5000, 'ap1': 0.5, 'fp2': 3500, 'bp2': 3000, 'ap2': 0.2, 'ab': 0.5, 'fhp': 1400, 'db': -27},
    'DH': {'fp1': 6000, 'bp1': 5000, 'ap1': 0.5, 'fp2': 3500, 'bp2': 3000, 'ap2': 0.2, 'ab': 0.5, 'fhp': 1400, 'db': -33},
    'labial': {'fp1': 1200, 'bp1': 2000, 'ap1': 0.6, 'fp2': 3500, 'bp2': 3000, 'ap2': 0.15, 'ab': 0.4, 'fhp': 500, 'db': -34},
    'alveolar': {'fp1': 4000, 'bp1': 2500, 'ap1': 1, 'fp2': 6500, 'bp2': 3000, 'ap2': 0.5, 'ab': 0.1, 'fhp': 2500, 'db': -30},
    'velar': {'fp1': 2000, 'bp1': 500, 'ap1': 1, 'fp2': 3500, 'bp2': 2000, 'ap2': 0.1, 'ab': 0.05, 'fhp': 1000, 'db': -36}
};


// Is a phoneme a vowel?
export function isVowel(ph) {
    return ph in vowels;
}
