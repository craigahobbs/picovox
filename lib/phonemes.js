// ARPAbet phonemes - classes, inherent and minimum durations (ms), and articulatory targets
//
// The durations are Klatt's (1979) inherent and minimum durations (lib/speak.js applies his rules), each phone's scaled
// so the synthesized ARCTIC prompts match the mean rate-normalized phone durations (duration x phones per second) of
// human speakers - the CMU ARCTIC speakers and the LibriTTS-R audiobook readers (tools/targets.py), equally weighted.
//
// Vowel formants (F1, F2, F3 in Hz) are reference tables of adult men's and women's averages ('men', 'women') - a voice
// blends them (its vowelBlend parameter, lib/voices.js). A diphthong has a start and an end target, as do the lax vowels
// IH, EH, AE, and UH, which glide slightly toward center (vowel-inherent spectral change). The glide runs between two
// fractions of the vowel ('glide', default defaultGlide in lib/speak.js) - the wide diphthongs hold their nucleus and
// glide late. The back rounded vowels front after a coronal consonant by their 'fronting' (a fraction of the voice's
// coronalFronting). Consonant formants are male loci (scaled for other voices by their formantScale), pulled toward the
// adjacent vowel by the coupling k.


// The vowels - modern American English: the targets are fit so that the synthesized vowels, measured as human speech is
// (tools/synthstats.js and tools/targets.py - F1 and F2 at 20%, 50%, and 80% of each vowel), match CMU ARCTIC speakers
// (men bdl and rms, women slt and clb) and the LibriTTS-R audiobook readers - fronted UW, UH, and OW, peripheral IY, and
// AY, AW, and OY gliding to a lower, more central off-glide. Tense IY is longer than lax IH and UH (Peterson and
// Lehiste, 1960) - the length difference is a key tense-lax cue.
export const vowels = {
    'IY': {'dur': [185, 62], 'men': [[300, 2410, 2860]], 'women': [[310, 2950, 3270]]},
    'IH': {'dur': [119, 40], 'men': [[420, 1990, 2680], [500, 1960, 2630]], 'women': [[470, 2430, 3120], [580, 2450, 3070]]},
    'EY': {'dur': [170, 94], 'men': [[500, 1930, 2570], [390, 2240, 2770]], 'women': [[560, 2450, 3020], [460, 2760, 3220]]},
    'EH': {'dur': [111, 52], 'men': [[570, 1690, 2570], [640, 1580, 2520]], 'women': [[740, 2030, 3000], [860, 1970, 3000]]},
    'AE': {'dur': [182, 64], 'men': [[610, 1870, 2550], [670, 1660, 2550]], 'women': [[860, 2120, 3000], [900, 2000, 3000]]},
    'AA': {'dur': [156, 65], 'men': [[690, 1200, 2490]], 'women': [[840, 1360, 2860]]},
    'AO': {'dur': [165, 69], 'men': [[550, 920, 2440]], 'women': [[690, 1120, 2860]]},
    'OW': {'dur': [190, 69], 'men': [[530, 1060, 2500], [440, 830, 2400]],
        'women': [[580, 1350, 2870], [470, 1050, 2820]], 'fronting': 0.4},
    'UH': {'dur': [99, 35], 'men': [[440, 1130, 2430], [490, 1170, 2430]],
        'women': [[450, 1440, 2830], [540, 1440, 2830]], 'fronting': 0.6},
    'UW': {'dur': [174, 58], 'men': [[330, 1100, 2300]], 'women': [[360, 1450, 2700]], 'fronting': 1},
    'AH': {'dur': [104, 44], 'men': [[570, 1240, 2560]], 'women': [[720, 1570, 3000]]},
    'ER': {'dur': [181, 80], 'men': [[430, 1300, 1640]], 'women': [[490, 1510, 1900]]},
    'AY': {'dur': [195, 117], 'men': [[680, 1230, 2470], [540, 1830, 2620]],
        'women': [[890, 1540, 2850], [690, 2210, 3000]], 'glide': [0.35, 0.9]},
    'AW': {'dur': [239, 92], 'men': [[670, 1420, 2500], [570, 950, 2400]],
        'women': [[880, 1730, 2860], [770, 1190, 2810]], 'glide': [0.35, 0.9]},
    'OY': {'dur': [242, 130], 'men': [[460, 830, 2320], [450, 1830, 2470]],
        'women': [[490, 1000, 2760], [500, 2210, 2910]], 'glide': [0.35, 0.9]}
};


// The reduced vowel (schwa) targets
export const schwa = {'men': [500, 1450, 2450], 'women': [580, 1700, 2850]};


// The consonants - type, voicing, place, durations, male formant loci, and the F2 (k) and F3 (k3) couplings
export const consonants = {
    'P': {'type': 'stop', 'voiced': false, 'place': 'labial', 'dur': [80, 45], 'f': [250, 900, 2100], 'k': 0.55},
    'B': {'type': 'stop', 'voiced': true, 'place': 'labial', 'dur': [63, 45], 'f': [250, 900, 2100], 'k': 0.55},
    'T': {'type': 'stop', 'voiced': false, 'place': 'alveolar', 'dur': [62, 42], 'f': [250, 1700, 2800], 'k': 0.4, 'k3': 0.2},
    'D': {'type': 'stop', 'voiced': true, 'place': 'alveolar', 'dur': [66, 45], 'f': [250, 1700, 2800], 'k': 0.4, 'k3': 0.2},
    'K': {'type': 'stop', 'voiced': false, 'place': 'velar', 'dur': [60, 45], 'f': [250, 1900, 2300], 'k': 0.5, 'k3': 0.3},
    'G': {'type': 'stop', 'voiced': true, 'place': 'velar', 'dur': [65, 49], 'f': [250, 1900, 2300], 'k': 0.5, 'k3': 0.3},
    'CH': {'type': 'affricate', 'voiced': false, 'place': 'postalveolar', 'dur': [144, 89], 'f': [300, 1840, 2750], 'k': 0.3},
    'JH': {'type': 'affricate', 'voiced': true, 'place': 'postalveolar', 'dur': [129, 82], 'f': [300, 1840, 2750], 'k': 0.3},
    'F': {'type': 'fricative', 'voiced': false, 'place': 'labiodental', 'dur': [115, 92], 'f': [340, 1100, 2080], 'k': 0.5},
    'V': {'type': 'fricative', 'voiced': true, 'place': 'labiodental', 'dur': [81, 54], 'f': [300, 1100, 2080], 'k': 0.5},
    'TH': {'type': 'fricative', 'voiced': false, 'place': 'dental', 'dur': [113, 75], 'f': [320, 1400, 2540], 'k': 0.45},
    'DH': {'type': 'fricative', 'voiced': true, 'place': 'dental', 'dur': [71, 43], 'f': [270, 1400, 2540], 'k': 0.45},
    'S': {'type': 'fricative', 'voiced': false, 'place': 'alveolar', 'dur': [140, 80], 'f': [320, 1600, 2600], 'k': 0.3},
    'Z': {'type': 'fricative', 'voiced': true, 'place': 'alveolar', 'dur': [123, 65], 'f': [280, 1600, 2600], 'k': 0.3},
    'SH': {'type': 'fricative', 'voiced': false, 'place': 'postalveolar', 'dur': [144, 110], 'f': [300, 1840, 2750], 'k': 0.3},
    'ZH': {'type': 'fricative', 'voiced': true, 'place': 'postalveolar', 'dur': [130, 74], 'f': [300, 1840, 2750], 'k': 0.3},
    'HH': {'type': 'aspirate', 'voiced': false, 'place': 'glottal', 'dur': [99, 25], 'f': [500, 1500, 2500], 'k': 1},
    'M': {'type': 'nasal', 'voiced': true, 'place': 'labial', 'dur': [91, 78], 'f': [480, 1000, 2200], 'k': 0.45},
    'N': {'type': 'nasal', 'voiced': true, 'place': 'alveolar', 'dur': [79, 67], 'f': [480, 1650, 2800], 'k': 0.3, 'k3': 0.2},
    'NG': {'type': 'nasal', 'voiced': true, 'place': 'velar', 'dur': [121, 77], 'f': [480, 1900, 2300], 'k': 0.5, 'k3': 0.3},
    'L': {'type': 'liquid', 'voiced': true, 'place': 'alveolar', 'dur': [118, 59], 'f': [300, 1050, 2800], 'k': 0.25},
    'R': {'type': 'liquid', 'voiced': true, 'place': 'postalveolar', 'dur': [112, 42], 'f': [320, 1050, 1450], 'k': 0.12, 'k3': 0},
    'W': {'type': 'glide', 'voiced': true, 'place': 'labial', 'dur': [97, 73], 'f': [290, 650, 2200], 'k': 0.1},
    'Y': {'type': 'glide', 'voiced': true, 'place': 'palatal', 'dur': [109, 54], 'f': [250, 2250, 3050], 'k': 0.12}
};


// Noise spectra (male) - two resonances (frequency, bandwidth, amplitude), a flat bypass amplitude, a high-pass
// cutoff, and the level (dB relative to a stressed vowel). Fricatives use their own; stop bursts use their place's. The
// non-sibilants (F, V, TH, DH) have no front-cavity resonance - their flat spectra scale by a voice's nonSibilantScale,
// not its noiseScale.
export const noiseSpectra = {
    'S': {'fp1': 5000, 'bp1': 2500, 'ap1': 1, 'fp2': 7500, 'bp2': 3000, 'ap2': 0.7, 'ab': 0.03, 'fhp': 3200, 'db': -10},
    'Z': {'fp1': 5000, 'bp1': 2500, 'ap1': 1, 'fp2': 7500, 'bp2': 3000, 'ap2': 0.7, 'ab': 0.03, 'fhp': 3200, 'db': -16},
    'SH': {'fp1': 2700, 'bp1': 1000, 'ap1': 1, 'fp2': 4200, 'bp2': 2500, 'ap2': 0.6, 'ab': 0.02, 'fhp': 1700, 'db': -8},
    'ZH': {'fp1': 2700, 'bp1': 1000, 'ap1': 1, 'fp2': 4200, 'bp2': 2500, 'ap2': 0.6, 'ab': 0.02, 'fhp': 1700, 'db': -14},
    'CH': {'fp1': 2700, 'bp1': 1000, 'ap1': 1, 'fp2': 4200, 'bp2': 2500, 'ap2': 0.6, 'ab': 0.02, 'fhp': 1700, 'db': -8},
    'JH': {'fp1': 2700, 'bp1': 1000, 'ap1': 1, 'fp2': 4200, 'bp2': 2500, 'ap2': 0.6, 'ab': 0.02, 'fhp': 1700, 'db': -14},
    'F': {'fp1': 5000, 'bp1': 5000, 'ap1': 0.4, 'fp2': 3000, 'bp2': 3000, 'ap2': 0.2, 'ab': 0.6, 'fhp': 1000, 'db': -22, 'sibilant': false},
    'V': {'fp1': 5000, 'bp1': 5000, 'ap1': 0.4, 'fp2': 3000, 'bp2': 3000, 'ap2': 0.2, 'ab': 0.6, 'fhp': 800, 'db': -35, 'sibilant': false},
    'TH': {'fp1': 5500, 'bp1': 5000, 'ap1': 0.5, 'fp2': 3500, 'bp2': 3000, 'ap2': 0.2, 'ab': 0.5,
        'fhp': 1400, 'db': -27, 'sibilant': false},
    'DH': {'fp1': 6000, 'bp1': 5000, 'ap1': 0.5, 'fp2': 3500, 'bp2': 3000, 'ap2': 0.2, 'ab': 0.5,
        'fhp': 1400, 'db': -33, 'sibilant': false},
    'labial': {'fp1': 1200, 'bp1': 2000, 'ap1': 0.6, 'fp2': 3500, 'bp2': 3000, 'ap2': 0.15, 'ab': 0.4, 'fhp': 500, 'db': -34},
    'alveolar': {'fp1': 4000, 'bp1': 2500, 'ap1': 1, 'fp2': 6500, 'bp2': 3000, 'ap2': 0.5, 'ab': 0.1, 'fhp': 2500, 'db': -30},
    'velar': {'fp1': 2000, 'bp1': 500, 'ap1': 1, 'fp2': 3500, 'bp2': 2000, 'ap2': 0.1, 'ab': 0.05, 'fhp': 1000, 'db': -36}
};


// Is a phoneme a vowel?
export function isVowel(ph) {
    return ph in vowels;
}
