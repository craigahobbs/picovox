// ARPAbet phonemes - the vowels


// The ARPAbet vowels
const vowels = new Set(['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']);


// Is a phoneme a vowel?
export function isVowel(ph) {
    return vowels.has(ph);
}
