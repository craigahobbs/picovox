// A counter-based random number generator - mulberry32's output function applied to a counter (nn/rng.py), so any
// draw can be computed directly and the Python and JavaScript synthesizers use identical noise. Draw ix of a stream
// with seed s is mulberry32's (ix + 1)th output for the initial state s; a Gaussian draw ix is the former rule synthesizer's
// approximation from the uniform draws 6 ix ... 6 ix + 5.


// The synthesizers' random streams
export const STREAM_ASPIRATION = 1;
export const STREAM_FRICATION = 2;
export const STREAM_JITTER = 3;
export const STREAM_SHIMMER = 4;


// The seed of a stream of a synthesis seed (32-bit)
export function streamSeed(seed, stream) {
    // (seed * 0x9E3779B1 + stream * 0x85EBCA77) mod 2^32, exactly
    let mixed = (Math.imul(seed >>> 0, 0x9E3779B1) + Math.imul(stream >>> 0, 0x85EBCA77)) >>> 0;
    mixed = (mixed ^ (mixed >>> 16)) >>> 0;
    mixed = Math.imul(mixed, 0x7FEB352D) >>> 0;
    mixed = (mixed ^ (mixed >>> 15)) >>> 0;
    return mixed;
}


// The uniform draw [0, 1) at a counter
export function uniformAt(seed, index) {
    // (seed + (index + 1) * 0x6D2B79F5) mod 2^32, exactly (Math.imul keeps the product's low 32 bits)
    const state = (seed + Math.imul((index + 1) >>> 0, 0x6D2B79F5)) >>> 0;
    let mixed = state;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
}


// The Gaussian draw (unit variance) at a counter
export function gaussAt(seed, index) {
    let sum = 0;
    for (let offset = 0; offset < 6; offset++) {
        sum += uniformAt(seed, index * 6 + offset);
    }
    return (sum - 3) / Math.sqrt(0.5);
}
