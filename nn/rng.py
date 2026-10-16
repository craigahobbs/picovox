# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
A counter-based random number generator - mulberry32's output function applied to a counter, so any draw can be
computed directly (vectorized here, sequentially in nn/js/rng.js) and the Python and JavaScript synthesizers can use
identical noise. Draw i of a stream with seed s is mulberry32's (i + 1)th output for the initial state s.

A Gaussian draw i is the former lib/klatt.js's approximation - the sum of the uniform draws 6i ... 6i + 5, centered and scaled to
unit variance.
"""

import numpy


# The synthesizers' random streams
STREAM_ASPIRATION = 1
STREAM_FRICATION = 2
STREAM_JITTER = 3
STREAM_SHIMMER = 4
STREAM_RESIDUAL = 5


# The seed of a stream of a synthesis seed (32-bit)
def stream_seed(seed, stream):
    mixed = (int(seed) * 0x9E3779B1 + int(stream) * 0x85EBCA77) & 0xFFFFFFFF
    mixed ^= mixed >> 16
    mixed = (mixed * 0x7FEB352D) & 0xFFFFFFFF
    mixed ^= mixed >> 15
    return mixed


# Uniform draws [0, 1) for counters (a uint64 numpy array of draw indices)
def uniform(seed, indices):
    with numpy.errstate(over='ignore'):
        state = (numpy.uint64(seed) + (indices.astype(numpy.uint64) + numpy.uint64(1)) * numpy.uint64(0x6D2B79F5)) & \
            numpy.uint64(0xFFFFFFFF)
        mixed = state.astype(numpy.uint32)
        mixed = (mixed ^ (mixed >> numpy.uint32(15))) * (mixed | numpy.uint32(1))
        mixed ^= mixed + (mixed ^ (mixed >> numpy.uint32(7))) * (mixed | numpy.uint32(61))
        mixed = mixed ^ (mixed >> numpy.uint32(14))
    return mixed.astype(numpy.float64) / 4294967296.0


# Gaussian draws (unit variance) for draw indices start ... start + count - 1
def gauss(seed, start, count):
    indices = (numpy.arange(count, dtype=numpy.uint64) + numpy.uint64(start)) * numpy.uint64(6)
    total = numpy.zeros(count)
    for offset in range(6):
        total += uniform(seed, indices + numpy.uint64(offset))
    return (total - 3) / numpy.sqrt(0.5)
