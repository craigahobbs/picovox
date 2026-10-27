# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The parity checks - the Python port against the package: each check's inputs are committed (test/parity-inputs.json -
checked against the tests' own, or written with PARITY_WRITE=1), and the package's outputs of them are built by make
parity (tools/parity.js - build/parity-outputs.json, before make commit's training tests)
"""

import json
import os
import unittest

import numpy

from picovox_train import ROOT


INPUTS = os.path.join(ROOT, 'test', 'parity-inputs.json')
OUTPUTS = os.path.join(ROOT, 'build', 'parity-outputs.json')


# A parity check's inputs (JSON values) - checked against the committed inputs, or written there (PARITY_WRITE=1)
def inputs(test, name, value):
    value = json.loads(json.dumps(value))
    with open(INPUTS, encoding='utf-8') as file:
        committed = json.load(file)
    if os.environ.get('PARITY_WRITE'):
        committed[name] = value
        with open(INPUTS, 'w', encoding='utf-8') as file:
            file.write('{\n' + ',\n'.join(f'{json.dumps(key)}: {json.dumps(committed[key])}' for key in sorted(committed)) + '\n}\n')
    else:
        test.assertEqual(value, committed.get(name), f'{name}: the inputs differ from {INPUTS} (PARITY_WRITE=1 writes them)')
    return value


# The package's outputs of a parity check's inputs (make parity) - skipped when the inputs are written (make parity
# builds the outputs of the new inputs)
def outputs(name):
    if os.environ.get('PARITY_WRITE'):
        raise unittest.SkipTest(f'{name}: inputs written - make parity, then the tests')
    if not os.path.exists(OUTPUTS):
        raise AssertionError(f'{OUTPUTS} is missing - make parity builds it')
    with open(OUTPUTS, encoding='utf-8') as file:
        return json.load(file)[name]


# A float64 array's bits, as hex strings (JSON has no NaN or infinity), and back
def to_hex(values):
    return [f'{int(bits):016x}' for bits in numpy.asarray(values, dtype=numpy.float64).view(numpy.uint64)]


def from_hex(texts):
    return numpy.array([int(text, 16) for text in texts], dtype=numpy.uint64).view(numpy.float64)
