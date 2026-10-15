# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The Python port of lib/g2p.js - the neural grapheme-to-phoneme (G2P) model's inference, from the same generated module,
lib/models/g2pModel.js (its manifest and base64 weights), so the Python front end (nn/frontend.py) pronounces every
word exactly as the app does. The design ("ar") is described in lib/g2p.js.

The arithmetic is lib/g2p.js's, operation for operation, so the floats - and so the decisions - are the same bit for
bit: every sum adds its terms one at a time in lib/g2p.js's order (numpy's matrix products and reductions sum in other
orders), and exp and tanh are V8's Math.exp and Math.tanh - fdlibm's algorithms, which V8 compiles with fused
multiply-adds on arm64 (emulated here exactly; on other machines, unfused). Words are pronounced in batches (each
word's arithmetic is unchanged by its batch).

    python -m nn.g2pmodel word...               # pronounce words
    python -m nn.g2pmodel --lines < words.txt   # a word a line in, "word<TAB>phones" out
"""

import argparse
import ast
import base64
import os
import platform
import re
import sys
import zlib

import numpy as np


MODULE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib', 'models', 'g2pModel.js')

# The words pronounced at a time
BATCH_SIZE = 128

# V8 compiles fdlibm with fused multiply-adds on arm64
FUSED = platform.machine().lower() in ('arm64', 'aarch64')


# Parse a generated model module's g2pModel object - a dict (the weights as base64 text)
def parse_module(text):
    body = text[text.index('export const g2pModel = ') + len('export const g2pModel = '):].strip().rstrip(';')
    weights = re.search(r'`([^`]*)`', body).group(1)
    body = re.sub(r'`[^`]*`', '\'\'', body)
    body = re.sub(r'\btrue\b', 'True', re.sub(r'\bfalse\b', 'False', body))
    model = ast.literal_eval(body)
    model['weights'] = weights
    return model


# Read the model module (lib/models/g2pModel.js)
def read_module(path=MODULE_PATH):
    with open(path, encoding='utf-8') as file:
        return parse_module(file.read())


# The tensors of a model - name to float64 array (as lib/g2p.js unpackTensors - 1-D tensors as given)
def unpack_tensors(model):
    data = base64.b64decode(re.sub(r'\s+', '', model['weights']))
    if model['deflate']:
        data = zlib.decompress(data, -15)
    data = np.frombuffer(data, dtype=np.uint8)
    group_size = model['groupSize']
    shapes = [(1, shape[0]) if len(shape) == 1 else tuple(shape) for shape in (tensor['shape'] for tensor in model['tensors'])]

    # The scales - bfloat16
    offset = 0
    scales = []
    for rows, columns in shapes:
        count = rows * (-(-columns // group_size) if group_size else 1)
        codes = data[offset:offset + 2 * count].view('<u2').astype(np.uint32) << 16
        scales.append(codes.view(np.float32).astype(np.float64))
        offset += 2 * count

    # The values
    tensors = {}
    for tensor, (rows, columns), tensor_scales in zip(model['tensors'], shapes, scales):
        bits = tensor['bits']
        count = rows * columns
        if model['layout'] == 'bytes':
            unsigned = data[offset:offset + count].astype(np.int64)
            offset += count
        else:
            byte_count = -(-count * bits // 8)
            stream = np.unpackbits(data[offset:offset + byte_count], bitorder='little')[:count * bits]
            unsigned = (stream.reshape(count, bits).astype(np.int64) << np.arange(bits)).sum(1)
            offset += byte_count
        values = (unsigned - ((1 << (bits - 1)) - 1)).astype(np.float64).reshape(rows, columns)
        if group_size:
            groups = np.repeat(tensor_scales.reshape(rows, -1), group_size, axis=1)[:, :columns]
        else:
            groups = tensor_scales.reshape(rows, 1)
        tensors[tensor['name']] = (values * groups).reshape(tensor['shape'])
    return tensors


#
# V8's Math.exp and Math.tanh (fdlibm), elementwise over arrays
#

# a + b and its rounding error, exactly (Knuth's TwoSum)
def _two_sum(first, second):
    total = first + second
    second_part = total - first
    return total, (first - (total - second_part)) + (second - second_part)


# A double split into two halves of 26 bits (Veltkamp)
def _split(value):
    scaled = 134217729.0 * value
    high = scaled - (scaled - value)
    return high, value - high


# a * b + c with one rounding (a fused multiply-add) - Dekker's exact product, then Boldo and Melquiond's sum with
# rounding to odd
def _fma(first, second, addend):
    product = first * second
    first_high, first_low = _split(first)
    second_high, second_low = _split(second)
    product_error = ((first_high * second_high - product) + first_high * second_low + first_low * second_high) + first_low * second_low
    high, low = _two_sum(addend, product)
    rest, rest_error = _two_sum(low, product_error)
    even = (rest.view(np.uint64) & np.uint64(1)) == 0
    rest = np.where((rest_error != 0) & even, np.nextafter(rest, np.where(rest_error > 0, np.inf, -np.inf)), rest)
    return high + rest


# a * b + c - fused on arm64, as the C compiler contracts it there
def _mul_add(first, second, addend):
    return _fma(first, second, addend) if FUSED else first * second + addend


# The high 32 bits of doubles, signed
def _high_word(values):
    return (values.view(np.uint64) >> np.uint64(32)).astype(np.uint32).view(np.int32).astype(np.int64)


# Powers of two - 2^k for integers k in the normal range
def _power_of_two(exponents):
    return ((exponents.astype(np.int64) + 1023) << 52).view(np.float64)


# fdlibm's argument reduction for exp and expm1 - (k, hi, lo, the reduced argument) for |x| > ln2 / 2, else k = 0
def _reduce(values, high_word, negative):
    large = high_word > 0x3FD62E42
    near = large & (high_word < 0x3FF0A2B2)
    far = large & ~near
    exponent = np.zeros(values.shape, dtype=np.int64)
    high = np.zeros_like(values)
    low = np.zeros_like(values)
    ln2_high, ln2_low = 6.93147180369123816490e-01, 1.90821492927058770002e-10
    high[near] = np.where(negative[near], values[near] + ln2_high, values[near] - ln2_high)
    low[near] = np.where(negative[near], -ln2_low, ln2_low)
    exponent[near] = np.where(negative[near], -1, 1)
    with np.errstate(over='ignore', invalid='ignore'):
        rounded = np.trunc(_mul_add(1.44269504088896338700e+00, values[far], np.where(negative[far], -0.5, 0.5)))
        exponent[far] = rounded.astype(np.int64)
    high[far] = _mul_add(-rounded, ln2_high, values[far])
    low[far] = rounded * ln2_low
    return exponent, high, low, np.where(large, high - low, values), large, near


# Math.exp
def js_exp(values):
    values = np.asarray(values, dtype=np.float64)
    high_word = _high_word(values)
    negative = high_word < 0
    high_word = high_word & 0x7FFFFFFF
    exponent, high, low, reduced, _, near = _reduce(values, high_word, negative)
    square = reduced * reduced
    poly = _mul_add(square, 4.13813679705723846039e-08, -1.65339022054652515390e-06)
    poly = _mul_add(square, poly, 6.61375632143793436117e-05)
    poly = _mul_add(square, poly, -2.77777777770155933842e-03)
    poly = _mul_add(square, poly, 1.66666666666666019037e-01)
    correction = _mul_add(-square, poly, reduced)
    with np.errstate(all='ignore'):
        unscaled = 1.0 - ((reduced * correction) / (correction - 2.0) - reduced)
        base = 1.0 - ((low - (reduced * correction) / (2.0 - correction)) - high)
        normal = exponent >= -1021
        scaled = np.where(normal, base * _power_of_two(np.where(normal, np.minimum(exponent, 1023), 0)),
                          base * _power_of_two(np.where(normal, 0, exponent + 1000)) * 9.33263618503218878990e-302)
        scaled = np.where(exponent == 1024, base * 2.0 * 8.988465674311579539e307, scaled)
    result = np.where(exponent == 0, unscaled, scaled)
    result = np.where(high_word < 0x3E300000, 1.0 + values, result)
    result = np.where(near & (values == 1.0), 2.718281828459045, result)
    result = np.where(values > 7.09782712893383973096e+02, np.inf, result)
    result = np.where(values < -7.45133219101941108420e+02, 0.0, result)
    return np.where(np.isnan(values), values, result)


# fdlibm's expm1 (Math.tanh's helper)
def _expm1(values):
    high_word = _high_word(values)
    negative = high_word < 0
    high_word = high_word & 0x7FFFFFFF
    exponent, high, low, reduced, large, _ = _reduce(values, high_word, negative)
    carry = np.where(large, (high - reduced) - low, 0.0)
    half = 0.5 * reduced
    half_square = reduced * half
    poly = _mul_add(half_square, -2.01099218183624371326e-07, 4.00821782732936239552e-06)
    poly = _mul_add(half_square, poly, -7.93650757867487942473e-05)
    poly = _mul_add(half_square, poly, 1.58730158725481460165e-03)
    poly = _mul_add(half_square, poly, -3.33333333333331316428e-02)
    ratio = _mul_add(half_square, poly, 1.0)
    term = _mul_add(-ratio, half, 3.0)
    error = half_square * ((ratio - term) / _mul_add(-reduced, term, 6.0))
    result_zero = reduced - _mul_add(reduced, error, -half_square)
    error = _mul_add(reduced, error - carry, -carry) - half_square
    with np.errstate(all='ignore'):
        result_minus_one = _mul_add(0.5, reduced - error, -0.5)
        result_one = np.where(reduced < -0.25, -2.0 * (error - (reduced + 0.5)), _mul_add(2.0, reduced - error, 1.0))
        two_k = _power_of_two(np.clip(exponent, -1022, 1023))
        result_large = 1.0 - (error - reduced)
        result_large = np.where(exponent == 1024, result_large * 2.0 * 8.98846567431158e+307, result_large * two_k) - 1.0
        shift = np.clip(exponent, 0, 63)
        one_minus = ((np.int64(0x3FF00000) - (np.int64(0x200000) >> shift)) << 32).view(np.float64)
        result_small = (one_minus - (error - reduced)) * two_k
        result_middle = ((reduced - (error + ((np.int64(0x3FF) - shift) << 52).view(np.float64))) + 1.0) * two_k
    result = np.where(exponent < 20, result_small, result_middle)
    result = np.where((exponent <= -2) | (exponent > 56), result_large, result)
    result = np.where(exponent == 1, result_one, result)
    result = np.where(exponent == -1, result_minus_one, result)
    result = np.where(exponent == 0, result_zero, result)
    result = np.where(high_word < 0x3C900000, values, result)
    result = np.where(negative & (high_word >= 0x4043687A), -1.0, result)
    return np.where(values > 7.09782712893383973096e+02, np.inf, result)


# Math.tanh
def js_tanh(values):
    values = np.asarray(values, dtype=np.float64)
    high_word = _high_word(values)
    magnitude_word = high_word & 0x7FFFFFFF
    magnitude = np.abs(values)
    large = magnitude_word >= 0x3FF00000
    with np.errstate(all='ignore'):
        expm1 = _expm1(np.where(large, 2.0 * magnitude, -2.0 * magnitude))
        result = np.where(large, 1.0 - 2.0 / (expm1 + 2.0), -expm1 / (expm1 + 2.0))
    result = np.where(magnitude_word >= 0x40360000, 1.0, result)
    result = np.where(high_word < 0, -result, result)
    result = np.where(magnitude_word < 0x3E300000, values, result)
    return np.where(np.isnan(values), values, result)


#
# The inference - lib/g2p.js loadG2p, batched
#

# Rows of matrix @ vector + bias, summed as lib/g2p.js matrixVector sums (the bias, then each column in order) - columns
# is the matrix transposed ([columns, rows], contiguous), vectors [batch, columns], bias [rows] or None (from zero) -
# [batch, rows]
def _matrix_vectors(columns, vectors, bias=None):
    sums = np.zeros((vectors.shape[0], columns.shape[1]))
    if bias is not None:
        sums[:] = bias
    product = np.empty_like(sums)
    for column, weights in enumerate(columns):
        np.multiply(vectors[:, column, None], weights, out=product)
        sums += product
    return sums


# A GRU's weights from the tensors by PyTorch name prefix and suffix (for example, "encoder", "_reverse") - the weight
# matrices transposed
def _gru_weights(tensors, prefix, suffix, hidden):
    return {
        'hidden': hidden,
        'inputColumns': np.ascontiguousarray(tensors[f'{prefix}.weight_ih_l0{suffix}'].T),
        'stateColumns': np.ascontiguousarray(tensors[f'{prefix}.weight_hh_l0{suffix}'].T),
        'inputBias': tensors[f'{prefix}.bias_ih_l0{suffix}'],
        'stateBias': tensors[f'{prefix}.bias_hh_l0{suffix}']
    }


# The input gate projections of every row of an embedding table - [rows, gates] (lib/g2p.js inputTable)
def _input_table(gru, embedding):
    return _matrix_vectors(gru['inputColumns'], embedding, gru['inputBias'])


# A GRU step from its input gates (PyTorch's gate order and formula: r, z, n; h' = n + z * (h - n)) - the new states
# [batch, hidden] (lib/g2p.js gruStep)
def _gru_step(gru, gates, state):
    hidden = gru['hidden']
    scratch = _matrix_vectors(gru['stateColumns'], state, gru['stateBias'])
    reset = 1 / (1 + js_exp(-(gates[:, :hidden] + scratch[:, :hidden])))
    update = 1 / (1 + js_exp(-(gates[:, hidden:2 * hidden] + scratch[:, hidden:2 * hidden])))
    candidate = js_tanh(gates[:, 2 * hidden:] + reset * scratch[:, 2 * hidden:])
    return candidate + update * (state - candidate)


class G2p:
    """A loaded G2P model - pronounce(word) gives the word's ARPAbet phones with stress, cached per word (lib/g2p.js loadG2p)"""

    def __init__(self, model):
        if model['design'] != 'ar':
            raise ValueError(f'Unsupported G2P model design {model["design"]}')
        self.model = model
        tensors = unpack_tensors(model)
        self.hidden, self.decoder = model['hidden'], model['decoder']
        letter_embed = tensors['letter_embed.weight']
        self.forward = _gru_weights(tensors, 'encoder', '', self.hidden)
        self.backward = _gru_weights(tensors, 'encoder', '_reverse', self.hidden)
        self.forward_table = _input_table(self.forward, letter_embed)
        self.backward_table = _input_table(self.backward, letter_embed)
        self.decoder_gru = _gru_weights(tensors, 'decoder', '', self.decoder)
        self.phone_table = _input_table(self.decoder_gru, tensors['phone_embed.weight'])
        combine = tensors['combine.weight']
        self.attention_columns = tensors['attention.weight']
        self.combine_state_columns = np.ascontiguousarray(combine[:, :self.decoder].T)
        self.combine_context_columns = np.ascontiguousarray(combine[:, self.decoder:].T)
        self.combine_bias = tensors['combine.bias']
        self.output_columns = np.ascontiguousarray(tensors['output.weight'].T)
        self.output_bias = tensors['output.bias']
        self.bridge_columns = np.ascontiguousarray(tensors['bridge.weight'].T)
        self.bridge_bias = tensors['bridge.bias']
        self.letter_ids = {letter: index + 1 for index, letter in enumerate(model['letters'])}
        self.cache = {}

    # The letter ids of a word - its known letters, at most maxLetters
    def word_ids(self, word):
        return [self.letter_ids[letter] for letter in word if letter in self.letter_ids][:self.model['maxLetters']]

    # Pronounce letter-id lists (each non-empty) - phone-id lists
    def pronounce_ids(self, id_lists):
        hidden, decoder = self.hidden, self.decoder
        count = len(id_lists)
        lengths = np.array([len(ids) for ids in id_lists])
        steps = int(lengths.max())
        ids = np.zeros((count, steps), dtype=np.int64)
        for index, word_ids in enumerate(id_lists):
            ids[index, :len(word_ids)] = word_ids

        # The bidirectional encoder - each step's output (forward then backward) and the final states
        encoded = np.zeros((count, steps, 2 * hidden))
        state = np.zeros((count, hidden))
        for step in range(steps):
            valid = (step < lengths)[:, None]
            state = np.where(valid, _gru_step(self.forward, self.forward_table[ids[:, step]], state), state)
            encoded[:, step, :hidden] = np.where(valid, state, 0.0)
        ends = np.zeros((count, 2 * hidden))
        ends[:, :hidden] = state
        state = np.zeros((count, hidden))
        for step in range(steps - 1, -1, -1):
            valid = (step < lengths)[:, None]
            state = np.where(valid, _gru_step(self.backward, self.backward_table[ids[:, step]], state), state)
            encoded[:, step, hidden:] = np.where(valid, state, 0.0)
        ends[:, hidden:] = state
        state = js_tanh(_matrix_vectors(self.bridge_columns, ends, self.bridge_bias))

        # Each letter's attention key (attention^T @ encoded) and context value (combine's context half @ encoded)
        flat = encoded.reshape(count * steps, 2 * hidden)
        keys = _matrix_vectors(self.attention_columns, flat).reshape(count, steps, decoder)
        values = _matrix_vectors(self.combine_context_columns, flat).reshape(count, steps, decoder)
        letter_valid = np.arange(steps)[None, :] < lengths[:, None]

        # Decode greedily until <eos> - at most 3 phones a letter, plus 6
        phones = [[] for _ in range(count)]
        limits = 3 * lengths + 6
        active = np.ones(count, dtype=bool)
        previous = np.ones(count, dtype=np.int64)
        phone_step = 0
        while active.any():
            state = _gru_step(self.decoder_gru, self.phone_table[previous], state)
            scores = np.zeros((count, steps))
            for index in range(decoder):
                scores += keys[:, :, index] * state[:, index, None]
            max_score = np.where(letter_valid, scores, -np.inf).max(axis=1)
            weights = np.where(letter_valid, js_exp(scores - max_score[:, None]), 0.0)
            total = np.zeros(count)
            for letter in range(steps):
                total += weights[:, letter]
            sums = _matrix_vectors(self.combine_state_columns, state, self.combine_bias)
            for letter in range(steps):
                sums += (weights[:, letter] / total)[:, None] * values[:, letter, :]
            logits = _matrix_vectors(self.output_columns, js_tanh(sums), self.output_bias)
            best = np.argmax(logits, axis=1)
            for index in np.nonzero(active)[0]:
                if best[index] <= 1:
                    active[index] = False
                else:
                    phones[index].append(int(best[index]))
            phone_step += 1
            active &= phone_step < limits
            previous = np.where(best > 1, best, previous)
        return phones

    # Pronounce words, filling the cache - the words' phones (the new words in batches of similar length)
    def pronounce_many(self, words):
        missing = sorted({word for word in words if word not in self.cache}, key=lambda word: (len(self.word_ids(word)), word))
        nonempty = []
        for word in missing:
            if self.word_ids(word):
                nonempty.append(word)
            else:
                self.cache[word] = []
        for start in range(0, len(nonempty), BATCH_SIZE):
            batch = nonempty[start:start + BATCH_SIZE]
            for word, phone_ids in zip(batch, self.pronounce_ids([self.word_ids(word) for word in batch])):
                self.cache[word] = [self.model['phones'][index] for index in phone_ids]
        return [self.cache[word] for word in words]

    # The ARPAbet phones (with stress) of a word
    def pronounce(self, word):
        phones = self.cache.get(word)
        if phones is None:
            phones = self.pronounce_many([word])[0]
        return phones


# The command line - pronounce words
def main(argv=None):
    parser = argparse.ArgumentParser(prog='nn.g2pmodel', description='Pronounce words with the G2P model (lib/models/g2pModel.js)')
    parser.add_argument('words', nargs='*', help='words to pronounce')
    parser.add_argument('--lines', action='store_true', help='also read words from standard input, a word a line')
    args = parser.parse_args(argv)
    words = list(args.words)
    if args.lines:
        words.extend(line.strip() for line in sys.stdin if line.strip())
    g2p = G2p(read_module())
    for word, phones in zip(words, g2p.pronounce_many(words)):
        print(f'{word}\t{" ".join(phones)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
