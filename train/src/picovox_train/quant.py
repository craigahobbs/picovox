# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
Weight quantization - the one definition shared by the export (export.py: the exported weights and the PyTorch model
with them, quantized_model) and the JavaScript loader (lib/model.js), and the error-aware rounding (gptq) the G2P's
export shares (context.py):

- The quantized tensors (is_quantized) are the matrices, convolution kernels, and embedding tables of at least
  MIN_QUANTIZED values, not the channel equalizers (exported as dB tables).
- BITS bits, symmetric, per row (a tensor's output channel - dim 0, the rest flattened): scale = max |w| / LIMIT
  (computed in float64, stored float32), q an integer within +-LIMIT, w' = q x scale (float32, as the loader computes
  it). A row of zeros has scale 1.
- q is the nearest integer to w / scale, or - given the layer's input covariance (H = E[x x^T] over calibration
  inputs) - GPTQ's: a row is rounded a column at a time, each column's rounding error compensated in the columns not yet
  rounded through H's inverse, so the layer's output (w x), not each weight, is kept. The quantization study (seed 1's
  voice, then two more seeds) found 7-bit GPTQ indistinguishable from float - UTMOS, WER, the recordings' spectra and
  pitch - where nearest rounding at 7 bits drifted the pitch three times as far.
- The values are packed (pack) as unsigned integers (q + LIMIT) of BITS bits, little-endian bit by bit, a tensor's
  starting on a byte.
"""

import numpy


# The weights' width, and the largest integer magnitude
BITS = 7
LIMIT = (1 << (BITS - 1)) - 1

# Tensors smaller than this stay float32
MIN_QUANTIZED = 1024

# The parameters (state names) that stay float32 whatever their size - the channel equalizers (exported as dB tables)
FLOAT_NAMES = ('eq_free', 'eq_table')

# GPTQ's damping - this share of H's mean diagonal added to its diagonal
DAMP = 0.01


# Whether a tensor (state name) is quantized
def is_quantized(tensor, name=None):
    return tensor.dim() >= 2 and tensor.numel() >= MIN_QUANTIZED and name not in FLOAT_NAMES


# GPTQ's integers of rows [rows, columns] at their scales [rows] (float64) - rows sharing the input covariance hessian
# [columns, columns] (float64); a scale of 0 gives a row of zeros. The integers are within +-limit (int64).
def gptq(rows, scales, hessian, limit):
    hessian = hessian.copy()
    dead = numpy.diag(hessian) == 0
    hessian[dead, dead] = 1.0
    rows = rows.copy()
    rows[:, dead] = 0.0
    hessian += DAMP * numpy.mean(numpy.diag(hessian)) * numpy.eye(len(hessian))
    upper = numpy.linalg.cholesky(numpy.linalg.inv(hessian)).T
    safe = numpy.where(scales > 0, scales, 1.0)
    integers = numpy.zeros(rows.shape, dtype=numpy.int64)
    for column in range(rows.shape[1]):
        values = rows[:, column]
        integers[:, column] = numpy.clip(numpy.round(values / safe), -limit, limit) * (scales > 0)
        error = (values - integers[:, column] * scales) / upper[column, column]
        rows[:, column + 1:] -= numpy.outer(error, upper[column, column + 1:])
    return integers


# Quantize a tensor (a CPU torch tensor, [rows, ...]) per row - (int8 values [rows, row length], float32 scales [rows]).
# covariance - None (nearest rounding), the layer's input covariance [row length, row length] (float64), or one per
# row [rows, row length, row length] (a depthwise convolution's channels, each its own window's)
def quantize(tensor, covariance=None):
    rows = tensor.reshape(tensor.shape[0], -1).numpy().astype(numpy.float64)
    scales = numpy.abs(rows).max(axis=1) / LIMIT
    scales[scales == 0] = 1.0
    if covariance is None:
        values = numpy.clip(numpy.round(rows / scales[:, None]), -LIMIT, LIMIT)
    elif covariance.ndim == 3:
        values = numpy.concatenate([gptq(rows[row:row + 1], scales[row:row + 1], covariance[row], LIMIT) for row in range(len(rows))])
    else:
        values = gptq(rows, scales, covariance, LIMIT)
    return values.astype(numpy.int8), scales.astype(numpy.float32)


# The dequantized weights of quantize's values and scales, shaped like the tensor (float32, as the loader computes them:
# float32 value times float32 scale)
def dequantize(values, scales, shape):
    return (values.astype(numpy.float32) * scales[:, None]).reshape(shape)


# The packed bytes of quantize's values - each value + LIMIT as BITS bits, little-endian bit by bit
def pack(values):
    unsigned = (values.reshape(-1).astype(numpy.int64) + LIMIT).astype(numpy.uint8)
    stream = ((unsigned[:, None] >> numpy.arange(BITS)) & 1).astype(numpy.uint8).reshape(-1)
    return numpy.packbits(stream, bitorder='little').tobytes()
