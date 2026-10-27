# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
Weight quantization - the one definition shared by the export (export.py: the exported weights and the PyTorch model
with them, quantized_model) and the JavaScript loader (lib/model.js):

- The quantized tensors (is_quantized) are the matrices, convolution kernels, and embedding tables of at least
  MIN_QUANTIZED values, not the channel equalizers (exported as dB tables).
- 8 bits, symmetric, per row (a tensor's output channel - dim 0, the rest flattened): scale = max |w| / 127 (computed
  in float64, stored float32), q = round(w / scale) clipped to +-127 (int8), w' = q x scale (float32, as the loader
  computes it). A row of zeros has scale 1.
"""

import numpy


# Tensors smaller than this stay float32
MIN_QUANTIZED = 1024

# The parameters (state names) that stay float32 whatever their size - the channel equalizers (exported as dB tables)
FLOAT_NAMES = ('eq_free', 'eq_table')


# Whether a tensor (state name) is quantized
def is_quantized(tensor, name=None):
    return tensor.dim() >= 2 and tensor.numel() >= MIN_QUANTIZED and name not in FLOAT_NAMES


# Quantize a tensor (a CPU torch tensor, [rows, ...]) per row - (int8 values [rows, row length], float32 scales [rows])
def quantize(tensor):
    rows = tensor.reshape(tensor.shape[0], -1).numpy().astype(numpy.float64)
    scales = numpy.abs(rows).max(axis=1) / 127
    scales[scales == 0] = 1.0
    values = numpy.clip(numpy.round(rows / scales[:, None]), -127, 127).astype(numpy.int8)
    return values, scales.astype(numpy.float32)


# The dequantized weights of quantize's values and scales, shaped like the tensor (float32, as the loader computes them:
# float32 value times float32 scale)
def dequantize(values, scales, shape):
    return (values.astype(numpy.float32) * scales[:, None]).reshape(shape)
