# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
Weight quantization - the one definition shared by the export (export.py), quantization-aware training (QAT,
trainmulti.py --qat-bits), and the JavaScript loader (lib/model.js):

- Symmetric, per row (a tensor's output channel - dim 0, the rest flattened) or per group (runs of `group` consecutive
  weights within a row, the last run shorter): scale = max |w| / QMAX, q = round(w / scale) clipped to +-QMAX (QMAX =
  2^(bits - 1) - 1), w' = q x scale. A row or group of zeros has scale 1.
- 8 bits per row is the original format (export.py 'int8': int8 values, float32 scales computed in float64). Every
  other bit width or grouping is the 'packed' format: the values offset to unsigned (q + 2^(bits - 1)) and packed
  little-endian, LSB first, `bits` bits each, row-major; the scales float16 (row-major [rows, groups per row]) - the
  scale is rounded to float16 before the values are computed, so the dequantized weights are exactly what the loader
  computes.
- QAT fake-quantizes the same tensors (is_quantized - matrices, convolution kernels, and embedding tables of at least
  MIN_QUANTIZED values, not the channel equalizers) the same way, with the straight-through estimator: the forward pass
  sees w', the gradient reaches w unchanged.
"""

import contextlib

import numpy
import torch


# Tensors smaller than this stay float32
MIN_QUANTIZED = 1024

# The parameters (state names) that stay float32 whatever their size - the channel equalizers (exported as dB tables)
FLOAT_NAMES = ('eq_free', 'eq_table')


# The largest quantized magnitude of a bit width
def qmax(bits):
    return (1 << (bits - 1)) - 1


# Whether a tensor (state name) is quantized
def is_quantized(tensor, name=None):
    return tensor.dim() >= 2 and tensor.numel() >= MIN_QUANTIZED and name not in FLOAT_NAMES


# Whether a bit width and grouping is the original 8-bit per-row format
def is_int8(bits, group):
    return bits == 8 and not group


# The group size of a row length (0 - per row - is the whole row)
def group_size(row_length, group):
    return row_length if not group or group >= row_length else group


# Quantize a tensor (numpy-convertible, [rows, ...]) - (integer values [rows, row length] int32, scales [rows, groups]
# float64 as stored - float32 for 8 bits per row, float16 otherwise)
def quantize_array(tensor, bits=8, group=0):
    rows = numpy.asarray(tensor, dtype=numpy.float64).reshape(tensor.shape[0], -1)
    length = rows.shape[1]
    size = group_size(length, group)
    count = -(-length // size)
    padded = numpy.zeros((rows.shape[0], count * size))
    padded[:, :length] = rows
    scales = numpy.abs(padded.reshape((rows.shape[0], count, size))).max(axis=2) / qmax(bits)
    scales[scales == 0] = 1.0
    if is_int8(bits, group):
        values = numpy.clip(numpy.round(rows / scales), -127, 127).astype(numpy.int32)
        return values, scales.astype(numpy.float32).astype(numpy.float64)
    scales = scales.astype(numpy.float16).astype(numpy.float64)
    scales[scales == 0] = 1.0
    per_value = numpy.repeat(scales, size, axis=1)[:, :length]
    values = numpy.clip(numpy.round(rows / per_value), -qmax(bits), qmax(bits)).astype(numpy.int32)
    return values, scales


# The dequantized weights of quantize_array's values and scales, shaped like the tensor (float32, as the loader
# computes them: float32 value times float32 scale)
def dequantize_array(values, scales, shape, group=0):
    length = values.shape[1]
    size = group_size(length, group)
    per_value = numpy.repeat(scales.astype(numpy.float32), size, axis=1)[:, :length]
    return (values.astype(numpy.float32) * per_value).reshape(shape)


# Pack integer values (|q| <= QMAX) as unsigned bits-wide fields, little-endian, LSB first - bytes
def pack(values, bits):
    unsigned = (numpy.asarray(values, dtype=numpy.int64).reshape(-1) + (1 << (bits - 1))).astype(numpy.uint64)
    bit_matrix = ((unsigned[:, None] >> numpy.arange(bits, dtype=numpy.uint64)[None, :]) & 1).astype(numpy.uint8)
    return numpy.packbits(bit_matrix.reshape(-1), bitorder='little').tobytes()


# Unpack pack's bytes - integer values [count]
def unpack(data, bits, count):
    bit_array = numpy.unpackbits(numpy.frombuffer(data, dtype=numpy.uint8), bitorder='little')[:count * bits]
    unsigned = (bit_array.reshape(count, bits).astype(numpy.int64) << numpy.arange(bits)[None, :]).sum(axis=1)
    return unsigned - (1 << (bits - 1))


#
# Quantization-aware training
#

# A tensor fake-quantized (bits, group - as quantize_array) with the straight-through estimator: the forward value is the
# dequantized weight, the gradient passes to the tensor unchanged
def fake_quantize(tensor, bits=8, group=0):
    rows = tensor.reshape(tensor.shape[0], -1)
    length = rows.shape[1]
    size = group_size(length, group)
    count = -(-length // size)
    with torch.no_grad():
        padded = torch.nn.functional.pad(rows.detach(), (0, count * size - length))
        scales = padded.abs().reshape(rows.shape[0], count, size).amax(dim=2) / qmax(bits)
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        if not is_int8(bits, group):
            scales = scales.half().float()
            scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        per_value = scales.repeat_interleave(size, dim=1)[:, :length]
        quantized = torch.clamp(torch.round(rows.detach() / per_value), -qmax(bits), qmax(bits)) * per_value
    return tensor + (quantized.reshape(tensor.shape) - tensor).detach()


@contextlib.contextmanager
def fake_quantized(model, bits=8, group=0):
    """Within the context, the model's quantized parameters (is_quantized) are their fake-quantized versions - the
    module attributes swapped for the straight-through tensors, the parameters themselves (the optimizer's) untouched"""
    swapped = []
    for module_name, module in model.named_modules():
        for name, parameter in list(module._parameters.items()):  # pylint: disable=protected-access
            full_name = f'{module_name}.{name}' if module_name else name
            if parameter is not None and is_quantized(parameter, full_name):
                swapped.append((module, name, parameter))
                module._parameters[name] = fake_quantize(parameter, bits, group)  # pylint: disable=protected-access
    try:
        yield
    finally:
        for module, name, parameter in swapped:
            module._parameters[name] = parameter  # pylint: disable=protected-access
