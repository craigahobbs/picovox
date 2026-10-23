// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// The neural voice - picovox_train/model.py's MultiModel forward pass in plain JavaScript (no dependencies, browser and
// Node): tokens (lib/tokens.js) to durations to synthesizer parameter frames (picovox_train/params.py), which
// lib/render.js renders.
//
//     import {loadModel, speakFrames} from './lib/model.js';
//     const model = loadModel(manifest, buffer);       // an export's <voice>.json (parsed) and <voice>.bin (ArrayBuffer)
//     const {frames, eq} = speakFrames(model, 'Hello, world.', {'speaker': 5});
//     const samples = render(frames, {'sampleRate': 44100, eq});
//
// The weights are 8-bit (per-row scales, picovox_train/export.py) - or, exported at lower precision, packed 4-7-bit (or 8-bit
// per-group) values with float16 scales per row or per group (picovox_train/quant.py) - dequantized on load; the arithmetic is
// float64.
//
// Long texts are spoken a sentence at a time (as the training data is - one or two sentences), each with its own <bos>.
//
// The model (lib/models/voiceModel.js) takes a voice and a style:
//
//     const {frames, eq} = speakFrames(model, 'Hello, world.', {
//         'speaker': 5,                                   // a speaker id (manifest.speakers), or a blend
//         // 'speaker': {'5': 0.6, '12': 0.4},            //   {id: weight} (normalized to sum to 1)
//         'style': {'emotion': 'happy', 'intensity': 0.7} // an emotion (manifest.styles, or 'neutral') at 0-1
//         // 'style': {'happy': 0.5, 'sad': 0.2}          //   or a blend {emotion: weight}
//     });

import {NAMES, PARAMETERS} from './params.js';
import {tokenize} from './tokens.js';


// The token classes (the manifest's vocab - picovox_train/vocab.json, picovox_train/model.py)
const PAD = 0;
const BOS = 2;
const WORD = 3;
const FIRST_PHONE = 9;
const SENTENCE_ENDS = new Set(['<period>', '<question>', '<wh>', '<exclamation>']);
const LAYER_NORM_EPS = 1e-5;

// A boundary token pauses when its pause probability is above this
const PAUSE_THRESHOLD = 0.5;


// A float16 value (its bits) as a number
function halfToFloat(bits) {
    const exponent = (bits >> 10) & 0x1f;
    const fraction = bits & 0x3ff;
    const sign = bits & 0x8000 ? -1 : 1;
    if (exponent === 0) {
        return sign * fraction * 2 ** -24;
    }
    if (exponent === 0x1f) {
        return fraction ? NaN : sign * Infinity;
    }
    return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}


// Dequantize a packed tensor (picovox_train/quant.py) into data - bits-wide unsigned values (little-endian, LSB first, offset by
// 2^(bits - 1)) times float16 scales per row or per group of entry.group values
function dequantizePacked(entry, buffer, data) {
    const [rows] = entry.shape;
    const rowSize = data.length / rows;
    const {bits, group} = entry;
    const groups = Math.ceil(rowSize / group);
    const bytes = new Uint8Array(buffer, entry.offset, Math.ceil(data.length * bits / 8));
    const scaleBits = new Uint16Array(buffer.slice(entry.scaleOffset, entry.scaleOffset + 2 * rows * groups));
    const scales = Float32Array.from(scaleBits, halfToFloat);
    const offset = 2 ** (bits - 1);
    const mask = 2 ** bits - 1;
    let bitPosition = 0;
    for (let row = 0; row < rows; row++) {
        for (let ix = 0; ix < rowSize; ix++) {
            // (a field spans at most two bytes for bits <= 8 - read a 16-bit window)
            const byte = bitPosition >> 3;
            const window = bytes[byte] | ((byte + 1 < bytes.length ? bytes[byte + 1] : 0) << 8);
            const value = ((window >> (bitPosition & 7)) & mask) - offset;
            bitPosition += bits;
            data[row * rowSize + ix] = Math.fround(value * scales[row * groups + Math.floor(ix / group)]);
        }
    }
}


// Load a model from its manifest (an export's <voice>.json, parsed) and weights (its <voice>.bin, an ArrayBuffer)
export function loadModel(manifest, buffer) {
    const tensors = {};
    for (const entry of manifest.tensors) {
        const size = entry.shape.reduce((product, dim) => product * dim, 1);
        const data = new Float64Array(size);
        if (entry.encoding === 'packed') {
            dequantizePacked(entry, buffer, data);
        } else if (entry.encoding === 'int8') {
            const [rows] = entry.shape;
            const rowSize = size / rows;
            const values = new Int8Array(buffer, entry.offset, size);
            const scales = new Float32Array(buffer.slice(entry.scaleOffset, entry.scaleOffset + 4 * rows));
            for (let row = 0; row < rows; row++) {
                for (let ix = 0; ix < rowSize; ix++) {
                    data[row * rowSize + ix] = values[row * rowSize + ix] * scales[row];
                }
            }
        } else {
            data.set(new Float32Array(buffer.slice(entry.offset, entry.offset + 4 * size)));
        }
        tensors[entry.name] = {data, 'shape': entry.shape};
    }
    const remap = new Map();
    for (const [from, to] of Object.entries(manifest.remap ?? {})) {
        remap.set(manifest.vocab.indexOf(from), manifest.vocab.indexOf(to));
    }
    return {manifest, 'config': manifest.config, tensors, remap};
}


// The tensor of a name
function weight(model, name) {
    const tensor = model.tensors[name];
    if (tensor === undefined) {
        throw new Error(`model tensor ${name} not found`);
    }
    return tensor;
}


//
// Layers - activations are matrices {rows, cols, data} (Float64Array, row-major - a row per time step)
//

// A matrix of zeros
function zeros(rows, cols) {
    return {rows, cols, 'data': new Float64Array(rows * cols)};
}


// A linear layer (weight [out, in], bias [out]) of a matrix's rows. Four rows at a time share each pass over a weight row,
// so the weights are read from memory a quarter as often (phones are memory-bound here); each output's sum is formed in
// the same order as one row at a time (four interleaved partial sums), so the results are identical.
function linear(model, prefix, input) {
    const {'data': matrix, shape} = weight(model, `${prefix}.weight`);
    const bias = weight(model, `${prefix}.bias`).data;
    const [outSize, inSize] = shape;
    const output = zeros(input.rows, outSize);
    const inData = input.data;
    const outData = output.data;
    const blockRows = input.rows - (input.rows % 4);
    for (let row = 0; row < blockRows; row += 4) {
        const inBase0 = row * inSize;
        const inBase1 = inBase0 + inSize;
        const inBase2 = inBase1 + inSize;
        const inBase3 = inBase2 + inSize;
        const outBase = row * outSize;
        for (let outIx = 0; outIx < outSize; outIx++) {
            const base = outIx * inSize;
            let a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0;
            let c0 = 0, c1 = 0, c2 = 0, c3 = 0, d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            let inIx = 0;
            for (; inIx + 7 < inSize; inIx += 8) {
                const w0 = matrix[base + inIx];
                const w1 = matrix[base + inIx + 1];
                const w2 = matrix[base + inIx + 2];
                const w3 = matrix[base + inIx + 3];
                const w4 = matrix[base + inIx + 4];
                const w5 = matrix[base + inIx + 5];
                const w6 = matrix[base + inIx + 6];
                const w7 = matrix[base + inIx + 7];
                const x0 = inBase0 + inIx;
                const x1 = inBase1 + inIx;
                const x2 = inBase2 + inIx;
                const x3 = inBase3 + inIx;
                a0 += w0 * inData[x0];
                a1 += w1 * inData[x0 + 1];
                a2 += w2 * inData[x0 + 2];
                a3 += w3 * inData[x0 + 3];
                b0 += w0 * inData[x1];
                b1 += w1 * inData[x1 + 1];
                b2 += w2 * inData[x1 + 2];
                b3 += w3 * inData[x1 + 3];
                c0 += w0 * inData[x2];
                c1 += w1 * inData[x2 + 1];
                c2 += w2 * inData[x2 + 2];
                c3 += w3 * inData[x2 + 3];
                d0 += w0 * inData[x3];
                d1 += w1 * inData[x3 + 1];
                d2 += w2 * inData[x3 + 2];
                d3 += w3 * inData[x3 + 3];
                a0 += w4 * inData[x0 + 4];
                a1 += w5 * inData[x0 + 5];
                a2 += w6 * inData[x0 + 6];
                a3 += w7 * inData[x0 + 7];
                b0 += w4 * inData[x1 + 4];
                b1 += w5 * inData[x1 + 5];
                b2 += w6 * inData[x1 + 6];
                b3 += w7 * inData[x1 + 7];
                c0 += w4 * inData[x2 + 4];
                c1 += w5 * inData[x2 + 5];
                c2 += w6 * inData[x2 + 6];
                c3 += w7 * inData[x2 + 7];
                d0 += w4 * inData[x3 + 4];
                d1 += w5 * inData[x3 + 5];
                d2 += w6 * inData[x3 + 6];
                d3 += w7 * inData[x3 + 7];
            }
            for (; inIx + 3 < inSize; inIx += 4) {
                const w0 = matrix[base + inIx];
                const w1 = matrix[base + inIx + 1];
                const w2 = matrix[base + inIx + 2];
                const w3 = matrix[base + inIx + 3];
                a0 += w0 * inData[inBase0 + inIx];
                a1 += w1 * inData[inBase0 + inIx + 1];
                a2 += w2 * inData[inBase0 + inIx + 2];
                a3 += w3 * inData[inBase0 + inIx + 3];
                b0 += w0 * inData[inBase1 + inIx];
                b1 += w1 * inData[inBase1 + inIx + 1];
                b2 += w2 * inData[inBase1 + inIx + 2];
                b3 += w3 * inData[inBase1 + inIx + 3];
                c0 += w0 * inData[inBase2 + inIx];
                c1 += w1 * inData[inBase2 + inIx + 1];
                c2 += w2 * inData[inBase2 + inIx + 2];
                c3 += w3 * inData[inBase2 + inIx + 3];
                d0 += w0 * inData[inBase3 + inIx];
                d1 += w1 * inData[inBase3 + inIx + 1];
                d2 += w2 * inData[inBase3 + inIx + 2];
                d3 += w3 * inData[inBase3 + inIx + 3];
            }
            for (; inIx < inSize; inIx++) {
                const w0 = matrix[base + inIx];
                a0 += w0 * inData[inBase0 + inIx];
                b0 += w0 * inData[inBase1 + inIx];
                c0 += w0 * inData[inBase2 + inIx];
                d0 += w0 * inData[inBase3 + inIx];
            }
            outData[outBase + outIx] = bias[outIx] + a0 + a1 + a2 + a3;
            outData[outBase + outSize + outIx] = bias[outIx] + b0 + b1 + b2 + b3;
            outData[outBase + 2 * outSize + outIx] = bias[outIx] + c0 + c1 + c2 + c3;
            outData[outBase + 3 * outSize + outIx] = bias[outIx] + d0 + d1 + d2 + d3;
        }
    }
    for (let row = blockRows; row < input.rows; row++) {
        const inBase = row * inSize;
        const outBase = row * outSize;
        for (let outIx = 0; outIx < outSize; outIx++) {
            const base = outIx * inSize;
            let sum0 = 0;
            let sum1 = 0;
            let sum2 = 0;
            let sum3 = 0;
            let inIx = 0;
            for (; inIx + 3 < inSize; inIx += 4) {
                sum0 += matrix[base + inIx] * inData[inBase + inIx];
                sum1 += matrix[base + inIx + 1] * inData[inBase + inIx + 1];
                sum2 += matrix[base + inIx + 2] * inData[inBase + inIx + 2];
                sum3 += matrix[base + inIx + 3] * inData[inBase + inIx + 3];
            }
            for (; inIx < inSize; inIx++) {
                sum0 += matrix[base + inIx] * inData[inBase + inIx];
            }
            outData[outBase + outIx] = bias[outIx] + sum0 + sum1 + sum2 + sum3;
        }
    }
    return output;
}


// Layer norm over each row's channels
function layerNorm(model, prefix, input) {
    const gain = weight(model, `${prefix}.weight`).data;
    const bias = weight(model, `${prefix}.bias`).data;
    const {rows, cols} = input;
    const output = zeros(rows, cols);
    const inData = input.data;
    const outData = output.data;
    for (let row = 0; row < rows; row++) {
        const base = row * cols;
        let mean = 0;
        for (let ix = 0; ix < cols; ix++) {
            mean += inData[base + ix];
        }
        mean /= cols;
        let variance = 0;
        for (let ix = 0; ix < cols; ix++) {
            variance += (inData[base + ix] - mean) ** 2;
        }
        const scale = 1 / Math.sqrt(variance / cols + LAYER_NORM_EPS);
        for (let ix = 0; ix < cols; ix++) {
            outData[base + ix] = (inData[base + ix] - mean) * scale * gain[ix] + bias[ix];
        }
    }
    return output;
}


// A depthwise convolution over time (weight [C, 1, K], 'same' zero padding)
function depthwiseConv(model, prefix, input, start = 0, end = input.rows) {
    const {'data': kernel, shape} = weight(model, `${prefix}.conv.weight`);
    const bias = weight(model, `${prefix}.conv.bias`).data;
    const [channels, , size] = shape;
    const half = Math.floor(size / 2);
    const output = zeros(end - start, channels);
    const outData = output.data;
    const inData = input.data;
    const inRows = input.rows;
    for (let row = start; row < end; row++) {
        const outBase = (row - start) * channels;
        const firstTap = Math.max(0, half - row);
        const endTap = Math.min(size, inRows - row + half);
        for (let channel = 0; channel < channels; channel++) {
            let sum = bias[channel];
            const kernelBase = channel * size;
            for (let tap = firstTap, inIx = (row + firstTap - half) * channels + channel; tap < endTap; tap++, inIx += channels) {
                sum += kernel[kernelBase + tap] * inData[inIx];
            }
            outData[outBase + channel] = sum;
        }
    }
    return output;
}


// SiLU, in place
function silu(input) {
    const {data} = input;
    for (let ix = 0; ix < data.length; ix++) {
        data[ix] /= 1 + Math.exp(-data[ix]);
    }
    return input;
}


// Add matrices (b to a, in place)
function addRows(inputA, inputB) {
    for (let ix = 0; ix < inputA.data.length; ix++) {
        inputA.data[ix] += inputB.data[ix];
    }
    return inputA;
}


// A conv-FFN block - x + W2 SiLU(DW(SiLU(W1 FiLM(LN(x))))), FiLM's [gamma, beta] the condition's
function convFFN(model, prefix, input, filmVectors) {
    const normed = layerNorm(model, `${prefix}.norm`, input);
    const hidden = silu(linear(model, `${prefix}.inner`, film(normed, filmVectors)));
    const convolved = silu(depthwiseConv(model, `${prefix}.conv`, hidden));
    return addRows(input, linear(model, `${prefix}.outer`, convolved));
}


// A multi-head self-attention block - x + Wo attention(LN(x))
function selfAttention(model, prefix, input) {
    const {heads} = model.config;
    const dim = input.cols;
    const length = input.rows;
    const headDim = dim / heads;
    const qkv = linear(model, `${prefix}.qkv`, layerNorm(model, `${prefix}.norm`, input)).data;
    const attended = zeros(length, dim);
    const scale = 1 / Math.sqrt(headDim);
    const scores = new Float64Array(length);
    const attendedData = attended.data;
    const stride = 3 * dim;
    const blockKeys = length - (length % 4);
    for (let head = 0; head < heads; head++) {
        const offset = head * headDim;
        for (let query = 0; query < length; query++) {
            const queryBase = query * stride + offset;
            let max = -Infinity;
            // The scores - four keys per pass over the query (each sum's terms in order)
            let key = 0;
            for (; key < blockKeys; key += 4) {
                const keyBase0 = key * stride + dim + offset;
                const keyBase1 = keyBase0 + stride;
                const keyBase2 = keyBase1 + stride;
                const keyBase3 = keyBase2 + stride;
                let sum0 = 0;
                let sum1 = 0;
                let sum2 = 0;
                let sum3 = 0;
                for (let ix = 0; ix < headDim; ix++) {
                    const queryValue = qkv[queryBase + ix];
                    sum0 += queryValue * qkv[keyBase0 + ix];
                    sum1 += queryValue * qkv[keyBase1 + ix];
                    sum2 += queryValue * qkv[keyBase2 + ix];
                    sum3 += queryValue * qkv[keyBase3 + ix];
                }
                const score0 = sum0 * scale;
                const score1 = sum1 * scale;
                const score2 = sum2 * scale;
                const score3 = sum3 * scale;
                scores[key] = score0;
                scores[key + 1] = score1;
                scores[key + 2] = score2;
                scores[key + 3] = score3;
                max = Math.max(max, score0, score1, score2, score3);
            }
            for (; key < length; key++) {
                const keyBase = key * stride + dim + offset;
                let sum = 0;
                for (let ix = 0; ix < headDim; ix++) {
                    sum += qkv[queryBase + ix] * qkv[keyBase + ix];
                }
                scores[key] = sum * scale;
                max = Math.max(max, scores[key]);
            }
            let total = 0;
            for (key = 0; key < length; key++) {
                const exp = Math.exp(scores[key] - max);
                scores[key] = exp;
                total += exp;
            }
            // The values - four keys per pass over the output (each output's terms in key order)
            const outBase = query * dim + offset;
            for (key = 0; key < blockKeys; key += 4) {
                const share0 = scores[key] / total;
                const share1 = scores[key + 1] / total;
                const share2 = scores[key + 2] / total;
                const share3 = scores[key + 3] / total;
                const valueBase0 = key * stride + 2 * dim + offset;
                const valueBase1 = valueBase0 + stride;
                const valueBase2 = valueBase1 + stride;
                const valueBase3 = valueBase2 + stride;
                for (let ix = 0; ix < headDim; ix++) {
                    let value = attendedData[outBase + ix];
                    value += share0 * qkv[valueBase0 + ix];
                    value += share1 * qkv[valueBase1 + ix];
                    value += share2 * qkv[valueBase2 + ix];
                    value += share3 * qkv[valueBase3 + ix];
                    attendedData[outBase + ix] = value;
                }
            }
            for (; key < length; key++) {
                const share = scores[key] / total;
                const valueBase = key * stride + 2 * dim + offset;
                for (let ix = 0; ix < headDim; ix++) {
                    attendedData[outBase + ix] += share * qkv[valueBase + ix];
                }
            }
        }
    }
    return addRows(input, linear(model, `${prefix}.out`, attended));
}


// Round half to even (as torch.round)
function roundHalfEven(value) {
    const rounded = Math.round(value);
    return Math.abs(value % 1) === 0.5 && rounded % 2 !== 0 ? rounded - 1 : rounded;
}


//
// The conditioning - the speaker and the style
//

// A matrix of one row
function rowMatrix(values) {
    return {'rows': 1, 'cols': values.length, 'data': Float64Array.from(values)};
}


// A speaker blend's normalized weights - [[id, weight], ...] - from an id or an {id: weight} object
function speakerWeights(model, speaker) {
    const {speakers} = model.config;
    const isBlend = typeof speaker === 'object' && speaker !== null;
    const entries = isBlend ? Object.entries(speaker).map(([id, share]) => [Number(id), Number(share)]) :
        [[Number(speaker ?? model.manifest.defaultSpeaker), 1]];
    const total = entries.reduce((sum, [, share]) => sum + share, 0);
    if (!(total > 0) || entries.some(([id, share]) => !Number.isInteger(id) || id < 0 || id >= speakers || share < 0)) {
        throw new Error(`invalid speaker ${JSON.stringify(speaker)}`);
    }
    return entries.map(([id, share]) => [id, share / total]);
}


// A style's weights over the style table's rows - from {emotion, intensity} or an {emotion: weight} blend (neutral has
// no row - it is the zero style)
function styleWeights(model, style) {
    const names = model.manifest.styles;
    const weights = new Float64Array(names.length);
    if (style === undefined || style === null) {
        return weights;
    }
    const entries = 'emotion' in style ? [[style.emotion, style.intensity ?? 1]] : Object.entries(style);
    for (const [emotion, share] of entries) {
        if (emotion !== 'neutral') {
            const index = names.indexOf(emotion);
            if (index < 0) {
                throw new Error(`unknown style ${emotion}`);
            }
            weights[index] += Number(share);
        }
    }
    return weights;
}


// A voice - {condition (the condition vector [speaker embedding, style embedding]), eq (the speakers' channel equalizer,
// dB)} - see speakFrames for the options
function voiceCondition(model, {speaker, style} = {}) {
    const {speakerDim, styleDim} = model.config;
    const speakerTable = weight(model, 'speaker.weight').data;
    const eqTable = weight(model, 'eq_table').data;
    const [, bands] = weight(model, 'eq_table').shape;
    const condition = new Float64Array(speakerDim + styleDim);
    const voiceEq = new Array(bands).fill(0);
    for (const [id, share] of speakerWeights(model, speaker)) {
        for (let ix = 0; ix < speakerDim; ix++) {
            condition[ix] += share * speakerTable[id * speakerDim + ix];
        }
        for (let ix = 0; ix < bands; ix++) {
            voiceEq[ix] += share * eqTable[id * bands + ix];
        }
    }
    const styleTable = weight(model, 'style').data;
    const weights = styleWeights(model, style);
    for (const [row, share] of weights.entries()) {
        for (let ix = 0; ix < styleDim; ix++) {
            condition[speakerDim + ix] += share * styleTable[row * styleDim + ix];
        }
    }
    return {condition, 'eq': voiceEq};
}


// The condition's projections - {encoder, duration, decoder (added vectors), films: {prefix: [gamma, beta]}}
function conditionProjections(model, condition) {
    const input = rowMatrix(condition);
    const projections = {
        'encoder': linear(model, 'condition_encoder', input).data,
        'duration': linear(model, 'condition_duration', input).data,
        'decoder': linear(model, 'condition_decoder', input).data,
        'films': {}
    };
    const prefixes = [];
    for (let layer = 0; layer < 2 * model.config.encoderLayers; layer += 2) {
        prefixes.push(`encoder.${layer}`);
    }
    for (let layer = 0; layer < model.config.decoderLayers; layer++) {
        prefixes.push(`decoder.${layer}`);
    }
    for (const prefix of prefixes) {
        const filmVector = linear(model, `${prefix}.film`, input).data;
        const half = filmVector.length / 2;
        projections.films[prefix] = [filmVector.subarray(0, half), filmVector.subarray(half)];
    }
    return projections;
}


// Add a vector to every row of a matrix, in place
function addVector(input, vector) {
    for (let ix = 0; ix < input.data.length; ix++) {
        input.data[ix] += vector[ix % input.cols];
    }
    return input;
}


// FiLM - each row times (1 + gamma) plus beta, in place
function film(input, [gamma, beta]) {
    for (let ix = 0; ix < input.data.length; ix++) {
        const channel = ix % input.cols;
        input.data[ix] = input.data[ix] * (1 + gamma[channel]) + beta[channel];
    }
    return input;
}


// Token ids to token features (rows [N][D]) - the token embeddings plus the condition's projection, through the encoder
// (FiLM by the condition)
function encode(model, ids, projections) {
    const embedding = weight(model, 'embedding.weight');
    const added = projections.encoder;
    const [, width] = embedding.shape;
    let rows = zeros(ids.length, width);
    for (const [index, id] of ids.entries()) {
        for (let ix = 0; ix < width; ix++) {
            rows.data[index * width + ix] = embedding.data[id * width + ix] + added[ix];
        }
    }
    for (let layer = 0; layer < 2 * model.config.encoderLayers; layer++) {
        const prefix = `encoder.${layer}`;
        rows = layer % 2 === 0 ? convFFN(model, prefix, rows, projections.films[prefix]) : selfAttention(model, prefix, rows);
    }
    return layerNorm(model, 'encoder_norm', rows);
}


// Predicted durations (10 ms frames) of token features
function predictDurations(model, ids, features, {rate = 1}, projections) {
    let rows = addVector({'rows': features.rows, 'cols': features.cols, 'data': Float64Array.from(features.data)}, projections.duration);
    for (let layer = 0; layer < 2; layer++) {
        const prefix = `duration_layers.${layer}`;
        rows = layerNorm(model, `${prefix}.norm`, silu(linear(model, `${prefix}.linear`, depthwiseConv(model, `${prefix}.conv`, rows))));
    }
    const outputs = linear(model, 'duration_out', rows);
    const durations = [];
    let total = 0;
    let previousEnd = 0;
    for (const [index, id] of ids.entries()) {
        const [logDuration, pauseLogit, logPaused] = outputs.data.subarray(3 * index, 3 * index + 3);
        let duration;
        if ((id >= WORD && id < FIRST_PHONE) || id === BOS) {
            duration = 1 / (1 + Math.exp(-pauseLogit)) > PAUSE_THRESHOLD ? Math.max(1, Math.expm1(logPaused)) * rate : 0;
        } else {
            duration = Math.max(1, Math.expm1(logDuration)) * rate;
        }
        total += id === PAD ? 0 : duration;
        const end = roundHalfEven(total);
        durations.push(end - previousEnd);
        previousEnd = end;
    }
    return durations;
}


// Rows [start, end) of a matrix, as a matrix sharing its data
function rowRange(input, start, end) {
    return {'rows': end - start, 'cols': input.cols, 'data': input.data.subarray(start * input.cols, end * input.cols)};
}


// Decode token features and durations to parameter frames (Float64Array rows of picovox_train/params.py's parameters, 5 ms),
// yielding them a chunk at a time (chunkFrames 10 ms frames). The decoder is per-frame but for each layer's depthwise
// convolution, so each layer runs only as far ahead as the layer above it needs (half a kernel), and every frame is
// computed once, as in a whole-sequence decode - the chunks are exactly its frames.
function* decodeChunks(model, features, durations, projections, chunkFrames = 32) {
    const {config} = model;
    const frameCount = durations.reduce((sum, duration) => sum + duration, 0);
    if (frameCount === 0) {
        return;
    }
    // (the decoder's input layer of each token - the same for every one of its frames - repeated for the frames)
    const tokenInput = linear(model, 'decoder_in', features);
    const inputDim = tokenInput.cols;
    const regulated = zeros(frameCount, inputDim);
    const positions = zeros(frameCount, 5);
    let frame = 0;
    for (const [index, duration] of durations.entries()) {
        for (let offset = 0; offset < duration; offset++) {
            regulated.data.set(tokenInput.data.subarray(index * inputDim, (index + 1) * inputDim), frame * inputDim);
            positions.data.set([
                (offset + 0.5) / duration, 1 - (offset + 0.5) / duration, Math.log1p(duration) / 3,
                Math.min(offset, 20) / 20, Math.min(duration - 1 - offset, 20) / 20
            ], frame * 5);
            frame++;
        }
    }
    const speakerVector = projections.decoder;
    const layers = config.decoderLayers;
    const half = Math.floor(config.decoderKernel / 2);

    // Each layer's input rows (the residual stream - layers + 1 of them, the last the decoder's output) and inner rows
    // (its convolution's input), and how many of each are computed
    const streams = Array.from({'length': layers + 1}, () => ({'matrix': null, 'done': 0}));
    const inners = Array.from({'length': layers}, () => ({'matrix': null, 'done': 0}));
    const store = (state, rows, start) => {
        if (state.matrix === null) {
            state.matrix = zeros(frameCount, rows.cols);
        }
        state.matrix.data.set(rows.data, start * rows.cols);
        state.done = start + rows.rows;
    };

    // Compute layer input rows up to end - the inner rows (and so the layer below) half a kernel further
    const ensureStream = (layer, end) => {
        // (a layer is only asked for rows past those it has - its inner rows, the layer above's input, run with it)
        const state = streams[layer];
        const start = state.done;
        if (layer === 0) {
            const regulatedRows = rowRange(regulated, start, end);
            const rows = addRows({'rows': regulatedRows.rows, 'cols': regulatedRows.cols, 'data': Float64Array.from(regulatedRows.data)},
                linear(model, 'position_in', rowRange(positions, start, end)));
            store(state, addVector(rows, speakerVector), start);
            return;
        }
        const prefix = `decoder.${layer - 1}`;
        const inner = inners[layer - 1];
        const innerEnd = Math.min(frameCount, end + half);
        if (inner.done < innerEnd) {
            ensureStream(layer - 1, innerEnd);
            const innerStart = inner.done;
            const normed = layerNorm(model, `${prefix}.norm`, rowRange(streams[layer - 1].matrix, innerStart, innerEnd));
            store(inner, silu(linear(model, `${prefix}.inner`, film(normed, projections.films[prefix]))), innerStart);
        }
        const convolved = silu(depthwiseConv(model, `${prefix}.conv`, inner.matrix, start, end));
        const below = rowRange(streams[layer - 1].matrix, start, end);
        const rows = linear(model, `${prefix}.outer`, convolved);
        const rowData = rows.data;
        const belowData = below.data;
        for (let ix = 0; ix < rowData.length; ix++) {
            rowData[ix] = belowData[ix] + rowData[ix];
        }
        store(state, rows, start);
    };

    const count = NAMES.length;
    const isLog = PARAMETERS.map(({scale}) => scale === 'log');
    const lows = PARAMETERS.map(({low}) => low);
    const ranges = PARAMETERS.map(({low, high}) => high - low);
    const logLows = PARAMETERS.map(({low}) => Math.log(low));
    const logRanges = PARAMETERS.map(({low, high}) => Math.log(high) - Math.log(low));
    const jitterIx = NAMES.indexOf('jitter');
    const shimmerIx = NAMES.indexOf('shimmer');
    for (let start = 0; start < frameCount; start += chunkFrames) {
        const end = Math.min(frameCount, start + chunkFrames);
        ensureStream(layers, end);
        const normed = layerNorm(model, 'decoder_norm', rowRange(streams[layers].matrix, start, end));
        const logits = linear(model, 'head', normed);
        const frames = [];
        for (let row = 0; row < logits.rows; row++) {
            for (let sub = 0; sub < config.subframes; sub++) {
                const out = new Float64Array(count);
                const base = row * logits.cols + sub * count;
                const logitData = logits.data;
                for (let ix = 0; ix < count; ix++) {
                    const unit = 1 / (1 + Math.exp(-logitData[base + ix]));
                    out[ix] = isLog[ix] ? Math.exp(logLows[ix] + unit * logRanges[ix]) : lows[ix] + unit * ranges[ix];
                }
                out[jitterIx] = config.jitter;
                out[shimmerIx] = config.shimmer;
                frames.push(out);
            }
        }
        yield frames;
    }
}


// Encode one token sequence and predict its durations - {features, durations}
function inferDurations(model, ids, options, projections) {
    const mapped = ids.map((id) => model.remap.get(id) ?? id);
    const features = encode(model, mapped, projections);
    return {features, 'durations': predictDurations(model, mapped, features, options, projections)};
}


// Split token ids into sentences (by a vocabulary's token names) - each starting with <bos>, ending at a sentence end
export function sentences(ids, vocabulary) {
    const out = [];
    let current = [BOS];
    for (const id of ids.slice(ids[0] === BOS ? 1 : 0)) {
        current.push(id);
        if (SENTENCE_ENDS.has(vocabulary[id])) {
            out.push(current);
            current = [BOS];
        }
    }
    if (current.length > 1) {
        out.push(current);
    }
    return out;
}


// Speak text - {frames (5 ms parameter frames, Float64Array rows, the end frame included), eq (the voice's channel
// equalizer, dB), sentences: [{ids, durations, frameStart}]}. options: rate (a duration multiplier),
// speaker (an id of manifest.speakers, or a blend {id: weight}; default manifest.defaultSpeaker), and style ({emotion,
// intensity} - an emotion of manifest.styles or 'neutral', intensity 0-1 - or a blend {emotion: weight}).
export function speakFrames(model, text, options = {}) {
    const {chunks, eq, sentences: parts} = streamFrames(model, text, options);
    const frames = [];
    for (const chunk of chunks) {
        frames.push(...chunk);
    }
    return {frames, eq, 'sentences': parts};
}


// Speak text as a stream - speakFrames's {eq, sentences (filled as the chunks are made), chunks}, where chunks is an
// iterator of the frames in order, a fraction of a second at a time - the network runs as the chunks are taken
export function streamFrames(model, text, options = {}) {
    const {vocab} = model.manifest;
    const {ids} = tokenize(text, vocab);
    const voice = voiceCondition(model, options);
    const projections = conditionProjections(model, voice.condition);
    const parts = [];
    function* chunks() {
        let frameCount = 0;
        let last = null;
        for (const sentenceIds of sentences(ids, vocab)) {
            const {features, durations} = inferDurations(model, sentenceIds, options, projections);
            parts.push({'ids': sentenceIds, durations, 'frameStart': frameCount});
            for (const chunk of decodeChunks(model, features, durations, projections)) {
                frameCount += chunk.length;
                last = chunk[chunk.length - 1];
                yield chunk;
            }
        }
        // The end frame - the last frame again
        if (last !== null) {
            yield [Float64Array.from(last)];
        }
    }
    return {'chunks': chunks(), 'eq': voice.eq, 'sentences': parts};
}
