// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// The neural voice - picovox_train/model.py's MultiModel forward pass in plain JavaScript (no dependencies, browser and
// Node): tokens (lib/tokens.js) to durations to synthesizer parameter frames (picovox_train/params.py), which
// lib/render.js renders.
//
//     import {loadModel, speakFrames} from './lib/model.js';
//     const model = loadModel(manifest, buffer);       // an export's <voice>.json (parsed) and <voice>.bin (ArrayBuffer)
//     const {frames, eq} = speakFrames(model, 'Hello, world.', {'speaker': 3});
//     const samples = render(frames, {'sampleRate': 44100, eq});
//
// The weights are packed integers (7-bit) with per-row scales (picovox_train/quant.py), dequantized on load; the arithmetic is float64.
//
// Long texts are spoken a sentence at a time (as the training data is - one or two sentences), each with its own <bos>.
//
// The model (lib/models/voiceModel.js) takes a voice - a speaker id (manifest.speakers: one row of the speaker table
// per voice).

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


// Load a model from its manifest (an export's <voice>.json, parsed) and weights (its <voice>.bin, an ArrayBuffer)
export function loadModel(manifest, buffer) {
    const tensors = {};
    for (const entry of manifest.tensors) {
        const size = entry.shape.reduce((product, dim) => product * dim, 1);
        const data = new Float64Array(size);
        if (entry.encoding === 'packed') {
            // (signed integers of the entry's bits, offset to unsigned, packed little-endian bit by bit; a float32
            // scale per row)
            const [rows] = entry.shape;
            const rowSize = size / rows;
            const bytes = new Uint8Array(buffer, entry.offset, Math.ceil((size * entry.bits) / 8));
            const scales = new Float32Array(buffer.slice(entry.scaleOffset, entry.scaleOffset + 4 * rows));
            const zero = (1 << (entry.bits - 1)) - 1;
            let bit = 0;
            for (let row = 0; row < rows; row++) {
                for (let ix = 0; ix < rowSize; ix++) {
                    let unsigned = 0;
                    for (let bitIndex = 0; bitIndex < entry.bits; bitIndex++, bit++) {
                        unsigned |= ((bytes[bit >> 3] >> (bit & 7)) & 1) << bitIndex;
                    }
                    data[row * rowSize + ix] = (unsigned - zero) * scales[row];
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


// A linear layer's outputs [outStart, outEnd) of one input row - each output's sum as four interleaved partial sums (the
// columns by their index modulo four, the columns past a multiple of four in the first), added to the bias in order
function linearRow(matrix, bias, inData, outData, inBase, outBase, inSize, outStart, outEnd) {
    const blockColumns = inSize - (inSize % 4);
    for (let outIx = outStart; outIx < outEnd; outIx++) {
        const base = outIx * inSize;
        let sum0 = 0;
        let sum1 = 0;
        let sum2 = 0;
        let sum3 = 0;
        for (let inIx = 0; inIx < blockColumns; inIx += 4) {
            sum0 += matrix[base + inIx] * inData[inBase + inIx];
            sum1 += matrix[base + inIx + 1] * inData[inBase + inIx + 1];
            sum2 += matrix[base + inIx + 2] * inData[inBase + inIx + 2];
            sum3 += matrix[base + inIx + 3] * inData[inBase + inIx + 3];
        }
        for (let inIx = blockColumns; inIx < inSize; inIx++) {
            sum0 += matrix[base + inIx] * inData[inBase + inIx];
        }
        outData[outBase + outIx] = bias[outIx] + sum0 + sum1 + sum2 + sum3;
    }
}


// A linear layer (weight [out, in], bias [out]) of a matrix's rows. Four input rows and three outputs are computed at a
// time, four columns a pass: the pass's twelve weights and sixteen inputs are read into locals once, for its 48 products
// - for engines without a JIT, where an array read costs several times a product of locals (the block's shape is the
// measured one - the paper's Section 12.4). Each output's sum is formed as linearRow forms it, so the results are
// identical.
function linear(model, prefix, input) {
    const {'data': matrix, shape} = weight(model, `${prefix}.weight`);
    const bias = weight(model, `${prefix}.bias`).data;
    const [outSize, inSize] = shape;
    const output = zeros(input.rows, outSize);
    const inData = input.data;
    const outData = output.data;
    const blockRows = input.rows - (input.rows % 4);
    const blockOuts = outSize - (outSize % 3);
    const blockColumns = inSize - (inSize % 4);
    for (let row = 0; row < blockRows; row += 4) {
        const inBase0 = row * inSize;
        const inBase1 = inBase0 + inSize;
        const inBase2 = inBase1 + inSize;
        const inBase3 = inBase2 + inSize;
        const outBase0 = row * outSize;
        const outBase1 = outBase0 + outSize;
        const outBase2 = outBase1 + outSize;
        const outBase3 = outBase2 + outSize;
        for (let outIx = 0; outIx < blockOuts; outIx += 3) {
            const base0 = outIx * inSize;
            const base1 = base0 + inSize;
            const base2 = base1 + inSize;

            // The sums - s[row][output][part]
            let s000 = 0, s001 = 0, s002 = 0, s003 = 0;
            let s010 = 0, s011 = 0, s012 = 0, s013 = 0;
            let s020 = 0, s021 = 0, s022 = 0, s023 = 0;
            let s100 = 0, s101 = 0, s102 = 0, s103 = 0;
            let s110 = 0, s111 = 0, s112 = 0, s113 = 0;
            let s120 = 0, s121 = 0, s122 = 0, s123 = 0;
            let s200 = 0, s201 = 0, s202 = 0, s203 = 0;
            let s210 = 0, s211 = 0, s212 = 0, s213 = 0;
            let s220 = 0, s221 = 0, s222 = 0, s223 = 0;
            let s300 = 0, s301 = 0, s302 = 0, s303 = 0;
            let s310 = 0, s311 = 0, s312 = 0, s313 = 0;
            let s320 = 0, s321 = 0, s322 = 0, s323 = 0;
            for (let inIx = 0; inIx < blockColumns; inIx += 4) {
                const w00 = matrix[base0 + inIx], w01 = matrix[base0 + inIx + 1];
                const w02 = matrix[base0 + inIx + 2], w03 = matrix[base0 + inIx + 3];
                const w10 = matrix[base1 + inIx], w11 = matrix[base1 + inIx + 1];
                const w12 = matrix[base1 + inIx + 2], w13 = matrix[base1 + inIx + 3];
                const w20 = matrix[base2 + inIx], w21 = matrix[base2 + inIx + 1];
                const w22 = matrix[base2 + inIx + 2], w23 = matrix[base2 + inIx + 3];
                const x00 = inData[inBase0 + inIx], x01 = inData[inBase0 + inIx + 1];
                const x02 = inData[inBase0 + inIx + 2], x03 = inData[inBase0 + inIx + 3];
                const x10 = inData[inBase1 + inIx], x11 = inData[inBase1 + inIx + 1];
                const x12 = inData[inBase1 + inIx + 2], x13 = inData[inBase1 + inIx + 3];
                const x20 = inData[inBase2 + inIx], x21 = inData[inBase2 + inIx + 1];
                const x22 = inData[inBase2 + inIx + 2], x23 = inData[inBase2 + inIx + 3];
                const x30 = inData[inBase3 + inIx], x31 = inData[inBase3 + inIx + 1];
                const x32 = inData[inBase3 + inIx + 2], x33 = inData[inBase3 + inIx + 3];
                s000 += w00 * x00;
                s001 += w01 * x01;
                s002 += w02 * x02;
                s003 += w03 * x03;
                s010 += w10 * x00;
                s011 += w11 * x01;
                s012 += w12 * x02;
                s013 += w13 * x03;
                s020 += w20 * x00;
                s021 += w21 * x01;
                s022 += w22 * x02;
                s023 += w23 * x03;
                s100 += w00 * x10;
                s101 += w01 * x11;
                s102 += w02 * x12;
                s103 += w03 * x13;
                s110 += w10 * x10;
                s111 += w11 * x11;
                s112 += w12 * x12;
                s113 += w13 * x13;
                s120 += w20 * x10;
                s121 += w21 * x11;
                s122 += w22 * x12;
                s123 += w23 * x13;
                s200 += w00 * x20;
                s201 += w01 * x21;
                s202 += w02 * x22;
                s203 += w03 * x23;
                s210 += w10 * x20;
                s211 += w11 * x21;
                s212 += w12 * x22;
                s213 += w13 * x23;
                s220 += w20 * x20;
                s221 += w21 * x21;
                s222 += w22 * x22;
                s223 += w23 * x23;
                s300 += w00 * x30;
                s301 += w01 * x31;
                s302 += w02 * x32;
                s303 += w03 * x33;
                s310 += w10 * x30;
                s311 += w11 * x31;
                s312 += w12 * x32;
                s313 += w13 * x33;
                s320 += w20 * x30;
                s321 += w21 * x31;
                s322 += w22 * x32;
                s323 += w23 * x33;
            }
            for (let inIx = blockColumns; inIx < inSize; inIx++) {
                const w0 = matrix[base0 + inIx], w1 = matrix[base1 + inIx], w2 = matrix[base2 + inIx];
                const x0 = inData[inBase0 + inIx], x1 = inData[inBase1 + inIx], x2 = inData[inBase2 + inIx], x3 = inData[inBase3 + inIx];
                s000 += w0 * x0;
                s010 += w1 * x0;
                s020 += w2 * x0;
                s100 += w0 * x1;
                s110 += w1 * x1;
                s120 += w2 * x1;
                s200 += w0 * x2;
                s210 += w1 * x2;
                s220 += w2 * x2;
                s300 += w0 * x3;
                s310 += w1 * x3;
                s320 += w2 * x3;
            }
            outData[outBase0 + outIx] = bias[outIx] + s000 + s001 + s002 + s003;
            outData[outBase0 + outIx + 1] = bias[outIx + 1] + s010 + s011 + s012 + s013;
            outData[outBase0 + outIx + 2] = bias[outIx + 2] + s020 + s021 + s022 + s023;
            outData[outBase1 + outIx] = bias[outIx] + s100 + s101 + s102 + s103;
            outData[outBase1 + outIx + 1] = bias[outIx + 1] + s110 + s111 + s112 + s113;
            outData[outBase1 + outIx + 2] = bias[outIx + 2] + s120 + s121 + s122 + s123;
            outData[outBase2 + outIx] = bias[outIx] + s200 + s201 + s202 + s203;
            outData[outBase2 + outIx + 1] = bias[outIx + 1] + s210 + s211 + s212 + s213;
            outData[outBase2 + outIx + 2] = bias[outIx + 2] + s220 + s221 + s222 + s223;
            outData[outBase3 + outIx] = bias[outIx] + s300 + s301 + s302 + s303;
            outData[outBase3 + outIx + 1] = bias[outIx + 1] + s310 + s311 + s312 + s313;
            outData[outBase3 + outIx + 2] = bias[outIx + 2] + s320 + s321 + s322 + s323;
        }

        // The outputs past a multiple of three
        linearRow(matrix, bias, inData, outData, inBase0, outBase0, inSize, blockOuts, outSize);
        linearRow(matrix, bias, inData, outData, inBase1, outBase1, inSize, blockOuts, outSize);
        linearRow(matrix, bias, inData, outData, inBase2, outBase2, inSize, blockOuts, outSize);
        linearRow(matrix, bias, inData, outData, inBase3, outBase3, inSize, blockOuts, outSize);
    }

    // The rows past a multiple of four
    for (let row = blockRows; row < input.rows; row++) {
        linearRow(matrix, bias, inData, outData, row * inSize, row * outSize, inSize, 0, outSize);
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
    const {length} = data;
    for (let ix = 0; ix < length; ix++) {
        const value = data[ix];
        data[ix] = value / (1 + Math.exp(-value));
    }
    return input;
}


// Add matrices (b to a, in place)
function addRows(inputA, inputB) {
    const dataA = inputA.data;
    const dataB = inputB.data;
    const {length} = dataA;
    for (let ix = 0; ix < length; ix++) {
        dataA[ix] += dataB[ix];
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
// The conditioning - the speaker
//

// A matrix of one row
function rowMatrix(values) {
    return {'rows': 1, 'cols': values.length, 'data': Float64Array.from(values)};
}


// A speaker's id - an id of the speaker table (default manifest.defaultSpeaker)
function speakerId(model, speaker) {
    const id = Number(speaker ?? model.manifest.defaultSpeaker);
    if (!Number.isInteger(id) || id < 0 || id >= model.config.speakers) {
        throw new Error(`invalid speaker ${JSON.stringify(speaker)}`);
    }
    return id;
}


// A voice - {condition (the condition vector - the speaker's embedding), eq (the speaker's channel equalizer, dB)} -
// see speakFrames for the options
function voiceCondition(model, {speaker} = {}) {
    const {speakerDim} = model.config;
    const [, bands] = weight(model, 'eq_table').shape;
    const id = speakerId(model, speaker);
    const condition = weight(model, 'speaker.weight').data.slice(id * speakerDim, (id + 1) * speakerDim);
    return {condition, 'eq': Array.from(weight(model, 'eq_table').data.slice(id * bands, (id + 1) * bands))};
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
    const {data, rows, cols} = input;
    for (let row = 0, ix = 0; row < rows; row++) {
        for (let channel = 0; channel < cols; channel++, ix++) {
            data[ix] += vector[channel];
        }
    }
    return input;
}


// FiLM - each row times (1 + gamma) plus beta, in place
function film(input, [gamma, beta]) {
    const {data, rows, cols} = input;
    for (let row = 0, ix = 0; row < rows; row++) {
        for (let channel = 0; channel < cols; channel++, ix++) {
            data[ix] = data[ix] * (1 + gamma[channel]) + beta[channel];
        }
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
        store(state, addRows(linear(model, `${prefix}.outer`, convolved), below), start);
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
// equalizer, dB), sentences: [{ids, durations, frameStart}]}. options: rate (a duration multiplier) and speaker (an id
// of manifest.speakers; default manifest.defaultSpeaker).
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
