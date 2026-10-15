// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

// The neural voice - nn/model.py's forward pass in plain JavaScript (no dependencies, browser and Node): tokens
// (nn/js/tokens.js) to durations to synthesizer parameter frames (nn/params.py), which nn/js/render.js renders.
//
//     import {loadModel, speakFrames} from './nn/js/model.js';
//     const model = loadModel(manifest, buffer);       // data/nn/<voice>.json (parsed) and data/nn/<voice>.bin (ArrayBuffer)
//     const {frames, eq} = speakFrames(model, 'Hello, world.');
//     const samples = render(frames, {'sampleRate': 44100, eq});
//
// The weights are 8-bit (per-row scales, nn/export.py), dequantized on load; the arithmetic is float64.
// Long texts are spoken a sentence at a time (as the training data is - one or two sentences), each with its own <bos>.
//
// The multi-speaker, multi-style model (data/nn/multi - manifest kind 'multi', nn/model.py's MultiModel) takes a voice
// and a style:
//
//     const {frames, eq} = speakFrames(model, 'Hello, world.', {
//         'speaker': 5,                                   // a speaker id (manifest.speakers), or a blend
//         // 'speaker': {'5': 0.6, '12': 0.4},            //   {id: weight} (normalized to sum to 1)
//         // 'embedding': [...],                          // or a raw speaker embedding (manifest.config.speakerDim)
//         'style': {'emotion': 'happy', 'intensity': 0.7} // an emotion (manifest.styles, or 'neutral') at 0-1
//         // 'style': {'happy': 0.5, 'sad': 0.2}          //   or a blend {emotion: weight}
//     });

import {NAMES, PARAMETERS} from './params.js';
import {tokenize, vocab} from './tokens.js';
import {bandwidthScale} from './transforms.js';


// The token classes (nn/vocab.json, nn/model.py)
const PAD = 0;
const BOS = 2;
const WORD = 3;
const FIRST_PHONE = 9;
const SENTENCE_ENDS = new Set(['<period>', '<question>', '<wh>', '<exclamation>']);
const LAYER_NORM_EPS = 1e-5;


// Load a model from its manifest (data/nn/<voice>.json, parsed) and weights (data/nn/<voice>.bin, an ArrayBuffer)
export function loadModel(manifest, buffer) {
    const tensors = {};
    for (const entry of manifest.tensors) {
        const size = entry.shape.reduce((product, dim) => product * dim, 1);
        const data = new Float64Array(size);
        if (entry.encoding === 'int8') {
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
        remap.set(vocab.tokens.indexOf(from), vocab.tokens.indexOf(to));
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


// A linear layer (weight [out, in], bias [out]) of a matrix's rows
function linear(model, prefix, input) {
    const {'data': matrix, shape} = weight(model, `${prefix}.weight`);
    const bias = weight(model, `${prefix}.bias`).data;
    const [outSize, inSize] = shape;
    const output = zeros(input.rows, outSize);
    const inData = input.data;
    const outData = output.data;
    for (let row = 0; row < input.rows; row++) {
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
    for (let row = 0; row < rows; row++) {
        const base = row * cols;
        let mean = 0;
        for (let ix = 0; ix < cols; ix++) {
            mean += input.data[base + ix];
        }
        mean /= cols;
        let variance = 0;
        for (let ix = 0; ix < cols; ix++) {
            variance += (input.data[base + ix] - mean) ** 2;
        }
        const scale = 1 / Math.sqrt(variance / cols + LAYER_NORM_EPS);
        for (let ix = 0; ix < cols; ix++) {
            output.data[base + ix] = (input.data[base + ix] - mean) * scale * gain[ix] + bias[ix];
        }
    }
    return output;
}


// A depthwise convolution over time (weight [C, 1, K], 'same' zero padding)
function depthwiseConv(model, prefix, input) {
    const {'data': kernel, shape} = weight(model, `${prefix}.conv.weight`);
    const bias = weight(model, `${prefix}.conv.bias`).data;
    const [channels, , size] = shape;
    const half = Math.floor(size / 2);
    const output = zeros(input.rows, channels);
    for (let row = 0; row < input.rows; row++) {
        const outBase = row * channels;
        for (let channel = 0; channel < channels; channel++) {
            output.data[outBase + channel] = bias[channel];
        }
        for (let tap = 0; tap < size; tap++) {
            const source = row + tap - half;
            if (source < 0 || source >= input.rows) {
                continue;
            }
            const inBase = source * channels;
            for (let channel = 0; channel < channels; channel++) {
                output.data[outBase + channel] += kernel[channel * size + tap] * input.data[inBase + channel];
            }
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


// A conv-FFN block - x + W2 SiLU(DW(SiLU(W1 LN(x)))), with FiLM ([gamma, beta]) after the norm when given
function convFFN(model, prefix, input, filmVectors = null) {
    const normed = layerNorm(model, `${prefix}.norm`, input);
    const hidden = silu(linear(model, `${prefix}.inner`, filmVectors ? film(normed, filmVectors) : normed));
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
    for (let head = 0; head < heads; head++) {
        const offset = head * headDim;
        for (let query = 0; query < length; query++) {
            const queryBase = query * 3 * dim + offset;
            let max = -Infinity;
            for (let key = 0; key < length; key++) {
                const keyBase = key * 3 * dim + dim + offset;
                let sum = 0;
                for (let ix = 0; ix < headDim; ix++) {
                    sum += qkv[queryBase + ix] * qkv[keyBase + ix];
                }
                scores[key] = sum * scale;
                max = Math.max(max, scores[key]);
            }
            let total = 0;
            for (let key = 0; key < length; key++) {
                scores[key] = Math.exp(scores[key] - max);
                total += scores[key];
            }
            const outBase = query * dim + offset;
            for (let key = 0; key < length; key++) {
                const share = scores[key] / total;
                const valueBase = key * 3 * dim + 2 * dim + offset;
                for (let ix = 0; ix < headDim; ix++) {
                    attended.data[outBase + ix] += share * qkv[valueBase + ix];
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
// The model
//

//
// The multi-speaker model's conditioning
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
        [[Number(speaker ?? model.manifest.defaultSpeaker ?? 0), 1]];
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


// The multi-speaker model's voice - {condition (the condition vector [speaker embedding, style embedding]), eq (the
// channel equalizer, dB)} - see speakFrames for the options
export function voiceCondition(model, {speaker, embedding, style, eq} = {}) {
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
    if (embedding !== undefined && embedding !== null) {
        if (embedding.length !== speakerDim) {
            throw new Error(`the speaker embedding must have ${speakerDim} values`);
        }
        condition.set(embedding);
    }
    const styleTable = weight(model, 'style').data;
    const weights = styleWeights(model, style);
    for (const [row, share] of weights.entries()) {
        for (let ix = 0; ix < styleDim; ix++) {
            condition[speakerDim + ix] += share * styleTable[row * styleDim + ix];
        }
    }
    return {condition, 'eq': eq ?? voiceEq};
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


// Token ids to token features (rows [N][D])
function encode(model, ids, projections) {
    const embedding = weight(model, 'embedding.weight');
    if (projections) {
        const [, width] = embedding.shape;
        let rows = zeros(ids.length, width);
        for (const [index, id] of ids.entries()) {
            for (let ix = 0; ix < width; ix++) {
                rows.data[index * width + ix] = embedding.data[id * width + ix] + projections.encoder[ix];
            }
        }
        for (let layer = 0; layer < 2 * model.config.encoderLayers; layer++) {
            const prefix = `encoder.${layer}`;
            rows = layer % 2 === 0 ? convFFN(model, prefix, rows, projections.films[prefix]) : selfAttention(model, prefix, rows);
        }
        return layerNorm(model, 'encoder_norm', rows);
    }
    const speaker = weight(model, 'speaker.weight').data;
    const [, dim] = embedding.shape;
    let rows = zeros(ids.length, dim);
    for (const [index, id] of ids.entries()) {
        for (let ix = 0; ix < dim; ix++) {
            rows.data[index * dim + ix] = embedding.data[id * dim + ix] + speaker[ix];
        }
    }
    for (let layer = 0; layer < 2 * model.config.encoderLayers; layer++) {
        rows = layer % 2 === 0 ? convFFN(model, `encoder.${layer}`, rows) : selfAttention(model, `encoder.${layer}`, rows);
    }
    return layerNorm(model, 'encoder_norm', rows);
}


// Predicted durations (10 ms frames) of token features
function predictDurations(model, ids, features, {pauseThreshold = 0.5, rate = 1} = {}, projections = null) {
    let rows = projections ? addVector({'rows': features.rows, 'cols': features.cols, 'data': Float64Array.from(features.data)},
        projections.duration) : features;
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
            duration = 1 / (1 + Math.exp(-pauseLogit)) > pauseThreshold ? Math.max(1, Math.expm1(logPaused)) * rate : 0;
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


// Decode token features and durations to parameter frames (Float64Array rows of nn/params.py's parameters, 5 ms)
function decode(model, features, durations, projections = null) {
    const {config} = model;
    const frameCount = durations.reduce((sum, duration) => sum + duration, 0);
    if (frameCount === 0) {
        return [];
    }
    const dim = features.cols;
    const regulated = zeros(frameCount, dim);
    const positions = zeros(frameCount, 5);
    let frame = 0;
    for (const [index, duration] of durations.entries()) {
        for (let offset = 0; offset < duration; offset++) {
            regulated.data.set(features.data.subarray(index * dim, (index + 1) * dim), frame * dim);
            positions.data.set([
                (offset + 0.5) / duration, 1 - (offset + 0.5) / duration, Math.log1p(duration) / 3,
                Math.min(offset, 20) / 20, Math.min(duration - 1 - offset, 20) / 20
            ], frame * 5);
            frame++;
        }
    }
    let rows = addRows(linear(model, 'decoder_in', regulated), linear(model, 'position_in', positions));
    addVector(rows, projections ? projections.decoder : weight(model, 'decoder_speaker.weight').data);
    for (let layer = 0; layer < config.decoderLayers; layer++) {
        const prefix = `decoder.${layer}`;
        rows = convFFN(model, prefix, rows, projections ? projections.films[prefix] : null);
    }
    const logits = linear(model, 'head', layerNorm(model, 'decoder_norm', rows));
    const count = NAMES.length;
    const jitterIx = NAMES.indexOf('jitter');
    const shimmerIx = NAMES.indexOf('shimmer');
    const frames = [];
    for (let row = 0; row < logits.rows; row++) {
        for (let sub = 0; sub < config.subframes; sub++) {
            const out = new Float64Array(count);
            const base = row * logits.cols + sub * count;
            for (let ix = 0; ix < count; ix++) {
                const unit = 1 / (1 + Math.exp(-logits.data[base + ix]));
                const {scale, low, high} = PARAMETERS[ix];
                out[ix] = scale === 'log' ? Math.exp(Math.log(low) + unit * (Math.log(high) - Math.log(low))) : low + unit * (high - low);
            }
            out[jitterIx] = config.jitter;
            out[shimmerIx] = config.shimmer;
            frames.push(out);
        }
    }
    return frames;
}


// Whether a model is the multi-speaker model
export function isMulti(model) {
    return model.config.kind === 'multi';
}


// Infer one token sequence (ids, starting with <bos>) - {durations (10 ms frames per token), frames (5 ms; 2F of them -
// the caller appends the end frame)}. A multi-speaker model's voice is options.projections (from options.condition, or
// the speakFrames voice options).
export function infer(model, ids, options = {}) {
    let projections = null;
    if (isMulti(model)) {
        projections = options.projections ?? conditionProjections(model, options.condition ?? voiceCondition(model, options).condition);
    }
    const mapped = ids.map((id) => model.remap.get(id) ?? id);
    const features = encode(model, mapped, projections);
    const durations = predictDurations(model, mapped, features, options, projections);
    return {durations, 'frames': decode(model, features, durations, projections)};
}


// Split token ids into sentences - each starting with <bos>, ending at a sentence end
export function sentences(ids) {
    const out = [];
    let current = [BOS];
    for (const id of ids.slice(ids[0] === BOS ? 1 : 0)) {
        current.push(id);
        if (SENTENCE_ENDS.has(vocab.tokens[id])) {
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
// equalizer, dB), sentences: [{ids, durations, frameStart}]}. options: pauseThreshold, rate (a duration multiplier),
// bandwidthScale (default: the voice's - the manifest's inference.bandwidthScale, which sharpens the network's
// over-smoothed formant bandwidths; 1 for the network's own). The multi-speaker model's voice options: speaker (an id
// of manifest.speakers, or a blend {id: weight}; default manifest.defaultSpeaker), embedding (a raw speaker embedding,
// in place of the speaker's), style ({emotion, intensity} - an emotion of manifest.styles or 'neutral', intensity 0-1 -
// or a blend {emotion: weight}), and eq (a channel equalizer in place of the speakers' blend).
export function speakFrames(model, text, options = {}) {
    const {ids} = tokenize(text);
    const frames = [];
    const parts = [];
    let voice = null;
    let sentenceOptions = options;
    if (isMulti(model)) {
        voice = voiceCondition(model, options);
        sentenceOptions = {...options, 'projections': conditionProjections(model, voice.condition)};
    }
    for (const sentenceIds of sentences(ids)) {
        const result = infer(model, sentenceIds, sentenceOptions);
        parts.push({'ids': sentenceIds, 'durations': result.durations, 'frameStart': frames.length});
        frames.push(...result.frames);
    }
    if (frames.length > 0) {
        frames.push(Float64Array.from(frames[frames.length - 1]));
    }
    const scale = options.bandwidthScale ?? model.manifest.inference?.bandwidthScale ?? 1;
    return {'frames': scale === 1 ? frames : bandwidthScale(frames, scale), 'eq': voice ? voice.eq : model.manifest.eq, 'sentences': parts};
}
