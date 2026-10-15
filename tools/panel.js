// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

// The listener panel - synthesize a fixed, varied evaluation set (tools/panel-texts.json) with one or more voices and
// score it with listener models: Whisper word error rate (whisper.cpp, as tools/eval.js), UTMOS naturalness, ECAPA
// speaker embeddings, and emotion2vec+ emotion probabilities (tools/listeners.py, in build/venv-listeners)
//
// node tools/panel.js -v male [-o overridesJson] [--style name] [-v female ...] [--ref wav|dir ... [--ref-label name]]
//     [-s seed] [-n count] [--kind harvard,narrative,question,exclamation] [--show] [--json]
//     [--no-wer] [--no-emotion] [--device auto|cpu|mps] [--ref-count count]
//
// A voice (-v) is a voice name (lib/voices.js), a voice spec JSON file, or an Apple system voice (samantha, daniel,
// fred, ...). -o and --style apply to the voice before them (or, before any -v, to every voice). --ref adds a group of
// reference recordings - WAV files, or directories searched recursively for their first --ref-count (16) WAV files -
// human speech to compare against; their transcripts (ARCTIC etc/txt.done.data, LibriTTS .normalized.txt) give their
// WER. Synthesized audio, Whisper transcripts, and listener results are cached by content hash under out/panel/.

import {execFileSync, spawnSync} from 'node:child_process';
import {existsSync, mkdirSync, readFileSync, readdirSync, statSync, symlinkSync, writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import process from 'node:process';
import {speak} from '../lib/speak.js';
import {transcribeAll} from './eval.js';
import {writeWav} from '../lib/wav.js';


const rootDir = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const panelDir = 'out/panel';
const audioDir = `${panelDir}/audio`;
const scoresDir = `${panelDir}/scores`;
const listenersPython = 'build/venv-listeners/bin/python';
const systemVoices = {
    'samantha': 'Samantha', 'daniel': 'Daniel', 'fred': 'Fred', 'kathy': 'Kathy', 'ralph': 'Ralph', 'alex': 'Alex',
    'victoria': 'Victoria', 'karen': 'Karen', 'moira': 'Moira', 'tessa': 'Tessa'
};


// The short SHA-256 of a string or buffer
function hash(data) {
    return createHash('sha256').update(data).digest('hex').slice(0, 20);
}


// The scoring path of an audio file - a link named by the audio's content hash, beside which the Whisper transcript and
// listener results are cached (so they survive code changes that leave the audio identical, and are never written beside
// a reference recording)
function scorePath(wavPath) {
    const linkPath = `${scoresDir}/${hash(readFileSync(wavPath))}.wav`;
    if (!existsSync(linkPath)) {
        symlinkSync(path.resolve(wavPath), linkPath);
    }
    return linkPath;
}


// The hash of the synthesizer's code and dictionary - a change to either invalidates the synthesized-audio cache
function libraryHash() {
    const libFiles = readdirSync('lib').filter((name) => name.endsWith('.js')).sort();
    const digest = createHash('sha256');
    for (const name of libFiles) {
        digest.update(name).update(readFileSync(`lib/${name}`));
    }
    digest.update(readFileSync('data/cmudict.dict'));
    return digest.digest('hex').slice(0, 20);
}


// Parse the command line - returns the options and the voice and reference entries
function parseArgs(args, cwd) {
    const options = {
        'seed': 1, 'count': Infinity, 'kinds': null, 'show': false, 'json': false, 'wer': true, 'emotion': true,
        'device': 'auto', 'refCount': 16
    };
    const defaults = {'overrides': {}, 'style': null};
    const entries = [];
    const last = () => (entries.length === 0 ? defaults : entries[entries.length - 1]);
    for (let ix = 0; ix < args.length; ix++) {
        const arg = args[ix];
        if (arg === '-v') {
            entries.push({'kind': 'voice', 'voice': args[++ix], 'overrides': null, 'style': null});
        } else if (arg === '-o' || arg === '--overrides') {
            last().overrides = JSON.parse(args[++ix]);
        } else if (arg === '--style') {
            last().style = parseStyle(args[++ix], cwd);
        } else if (arg === '--ref') {
            const paths = [];
            while (ix + 1 < args.length && !args[ix + 1].startsWith('-')) {
                paths.push(path.resolve(cwd, args[++ix]));
            }
            entries.push({'kind': 'ref', 'paths': paths, 'label': null});
        } else if (arg === '--ref-label') {
            last().label = args[++ix];
        } else if (arg === '--ref-count') {
            options.refCount = Number(args[++ix]);
        } else if (arg === '-s' || arg === '--seed') {
            options.seed = Number(args[++ix]);
        } else if (arg === '-n') {
            options.count = Number(args[++ix]);
        } else if (arg === '--kind') {
            options.kinds = args[++ix].split(',');
        } else if (arg === '--show') {
            options.show = true;
        } else if (arg === '--json') {
            options.json = true;
        } else if (arg === '--no-wer') {
            options.wer = false;
        } else if (arg === '--no-emotion') {
            options.emotion = false;
        } else if (arg === '--device') {
            options.device = args[++ix];
        } else {
            throw new Error(`Unknown argument "${arg}"`);
        }
    }
    if (entries.length === 0) {
        entries.push({'kind': 'voice', 'voice': 'male', 'overrides': null, 'style': null});
    }
    for (const entry of entries) {
        if (entry.kind === 'voice') {
            entry.overrides ??= defaults.overrides;
            entry.style ??= defaults.style;
            if (entry.voice.endsWith('.json')) {
                entry.voice = path.resolve(cwd, entry.voice);
            }
        }
    }
    return {options, entries};
}


// Parse a --style argument - a style JSON object or array ({'scale', 'add', 'set', 'intensity'} - see lib/voices.js
// applyStyle), a style JSON file, or a style name (passed to speak as is) - returns {name, style}
function parseStyle(arg, cwd) {
    if (arg.startsWith('{') || arg.startsWith('[')) {
        return {'name': 'style', 'style': JSON.parse(arg)};
    }
    if (arg.endsWith('.json')) {
        return {'name': path.basename(arg, '.json'), 'style': JSON.parse(readFileSync(path.resolve(cwd, arg), 'utf8'))};
    }
    return {'name': arg, 'style': arg};
}


// The display label of a voice entry
function voiceLabel(entry) {
    let label = entry.voice.endsWith('.json') ? path.basename(entry.voice, '.json') : entry.voice;
    if (entry.style !== null) {
        label += `:${entry.style.name === 'style' ? JSON.stringify(entry.style.style) : entry.style.name}`;
    }
    if (Object.keys(entry.overrides).length !== 0) {
        label += ` ${JSON.stringify(entry.overrides)}`;
    }
    return label;
}


// Synthesize (or find in the cache) a voice entry's evaluation set - returns its file records
function synthesizeVoice(entry, texts, options, libHash) {
    const spec = entry.voice.endsWith('.json') ? JSON.parse(readFileSync(entry.voice, 'utf8')) : entry.voice;
    const systemVoice = typeof spec === 'string' ? systemVoices[spec.toLowerCase()] : undefined;
    const style = entry.style === null ? null : entry.style.style;
    return texts.map((item, ix) => {
        const seed = options.seed + ix;
        const key = systemVoice !== undefined
            ? JSON.stringify({'system': systemVoice, 'text': item.text})
            : JSON.stringify({'lib': libHash, spec, 'overrides': entry.overrides, 'style': style, seed, 'text': item.text});
        const wav = `${audioDir}/${hash(key)}.wav`;
        if (!existsSync(wav)) {
            if (systemVoice !== undefined) {
                execFileSync('say', ['-v', systemVoice, '-o', wav, '--data-format=LEI16@22050', item.text]);
            } else {
                const {samples, sampleRate} = speak(item.text, spec, {seed, 'overrides': entry.overrides, style});
                writeWav(wav, samples, sampleRate);
            }
        }
        return {'id': item.id, 'kind': item.kind, 'text': item.text, wav, 'score': scorePath(wav)};
    });
}


// The ARCTIC transcripts of a speaker directory (etc/txt.done.data) - returns {utteranceId: text}, cached
const arcticCache = {};
function arcticTranscripts(speakerDir) {
    if (!(speakerDir in arcticCache)) {
        const transcripts = {};
        const dataPath = path.join(speakerDir, 'etc', 'txt.done.data');
        if (existsSync(dataPath)) {
            for (const match of readFileSync(dataPath, 'utf8').matchAll(/\(\s*(\S+)\s+"(.*)"\s*\)/g)) {
                const [, utteranceId, text] = match;
                transcripts[utteranceId] = text;
            }
        }
        arcticCache[speakerDir] = transcripts;
    }
    return arcticCache[speakerDir];
}


// The reference transcript of a recording, if one is known - LibriTTS (<name>.normalized.txt) or ARCTIC - else null
function referenceText(wavPath) {
    const base = wavPath.replace(/\.wav$/, '');
    for (const suffix of ['.normalized.txt', '.original.txt', '.txt']) {
        if (existsSync(base + suffix)) {
            return readFileSync(base + suffix, 'utf8').trim();
        }
    }
    const arcticMatch = wavPath.match(/^(.*)\/wav\/([^/]+)\.wav$/);
    if (arcticMatch !== null) {
        return arcticTranscripts(arcticMatch[1])[arcticMatch[2]] ?? null;
    }
    return null;
}


// Collect a reference entry's recordings - returns its file records
function referenceFiles(entry, options) {
    const wavPaths = [];
    for (const refPath of entry.paths) {
        if (statSync(refPath).isDirectory()) {
            const names = readdirSync(refPath, {'recursive': true}).filter((name) => name.endsWith('.wav')).sort();
            wavPaths.push(...names.slice(0, options.refCount).map((name) => path.join(refPath, name)));
        } else {
            wavPaths.push(refPath);
        }
    }
    if (entry.label === null) {
        const first = entry.paths[0] ?? 'ref';
        const firstDir = statSync(first).isDirectory() ? first : path.dirname(first);
        const dirName = path.basename(firstDir);
        const name = dirName === 'wav' ? path.basename(path.dirname(firstDir)) : dirName;
        const corpus = /LibriTTS/i.test(firstDir) ? 'libritts-' : '';
        entry.label = `ref:${corpus}${name.replace(/^cmu_us_(\w+)_arctic$/, 'arctic-$1')}`;
    }
    return wavPaths.map((wavPath) => ({
        'id': path.basename(wavPath, '.wav'), 'kind': 'reference', 'text': referenceText(wavPath), 'wav': wavPath,
        'score': scorePath(wavPath)
    }));
}


// Normalize text to words for scoring (as tools/eval.js)
function scoreWords(text) {
    return text.toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').replace(/'/g, '').split(/\s+/).filter((word) => word !== '');
}


// Word-level edit distance
function editDistance(reference, hypothesis) {
    let previous = Array.from({'length': hypothesis.length + 1}, (_value, jx) => jx);
    for (let ix = 1; ix <= reference.length; ix++) {
        const current = [ix];
        for (let jx = 1; jx <= hypothesis.length; jx++) {
            const cost = reference[ix - 1] === hypothesis[jx - 1] ? 0 : 1;
            current.push(Math.min(previous[jx] + 1, current[jx - 1] + 1, previous[jx - 1] + cost));
        }
        previous = current;
    }
    return previous[hypothesis.length];
}


// Read a cached whisper.cpp JSON transcript - returns its text
function readTranscript(wav) {
    const result = JSON.parse(readFileSync(wav.replace(/\.wav$/, '.json'), 'utf8'));
    return result.transcription.map((segment) => segment.text).join('').trim();
}


// Transcribe the files with Whisper (the cached ones are read) and score each file's word errors
async function scoreWer(files) {
    const scored = files.filter((file) => file.text !== null);
    const missing = scored.filter((file) => !existsSync(file.score.replace(/\.wav$/, '.json')));
    await transcribeAll(missing.map((file) => file.score));
    for (const file of scored) {
        file.transcript = readTranscript(file.score);
        const reference = scoreWords(file.text);
        file.words = reference.length;
        file.errors = editDistance(reference, scoreWords(file.transcript));
    }
}


// Score the files with the listener models (tools/listeners.py; the cached results are read)
function scoreListeners(files, options) {
    const needed = ['mos', 'embedding', ...(options.emotion ? ['emotion'] : [])];
    const cachePath = (file) => file.score.replace(/\.wav$/, '.listeners.json');
    const readCache = (file) => (existsSync(cachePath(file)) ? JSON.parse(readFileSync(cachePath(file), 'utf8')) : {});
    const missing = files.filter((file) => needed.some((key) => !(key in readCache(file))));
    if (missing.length !== 0) {
        if (!existsSync(listenersPython)) {
            throw new Error(`${listenersPython} not found - run "make listeners" first`);
        }
        const pythonArgs = ['-I', 'tools/listeners.py', '--files-from', '-', '--device', options.device];
        if (!options.emotion) {
            pythonArgs.push('--no-emotion');
        }
        const run = spawnSync(listenersPython, pythonArgs, {
            'input': missing.map((file) => file.score).join('\n'), 'encoding': 'utf8', 'maxBuffer': 1024 * 1024 * 1024
        });
        if (run.status !== 0) {
            throw new Error(`tools/listeners.py failed:\n${run.stderr}`);
        }
        const lines = run.stdout.split('\n').filter((line) => line !== '');
        for (const [ix, line] of lines.entries()) {
            const result = JSON.parse(line);
            delete result.file;
            writeFileSync(cachePath(missing[ix]), JSON.stringify({...readCache(missing[ix]), ...result}));
        }
    }
    for (const file of files) {
        Object.assign(file, readCache(file));
    }
}


// The L2 norm of a vector
function norm(vector) {
    return Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
}

// A vector scaled to unit length
function unit(vector) {
    const length = norm(vector) || 1;
    return vector.map((value) => value / length);
}

// The element-wise mean of vectors
function meanVector(vectors) {
    return vectors[0].map((_value, ix) => vectors.reduce((sum, vector) => sum + vector[ix], 0) / vectors.length);
}

// The cosine similarity of two vectors
function cosine(vectorA, vectorB) {
    return vectorA.reduce((sum, value, ix) => sum + value * vectorB[ix], 0) / ((norm(vectorA) * norm(vectorB)) || 1);
}


// The mean of numbers (null if none)
function mean(values) {
    return values.length === 0 ? null : values.reduce((sum, value) => sum + value, 0) / values.length;
}

// The median of numbers (null if none)
function median(values) {
    if (values.length === 0) {
        return null;
    }
    const sorted = [...values].sort((valueA, valueB) => valueA - valueB);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}


// Summarize an entry's scored files - MOS statistics (overall and by text kind), WER, speaker consistency (the mean
// cosine similarity of each file's embedding to the voice's mean embedding), and the mean emotion probabilities
function summarize(files) {
    const moses = files.map((file) => file.mos).filter((mos) => mos !== undefined);
    const mosByKind = {};
    for (const kind of [...new Set(files.map((file) => file.kind))]) {
        mosByKind[kind] = mean(files.filter((file) => file.kind === kind && file.mos !== undefined).map((file) => file.mos));
    }
    const transcribed = files.filter((file) => file.errors !== undefined);
    const wordCount = transcribed.reduce((sum, file) => sum + file.words, 0);
    const embeddings = files.filter((file) => file.embedding !== undefined).map((file) => unit(file.embedding));
    const meanEmbedding = embeddings.length === 0 ? null : unit(meanVector(embeddings));
    const emotions = files.filter((file) => file.emotion !== undefined).map((file) => file.emotion);
    let emotion = null;
    if (emotions.length !== 0) {
        emotion = {};
        for (const name of Object.keys(emotions[0])) {
            emotion[name] = mean(emotions.map((probabilities) => probabilities[name] ?? 0));
        }
    }
    return {
        'files': files.length,
        'seconds': files.reduce((sum, file) => sum + (file.seconds ?? 0), 0),
        'mosMean': mean(moses), 'mosMedian': median(moses), 'mosMin': moses.length === 0 ? null : Math.min(...moses),
        'mosMax': moses.length === 0 ? null : Math.max(...moses), mosByKind,
        'wer': wordCount === 0 ? null : transcribed.reduce((sum, file) => sum + file.errors, 0) / wordCount,
        'consistency': meanEmbedding === null ? null : mean(embeddings.map((embedding) => cosine(embedding, meanEmbedding))),
        emotion, meanEmbedding
    };
}


// Format a number (or a dash for null)
function fmt(value, digits = 2, scale = 1) {
    return value === null || value === undefined ? '-' : (value * scale).toFixed(digits);
}


// The top emotions of a summary, as text
function topEmotions(emotion, count = 2) {
    if (emotion === null) {
        return '-';
    }
    return Object.entries(emotion).sort((entryA, entryB) => entryB[1] - entryA[1]).slice(0, count).
        map(([name, probability]) => `${name} ${probability.toFixed(2)}`).join(', ');
}


// Print the text report - the per-entry summary table, the per-file details (--show), and the similarity matrix
function printReport(results, similarity, timing, options) {
    const out = (line = '') => process.stdout.write(`${line}\n`);
    const labelWidth = Math.max(5, ...results.map((result) => result.label.length));
    const kinds = ['harvard', 'narrative', 'question', 'exclamation', 'reference'].filter(
        (kind) => results.some((result) => kind in result.summary.mosByKind)
    );
    const kindHeads = kinds.map((kind) => kind.slice(0, 5).padStart(6)).join('');
    out(`${'voice'.padEnd(labelWidth)}   n  MOS mean   med   min  ${kindHeads}    WER  spkr  emotion`);
    for (const {label, summary} of results) {
        const kindCells = kinds.map((kind) => fmt(summary.mosByKind[kind] ?? null).padStart(6)).join('');
        out(
            `${label.padEnd(labelWidth)} ${String(summary.files).padStart(3)}  ${fmt(summary.mosMean).padStart(8)} ` +
            `${fmt(summary.mosMedian).padStart(5)} ${fmt(summary.mosMin).padStart(5)}  ${kindCells} ` +
            `${(summary.wer === null ? '-' : `${fmt(summary.wer, 1, 100)}%`).padStart(6)}  ${fmt(summary.consistency).padStart(4)}  ` +
            `${topEmotions(summary.emotion)}`
        );
    }
    if (options.show) {
        for (const {label, files} of results) {
            out();
            out(`${label}:`);
            for (const file of files) {
                const errors = file.errors === undefined ? '' : ` ${file.errors}/${file.words} errors`;
                const heard = file.errors ? ` | heard: ${file.transcript}` : '';
                out(`  ${fmt(file.mos)} ${file.id}${errors} ${topEmotions(file.emotion ?? null, 1)}${heard}`);
            }
        }
    }
    if (similarity !== null) {
        out();
        out('Speaker similarity - the cosine of mean ECAPA embeddings (one human speaker ~0.9, two human speakers 0.0-0.5)');
        const rowLabels = similarity.labels.map((label, ix) => `${ix + 1}. ${label}`);
        const rowWidth = Math.max(...rowLabels.map((label) => label.length));
        out(`${''.padEnd(rowWidth)} ${similarity.labels.map((_label, ix) => String(ix + 1).padStart(6)).join('')}`);
        for (const [ix, row] of similarity.matrix.entries()) {
            out(`${rowLabels[ix].padEnd(rowWidth)} ${row.map((value) => fmt(value).padStart(6)).join('')}`);
        }
    }
    out();
    out(
        `time: synthesis ${fmt(timing.synthesis / 1000, 1)} s, whisper ${fmt(timing.whisper / 1000, 1)} s, ` +
        `listeners ${fmt(timing.listeners / 1000, 1)} s`
    );
}


// Run the panel
async function main() {
    const cwd = process.cwd();
    const {options, entries} = parseArgs(process.argv.slice(2), cwd);
    process.chdir(rootDir);
    mkdirSync(audioDir, {'recursive': true});
    mkdirSync(scoresDir, {'recursive': true});
    const panelTexts = JSON.parse(readFileSync('tools/panel-texts.json', 'utf8')).texts;
    const texts = panelTexts.filter((item) => options.kinds === null || options.kinds.includes(item.kind)).slice(0, options.count);

    // Synthesize (or collect) every entry's audio
    const timing = {'synthesis': 0, 'whisper': 0, 'listeners': 0};
    let start = Date.now();
    const libHash = libraryHash();
    const results = entries.map((entry) => {
        if (entry.kind === 'ref') {
            const files = referenceFiles(entry, options);
            return {'label': entry.label, 'kind': 'reference', 'paths': entry.paths, files};
        }
        const files = synthesizeVoice(entry, texts, options, libHash);
        const kind = typeof entry.voice === 'string' && entry.voice.toLowerCase() in systemVoices ? 'system' : 'synthesized';
        return {'label': voiceLabel(entry), kind, 'voice': entry.voice, 'overrides': entry.overrides, 'style': entry.style, files};
    });
    timing.synthesis = Date.now() - start;
    const allFiles = results.flatMap((result) => result.files);

    // Score with Whisper and the listener models
    start = Date.now();
    if (options.wer) {
        await scoreWer(allFiles);
    }
    timing.whisper = Date.now() - start;
    start = Date.now();
    scoreListeners(allFiles, options);
    timing.listeners = Date.now() - start;

    // Summaries and the speaker-similarity matrix
    for (const result of results) {
        result.summary = summarize(result.files);
    }
    let similarity = null;
    const embedded = results.filter((result) => result.summary.meanEmbedding !== null);
    if (embedded.length > 1) {
        similarity = {
            'labels': embedded.map((result) => result.label),
            'matrix': embedded.map(
                (resultA) => embedded.map((resultB) => cosine(resultA.summary.meanEmbedding, resultB.summary.meanEmbedding))
            )
        };
    }

    // Output
    if (options.json) {
        for (const result of results) {
            for (const file of result.files) {
                delete file.embedding;
            }
        }
        process.stdout.write(`${JSON.stringify({results, similarity, timing}, null, 2)}\n`);
    } else {
        printReport(results, similarity, timing, options);
    }
}


if (process.argv[1] && process.argv[1].endsWith('panel.js')) {
    await main();
}
