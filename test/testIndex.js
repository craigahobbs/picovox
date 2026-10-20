// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {modelLoader, textBlocks, voiceSpeaker} from '../lib/synthesis.js';
import {readFileSync, readdirSync} from 'node:fs';
import {speak, stream, synthesize, toWav, voices} from '../lib/index.js';
import {strict as assert} from 'node:assert';
import {encodeWav} from '../lib/wav.js';
import test from 'node:test';


// Blocks concatenated
function concat(blocks) {
    const samples = new Float32Array(blocks.reduce((sum, block) => sum + block.length, 0));
    let offset = 0;
    for (const block of blocks) {
        samples.set(block, offset);
        offset += block.length;
    }
    return samples;
}


// A stream's blocks
async function streamBlocks(text, options) {
    const blocks = [];
    for await (const block of stream(text, options)) {
        blocks.push(block);
    }
    return blocks;
}


// A text of two paragraphs
const text = 'It was the best of times. "Was it?" she asked.\n\nNobody answered.';


test('voices', () => {
    assert.deepEqual(voices().map(({key, label, sex}) => [key, label, sex]), [
        ['male', 'Male', 'male'], ['female', 'Female', 'female'], ['sarah', 'Sarah', 'female'], ['helen', 'Helen', 'female'],
        ['amy', 'Amy', 'female'], ['kate', 'Kate', 'female'], ['grace', 'Grace', 'female'], ['james', 'James', 'male'],
        ['robert', 'Robert', 'male'], ['michael', 'Michael', 'male'], ['david', 'David', 'male'], ['thomas', 'Thomas', 'male']
    ]);
    assert(voices().every((voice) => voice.description.length > 20));

    // Each call's voices are its own
    voices()[0].key = 'other';
    assert.equal(voices()[0].key, 'male');
});


test('synthesize', async () => {
    const {samples, sampleRate} = await synthesize(text, {'voice': 'female'});
    assert(samples instanceof Float32Array);
    assert.equal(sampleRate, 24000);
    assert(samples.length / sampleRate > 2 && samples.length / sampleRate < 6);
    assert(samples.every((value) => value > -1 && value < 1));

    // The synthesis pipeline - the female voice, seed 1000 for the first sentence, normal speed
    const model = await modelLoader(() => import('../lib/models/g2pModel.js'), () => import('../lib/models/voiceModel.js'))();
    const expected = concat([...textBlocks(model, text, {'speaker': voiceSpeaker(model, 'female'), 'seed': 1000})]);
    assert.deepEqual(samples, expected);
});


test('synthesize, defaults', async () => {
    // The male voice, normal speed, 24 kHz
    const {samples, sampleRate} = await synthesize('Hello.');
    assert.equal(sampleRate, 24000);
    assert.deepEqual(samples, (await synthesize('Hello.', {'voice': 'male', 'rate': 1, 'seed': 1000, 'sampleRate': 24000})).samples);
    assert.notDeepEqual(samples, (await synthesize('Hello.', {'voice': 'female'})).samples);

    // Undefined options take their defaults
    assert.deepEqual((await synthesize('Hello.', {'voice': undefined, 'rate': undefined})).samples, samples);
});


test('synthesize, options', async () => {
    const base = (await synthesize('Call me Ishmael.')).samples;

    // Faster is shorter
    const fast = (await synthesize('Call me Ishmael.', {'rate': 2})).samples;
    const slow = (await synthesize('Call me Ishmael.', {'rate': 0.5})).samples;
    assert(fast.length < base.length && base.length < slow.length);

    // Another seed is other noise, the same length
    const seeded = (await synthesize('Call me Ishmael.', {'seed': 7})).samples;
    assert.equal(seeded.length, base.length);
    assert.notDeepEqual(seeded, base);

    // Another sample rate
    const {samples, sampleRate} = await synthesize('Call me Ishmael.', {'sampleRate': 44100});
    assert.equal(sampleRate, 44100);
    assert(Math.abs(samples.length / 44100 - base.length / 24000) < 0.01);
});


test('synthesize, no speech', async () => {
    assert.deepEqual(await synthesize(''), {'samples': new Float32Array(0), 'sampleRate': 24000});
    assert.deepEqual(await synthesize(' -- ... \n\n ', {'sampleRate': 16000}), {'samples': new Float32Array(0), 'sampleRate': 16000});
});


test('stream', async () => {
    // The blocks, concatenated, are synthesize's samples
    const blocks = await streamBlocks(text, {'voice': 'female', 'rate': 1.2});
    assert(blocks.length > 10);
    assert(blocks.every((block) => block instanceof Float32Array));
    assert.deepEqual(concat(blocks), (await synthesize(text, {'voice': 'female', 'rate': 1.2})).samples);

    // The first block is the first sentence's pause (a paragraph's - 0.4 s at 1.2x)
    assert.equal(blocks[0].length, Math.round(24000 * 0.4 / 1.2));
    assert(blocks[0].every((value) => value === 0));

    // Defaults
    assert.deepEqual(concat(await streamBlocks('Hello.')), (await synthesize('Hello.')).samples);
});


test('stream, early stop', async () => {
    // Taking a few blocks synthesizes only those
    const blocks = [];
    for await (const block of stream(text)) {
        blocks.push(block);
        if (blocks.length === 3) {
            break;
        }
    }
    const all = await streamBlocks(text);
    assert.deepEqual(blocks, all.slice(0, 3));
});


// The invalid-voice error's list of the voices
const voiceList = 'the voices are "male", "female", "sarah", "helen", "amy", "kate", "grace", "james", "robert", "michael", ' +
    '"david", "thomas"';


test('stream, invalid', () => {
    // The text and options are checked at once
    assert.throws(() => stream(42), {'name': 'TypeError', 'message': 'The text must be a string, not 42'});
    assert.throws(() => stream('Hi', {'voice': 'robot'}), {
        'name': 'RangeError', 'message': `Unknown voice "robot" - ${voiceList}`
    });
});


test('synthesize, invalid', async () => {
    const invalid = (message, name, textValue, options) => assert.rejects(synthesize(textValue, options), {name, message});

    // The text
    await invalid('The text must be a string, not undefined', 'TypeError');
    await invalid('The text must be a string, not null', 'TypeError', null);
    await invalid('The text must be a string, not an array', 'TypeError', ['Hi']);
    await invalid('The text must be a string, not a object', 'TypeError', {'text': 'Hi'});
    await invalid('The text must be a string, not true', 'TypeError', true);

    // The options
    await invalid('The options must be an object, not null', 'TypeError', 'Hi', null);
    await invalid('The options must be an object, not an array', 'TypeError', 'Hi', []);
    await invalid('The options must be an object, not "fast"', 'TypeError', 'Hi', 'fast');
    await invalid('Unknown option "speed" - the options are "voice", "rate", "seed", "sampleRate"', 'TypeError', 'Hi', {'speed': 2});
    await invalid('Unknown option "context" - the options are "voice", "rate", "seed", "sampleRate"', 'TypeError', 'Hi', {'context': {}});
    await invalid(`Unknown voice 1 - ${voiceList}`, 'RangeError', 'Hi', {'voice': 1});
    await invalid(`Unknown voice "Male" - ${voiceList}`, 'RangeError', 'Hi', {'voice': 'Male'});
    await invalid('The rate must be a number from 0.5 to 2, not 3', 'RangeError', 'Hi', {'rate': 3});
    await invalid('The rate must be a number from 0.5 to 2, not 0.4', 'RangeError', 'Hi', {'rate': 0.4});
    await invalid('The rate must be a number from 0.5 to 2, not NaN', 'RangeError', 'Hi', {'rate': NaN});
    await invalid('The rate must be a number from 0.5 to 2, not "1"', 'RangeError', 'Hi', {'rate': '1'});
    await invalid('The seed must be an integer, not 1.5', 'RangeError', 'Hi', {'seed': 1.5});
    await invalid('The seed must be an integer, not a function', 'RangeError', 'Hi', {'seed': Math.floor});
    await invalid('The sample rate must be an integer from 8000 to 96000, not 4000', 'RangeError', 'Hi', {'sampleRate': 4000});
    await invalid('The sample rate must be an integer from 8000 to 96000, not 100000', 'RangeError', 'Hi', {'sampleRate': 100000});
    await invalid('The sample rate must be an integer from 8000 to 96000, not 22050.5', 'RangeError', 'Hi', {'sampleRate': 22050.5});

    // The range's ends are valid
    assert.equal((await synthesize('Hi', {'rate': 0.5, 'seed': -3, 'sampleRate': 8000})).sampleRate, 8000);
    assert.equal((await synthesize('Hi', {'rate': 2, 'sampleRate': 96000})).sampleRate, 96000);
});


test('toWav', () => {
    const samples = Float32Array.from([0, 0.5, -0.5, 1, -2]);
    const wav = toWav(samples, 24000);
    assert(wav instanceof Uint8Array);
    assert.deepEqual(wav, encodeWav(samples, 24000));
    assert.equal(wav.length, 44 + 2 * samples.length);
    assert.equal(new TextDecoder().decode(wav.subarray(0, 4)), 'RIFF');

    // Float64Array and array samples
    assert.deepEqual(toWav(Float64Array.from(samples), 24000), wav);
    assert.deepEqual(toWav(Array.from(samples), 24000), wav);

    // Invalid
    assert.throws(() => toWav('samples', 24000), {
        'name': 'TypeError', 'message': 'The samples must be a Float32Array (or an array of numbers), not "samples"'
    });
    assert.throws(() => toWav(new Int16Array(2), 24000), {
        'name': 'TypeError', 'message': 'The samples must be a Float32Array (or an array of numbers), not a object'
    });
    assert.throws(() => toWav(samples), {'name': 'RangeError', 'message': 'The sample rate must be a positive integer, not undefined'});
    assert.throws(() => toWav(samples, 0), {'name': 'RangeError', 'message': 'The sample rate must be a positive integer, not 0'});
    assert.throws(() => toWav(samples, 2 ** 31), {
        'name': 'RangeError', 'message': 'The sample rate must be a positive integer, not 2147483648'
    });
});


test('toWav, synthesize', async () => {
    const {samples, sampleRate} = await synthesize('Hello.');
    const wav = toWav(samples, sampleRate);
    const view = new DataView(wav.buffer);
    assert.equal(view.getUint32(24, true), 24000);
    assert.equal(view.getUint32(40, true), 2 * samples.length);
});


//
// speak - with a fake Web Audio context
//

// A fake AudioContext - its time advances when advance() is called, ending the blocks played by then
class FakeAudioContext {
    constructor({state = 'running', instant = false, failBuffer = false} = {}) {
        this.currentTime = 0;
        this.state = state;
        this.destination = {'name': 'destination'};
        this.sources = [];
        this.instant = instant;
        this.failBuffer = failBuffer;
        this.resumed = 0;
        this.closed = 0;
        FakeAudioContext.instances.push(this);
    }

    createBuffer(channels, length, sampleRate) {
        if (this.failBuffer) {
            throw new Error('no buffers');
        }
        return {
            channels, length, sampleRate, 'data': null,
            copyToChannel(data, channel) {
                assert.equal(channel, 0);
                this.data = Float32Array.from(data);
            }
        };
    }

    createBufferSource() {
        const {instant, sources} = this;
        return {
            'buffer': null, 'onended': null, 'output': null, 'startTime': null, 'stopped': false, 'ended': false,
            connect(output) {
                this.output = output;
            },
            start(time) {
                this.startTime = time;
                sources.push(this);
                if (instant) {
                    queueMicrotask(() => {
                        this.ended = true;
                        this.onended?.();
                    });
                }
            },
            stop() {
                this.stopped = true;
            }
        };
    }

    resume() {
        this.resumed += 1;
        this.state = 'running';
        return Promise.resolve();
    }

    close() {
        this.closed += 1;
        this.state = 'closed';
        return Promise.resolve();
    }

    // Advance the time, ending the blocks that have played
    advance(seconds) {
        this.currentTime += seconds;
        for (const source of this.sources) {
            if (!source.ended && source.startTime + source.buffer.length / source.buffer.sampleRate <= this.currentTime) {
                source.ended = true;
                source.onended?.();
            }
        }
    }

    // The scheduled samples, in order
    samples() {
        return concat(this.sources.map((source) => source.buffer.data));
    }

    // How far ahead of the time the blocks are scheduled (s)
    ahead() {
        const last = this.sources[this.sources.length - 1];
        return last ? last.startTime + last.buffer.length / last.buffer.sampleRate - this.currentTime : 0;
    }
}
FakeAudioContext.instances = [];


// Run a function with a global AudioContext class (or webkitAudioContext) - Node has neither
async function withAudioContext(name, callback) {
    assert.equal(globalThis[name], undefined);
    globalThis[name] = FakeAudioContext;
    FakeAudioContext.instances = [];
    try {
        await callback();
    } finally {
        delete globalThis[name];
    }
}


// A short wait (ms)
function wait(ms = 1) {
    return new Promise((resolve) => {
        setTimeout(resolve, ms);
    });
}


// Wait until speak is idle - no new block scheduled across a few event-loop turns (it then waits for playback) - so a
// test's fake clock never outruns synthesis, however loaded the machine
async function speakIdle(context) {
    let count = -1;
    let quiet = 0;
    while (quiet < 3) {
        await wait();
        if (context.sources.length === count) {
            quiet += 1;
        } else {
            quiet = 0;
            count = context.sources.length;
        }
    }
}


// Play a playback to its end - advancing the context's time a step at a time
async function playThrough(context, playback, step = 0.5) {
    const state = {'settled': false};
    const settle = () => {
        state.settled = true;
    };
    playback.finished.then(settle, settle);
    while (!state.settled) {
        await wait();
        context.advance(step);
    }
    return playback.finished;
}


test('speak, Node', () => {
    // Under Node there's no Web Audio
    assert.equal(globalThis.AudioContext, undefined);
    assert.throws(() => speak('Hello.'), {
        'message': 'speak requires Web Audio (AudioContext), as in a browser - under Node, use synthesize and toWav'
    });

    // The text and options are checked first
    assert.throws(() => speak(42), {'name': 'TypeError', 'message': 'The text must be a string, not 42'});
    assert.throws(() => speak('Hi', {'context': null}), {'name': 'TypeError', 'message': 'The context must be an AudioContext, not null'});
    assert.throws(() => speak('Hi', {'context': 'audio'}), {
        'name': 'TypeError', 'message': 'The context must be an AudioContext, not "audio"'
    });
    assert.throws(() => speak('Hi', {'speed': 1}), {
        'name': 'TypeError', 'message': 'Unknown option "speed" - the options are "voice", "rate", "seed", "sampleRate", "context"'
    });
});


test('speak', async () => {
    await withAudioContext('AudioContext', async () => {
        const shortText = 'Hello there. How are you?';
        const {samples} = await synthesize(shortText, {'voice': 'female'});
        const playback = speak(shortText, {'voice': 'female'});
        assert.equal(typeof playback.stop, 'function');
        const [context] = FakeAudioContext.instances;

        // The blocks are scheduled one after another, from just after the start - the synthesized speech
        while (context.samples().length < samples.length) {
            await wait();
        }
        assert.deepEqual(context.samples(), samples);
        assert.equal(await playThrough(context, playback), undefined);
        let time = 0.05;
        for (const source of context.sources) {
            assert.equal(source.output, context.destination);
            assert.equal(source.buffer.sampleRate, 24000);
            assert(Math.abs(source.startTime - time) < 1e-9);
            time = source.startTime + source.buffer.length / 24000;
        }

        // The playback's own context is closed at the end; stopping after the end does nothing
        assert.equal(context.closed, 1);
        assert.equal(context.resumed, 0);
        playback.stop();
        assert.equal(context.closed, 1);
        assert(context.sources.every((source) => !source.stopped));
    });
});


test('speak, ahead of playback', async () => {
    await withAudioContext('AudioContext', async () => {
        // Synthesis stays at most a few seconds ahead of playback
        const longText = 'It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of ' +
            'foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of light.';
        const playback = speak(longText, {'rate': 0.5});
        const [context] = FakeAudioContext.instances;
        let maxAhead = 0;
        const state = {'settled': false};
        playback.finished.then(() => {
            state.settled = true;
        });
        let steps = 0;
        while (!state.settled) {
            await speakIdle(context);
            maxAhead = Math.max(maxAhead, context.ahead());
            context.advance(0.2);
            steps += 1;
        }
        assert(maxAhead > 3 && maxAhead < 4.3, `ahead ${maxAhead} s`);
        assert(steps > 20);
        assert.deepEqual(context.samples(), (await synthesize(longText, {'rate': 0.5})).samples);
        assert.equal(context.closed, 1);
    });
});


test('speak, stop', async () => {
    await withAudioContext('AudioContext', async () => {
        // Stopped while ahead of playback - the scheduled blocks are stopped, no more are synthesized
        const longText = 'It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of ' +
            'foolishness, it was the epoch of belief, it was the epoch of incredulity.';
        const playback = speak(longText, {'rate': 0.5});
        const [context] = FakeAudioContext.instances;
        while (context.ahead() < 4) {
            await wait();
        }
        await wait(20);
        const scheduled = context.sources.length;
        playback.stop();
        assert.equal(await playback.finished, undefined);
        await wait(20);
        assert.equal(context.sources.length, scheduled);
        assert(context.sources.every((source) => source.stopped));
        assert.equal(context.closed, 1);

        // Stopping again does nothing
        playback.stop();
        assert.equal(context.closed, 1);
    });
});


test('speak, stop at once', async () => {
    await withAudioContext('AudioContext', async () => {
        // Stopped before the first block - nothing is played
        const playback = speak(text);
        playback.stop();
        await playback.finished;
        await wait(50);
        const [context] = FakeAudioContext.instances;
        assert.equal(context.sources.length, 0);
        assert.equal(context.closed, 1);
    });
});


test('speak, a given context', async () => {
    // A given context is resumed if suspended, and not closed - with no global AudioContext
    assert.equal(globalThis.AudioContext, undefined);
    const context = new FakeAudioContext({'state': 'suspended'});
    const playback = speak('Hello.', {context, 'seed': 5, 'sampleRate': 16000});
    assert.equal(context.resumed, 1);
    await playThrough(context, playback);
    assert.equal(context.closed, 0);
    assert.deepEqual(context.samples(), (await synthesize('Hello.', {'seed': 5, 'sampleRate': 16000})).samples);
    assert(context.sources.every((source) => source.buffer.sampleRate === 16000));
});


test('speak, webkitAudioContext', async () => {
    await withAudioContext('webkitAudioContext', async () => {
        const playback = speak('Hello.');
        const [context] = FakeAudioContext.instances;
        await playThrough(context, playback);
        assert(context.sources.length > 0);
        assert.equal(context.closed, 1);
    });
});


test('speak, played as synthesized', async () => {
    // Each block ends as soon as it starts (synthesis slower than playback) - the playback finishes with the last
    const context = new FakeAudioContext({'instant': true});
    const playback = speak('Hello.', {context});
    await playback.finished;
    assert(context.sources.length > 0);
    assert(context.sources.every((source) => source.ended && !source.stopped));
});


test('speak, no speech', async () => {
    const context = new FakeAudioContext();
    await speak(' -- ', {context}).finished;
    assert.equal(context.sources.length, 0);
});


test('speak, failure', async () => {
    // A playback error rejects the finished promise
    const context = new FakeAudioContext({'failBuffer': true});
    await assert.rejects(speak('Hello.', {context}).finished, {'message': 'no buffers'});
});


//
// The package
//

test('package, browser-safe', () => {
    // The package's modules import only each other - no Node built-ins, no packages - and use no Node globals
    const files = readdirSync(new URL('../lib/', import.meta.url), {'recursive': true}).filter((file) => file.endsWith('.js'));
    assert(files.length >= 18);
    for (const file of files) {
        const source = readFileSync(new URL(`../lib/${file}`, import.meta.url), 'utf8');
        const imports = source.matchAll(/^import\s[^;]*?from\s+'([^']+)';|import\('([^']+)'\)/gm);
        const specifiers = [...imports].map((match) => match[1] ?? match[2]);
        for (const specifier of specifiers) {
            assert.match(specifier, /^\.\.?\//, `${file} imports ${specifier}`);
        }
        assert.doesNotMatch(source, /\bprocess\.|\brequire\(|getBuiltinModule|\bBuffer\b/, file);
    }
});


test('package, exports', async () => {
    // The package's entry and its modules, by the package's name
    const picovox = await import('picovox');
    assert.deepEqual(Object.keys(picovox).sort(), ['speak', 'stream', 'synthesize', 'toWav', 'voices']);
    assert.equal(picovox.synthesize, synthesize);
    const {splitSentences} = await import('picovox/lib/sentences.js');
    assert.equal(splitSentences('One. Two.').length, 2);
    const {manifest} = await import('picovox/lib/models/voiceModel.js');
    assert.equal(manifest.kind, 'multi');

    // The package metadata
    const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
    assert.equal(pkg.name, 'picovox');
    assert.equal(pkg.description, 'PicoVox - realistic speech synthesis in pure JavaScript, under 1 MB');
    assert.equal(pkg.type, 'module');
    assert.equal(pkg.main, './lib/index.js');
    assert.deepEqual(pkg.files, ['lib/**/*']);
    assert.equal(pkg.dependencies, undefined);
});
