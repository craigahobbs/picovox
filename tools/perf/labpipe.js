// The performance lab's pipeline workers - a sentence's network and renderer in two workers (synthesis.js sentenceFrames
// and createSentenceRenderer): the network worker posts each frame chunk to the render worker (a MessagePort) as it's
// made, so the two run at once. One script, either role (in its first message):
//
// - {'role': 'network', lib, port, texts, voice, sampleRate} - speaks the texts sentence by sentence (as textBlocks), the
//   frames to port; posts {'type': 'done', networkMs, loadMs}
// - {'role': 'render', lib, port} - renders what comes on port; posts {'type': 'done', audio, renderMs, firstMs, wallMs}
//
// On the port: {'type': 'begin', eq, seed, pause}, {'type': 'chunk', frames (Float64Array rows, flat), count}, {'type':
// 'end'}, {'type': 'finish'}.

self.onmessage = async (event) => {
    const {role, lib, port} = event.data;
    const synthesis = await import(`${lib}/synthesis.js`);
    if (role === 'network') {
        const loadStart = performance.now();
        const [lexicon, g2p, packaged, sentences, g2pModule, voiceModule] = await Promise.all(
            ['lexicon', 'g2p', 'package', 'sentences', 'models/g2pModel', 'models/voiceModel'].map((name) => import(`${lib}/${name}.js`))
        );
        lexicon.setPronouncer(await g2p.loadG2p(g2pModule.g2pModel));
        const model = await packaged.loadPackagedModel(voiceModule);
        const loadMs = performance.now() - loadStart;
        const speaker = synthesis.voiceSpeaker(model, event.data.voice ?? 'male');
        let networkMs = 0;
        port.postMessage({'type': 'start'});
        for (const text of event.data.texts) {
            for (const sentence of sentences.splitSentences(text)) {
                const pause = sentence.paragraphStart ? synthesis.paragraphPause : synthesis.sentencePause;
                let start = performance.now();
                const {chunks, eq} = synthesis.sentenceFrames(model, sentences.speechText(sentence.text), {speaker});
                port.postMessage({'type': 'begin', eq, 'seed': 1 + sentence.index, pause});
                for (const chunk of chunks) {
                    const count = chunk.length;
                    const frames = new Float64Array(count * (count ? chunk[0].length : 0));
                    for (const [index, frame] of chunk.entries()) {
                        frames.set(frame, index * frame.length);
                    }
                    networkMs += performance.now() - start;
                    port.postMessage({'type': 'chunk', frames, count}, [frames.buffer]);
                    start = performance.now();
                }
                networkMs += performance.now() - start;
                port.postMessage({'type': 'end'});
            }
        }
        port.postMessage({'type': 'finish'});
        self.postMessage({'type': 'done', networkMs, loadMs});
    } else {
        let renderer = null;
        let audio = 0;
        let renderMs = 0;
        let firstMs = null;
        let wallStart = null;
        const sampleRate = event.data.sampleRate ?? synthesis.defaultSampleRate;
        port.onmessage = ({data}) => {
            const start = performance.now();
            if (data.type === 'start') {
                wallStart = start;
            } else if (data.type === 'begin') {
                renderer = synthesis.createSentenceRenderer({'onBlock': (block) => {
                    audio += block.length / sampleRate;
                    if (firstMs === null && block.some((value) => value !== 0)) {
                        firstMs = performance.now() - wallStart;
                    }
                }, 'eq': data.eq, 'seed': data.seed, 'pause': data.pause, sampleRate});
            } else if (data.type === 'chunk') {
                const width = data.count ? data.frames.length / data.count : 0;
                const frames = Array.from({'length': data.count},
                    (_value, index) => data.frames.subarray(index * width, (index + 1) * width));
                renderer.push(frames);
            } else if (data.type === 'end') {
                renderer.end();
            } else if (data.type === 'finish') {
                self.postMessage({'type': 'done', audio, 'renderMs': renderMs + performance.now() - start, firstMs,
                    'wallMs': performance.now() - wallStart});
                return;
            }
            renderMs += performance.now() - start;
        };
    }
};
