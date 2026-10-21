// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// Style "auto" - expressive reading by the text's structure, as a narrator reads: the quotations (lib/sentences.js
// quoteSpans) in the dialogue style, the narration neutral (the dialogue beat), and each paragraph as one arc (the
// paragraph beat). Every choice follows the text, never chance, so the reading is deterministic.

import {quoteSpans} from './sentences.js';


// A text's automatic styles - {spans(index)}: a sentence's spans ([{text, style}] - its quotations' words in the dialogue
// style, its narration neutral, adjacent spans of one style merged), or null when the sentence is all narration
export function autoStyles(sentences) {
    const sentenceSpans = quoteSpans(sentences);
    return {
        spans(index) {
            const spans = [];
            for (const span of sentenceSpans[index]) {
                const style = span.quotation === null ? 'neutral' : 'dialogue';
                const last = spans[spans.length - 1];
                if (last !== undefined && last.style === style) {
                    last.text = `${last.text} ${span.text}`;
                } else {
                    spans.push({'text': span.text, style});
                }
            }
            return spans.some((span) => span.style !== 'neutral') ? spans : null;
        }
    };
}


// The paragraph beat (style "auto") - each paragraph spoken as one arc, as a narrator reads: its first sentence lifted
// (pitch up by lift, level and frication up by level dB), the pitch settling to -lift/2 by its last sentence, which is
// slowed by settle. Tuned by ear against the flat reading (Moby-Dick's first chapter, by James).
export const paragraphBeat = {'lift': 0.08, 'level': 2, 'settle': 0.07};


// Sentences' paragraph beats - [{transforms, rate}] by sentence index (lib/synthesis.js sentenceFrames's beat): each
// sentence's place in its paragraph (lib/sentences.js paragraphStart) as paragraphBeat's arc
export function paragraphBeats(sentences) {
    const paragraphs = [];
    for (const sentence of sentences) {
        if (sentence.paragraphStart || paragraphs.length === 0) {
            paragraphs.push([]);
        }
        paragraphs[paragraphs.length - 1].push(sentence.index);
    }
    const {lift, level, settle} = paragraphBeat;
    const beats = [];
    for (const indexes of paragraphs) {
        for (const [position, index] of indexes.entries()) {
            const along = indexes.length > 1 ? position / (indexes.length - 1) : 0;
            const transforms = [{'pitchScale': 1 + lift - 1.5 * lift * along}];
            if (position === 0) {
                transforms.push({level}, {'frication': level});
            }
            beats[index] = {transforms, 'rate': indexes.length > 1 && position === indexes.length - 1 ? 1 - settle : 1};
        }
    }
    return beats;
}
