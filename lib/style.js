// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// Style "auto" - expressive dialogue: the quotations (lib/sentences.js quoteSpans) in the dialogue style, an
// exclamation's livelier (the exclaim style), the narration neutral. Every choice follows the text, never chance, so the
// reading is deterministic.

import {quoteSpans} from './sentences.js';


// A text's automatic styles - {spans(index)}: a sentence's spans ([{text, style}] - its quotations' words in the dialogue
// style, or the exclaim style where the quoted sentence ends in "!", its narration neutral, adjacent spans of one style
// merged), or null when the sentence is all narration
export function autoStyles(sentences) {
    const sentenceSpans = quoteSpans(sentences);
    return {
        spans(index) {
            const spans = [];
            for (const span of sentenceSpans[index]) {
                let style = 'neutral';
                if (span.quotation !== null) {
                    style = /!\W*$/u.test(span.text) ? 'exclaim' : 'dialogue';
                }
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

