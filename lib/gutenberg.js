// Project Gutenberg plain text for reading aloud - the license header and footer, transcriber's notes, tables of
// contents, title pages, and markup removed, paragraphs unwrapped (verse lines kept), and chapter headings found,
// labeled, and listed
//
// The text is paragraphs separated by blank lines, verse lines separated by single newlines: a title paragraph, a
// "By <author>." paragraph, then the book. Headings are paragraphs ending in a period, and the level 1 and 2 headings are
// the chapters, {title, char, level} - the heading without its period, its character offset in the text, and its level.


// Decode a plain text file's bytes - 'utf-8' (a file that isn't valid UTF-8 is decoded as Windows-1252, as older
// Latin-1 files are) or another TextDecoder encoding (e.g. 'windows-1252')
export function decodeText(bytes, encoding = 'utf-8') {
    if (encoding.toLowerCase() === 'utf-8') {
        try {
            return new TextDecoder('utf-8', {'fatal': true}).decode(bytes);
        } catch {
            // Not UTF-8 - an older Latin-1 file
        }
        return new TextDecoder('windows-1252').decode(bytes);
    }
    return new TextDecoder(encoding).decode(bytes);
}



// A Gutendex title for display - one line, without a long or generic subtitle
export function displayTitle(title) {
    let result = title.replace(/\s*[\r\n]+\s*/g, (match, offset) => (/[;:,]$/.test(title.slice(0, offset)) ? ' ' : ': ')).trim();
    result = result.replace(/[:;,]\s+(?:a|an)\s+(?:novel|romance|tale|story|play|poem)\.?$/i, '');
    if (result.length > 40) {
        const match = result.match(/^(.{3,}?)\s*(?:[:;]|\s—\s)/);
        if (match) {
            [, result] = match;
        }
    }
    return result.trim();
}


// A Gutendex author name ('Last, First (Full Names), suffix') in display order ('First Last')
export function displayAuthor(name) {
    const parts = name.replace(/\s*\([^)]*\)/g, '').split(/\s*,\s*/).filter((part) => part !== '');
    if (parts.length < 2) {
        return parts.join('');
    }
    // An epithet ('Marcus Aurelius, Emperor of Rome') isn't a first name
    if (/^(?:Emperor|Empress|King|Queen|Saint|St\.|Pope|Prince|Princess|Duke|Earl|Lord|Lady|Baron|Count|Countess)\b/.test(parts[1])) {
        return parts[0];
    }
    const suffix = parts.slice(2).find((part) => /^(?:Jr\.?|Sr\.?|[IVX]+)$/.test(part));
    return `${parts[1]} ${parts[0]}${suffix ? ` ${suffix}` : ''}`;
}


// Heading patterns - labels, ordinals (arabic, roman, and words), and front and back matter
const labelRanks = {
    'book': 0, 'volume': 0, 'part': 0, 'series': 0,
    'chapter': 1, 'letter': 1, 'stave': 1, 'adventure': 1, 'canto': 1, 'act': 1, 'story': 1, 'episode': 1,
    'section': 2, 'scene': 2
};
const casings = (word) => [word.toUpperCase(), `${word[0].toUpperCase()}${word.slice(1)}`, word];
const labelPattern = Object.keys(labelRanks).flatMap((label) => casings(label).slice(0, 2)).join('|');
const numberWords = [
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
    'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty'
];
const ordinalWords = [
    'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh',
    'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth',
    'twentieth'
];
const tensWords = ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
const wordNumber = `(?:${[...tensWords, ...numberWords, ...ordinalWords, 'last', 'final'].flatMap(casings).join('|')})`;
const theWord = '(?:the|The|THE)\\s+';
const ordinalPattern = `\\d{1,3}|[IVXLCDM]{1,7}|(?:${theWord})?${wordNumber}(?:[- ]${wordNumber})?`;
const labelHeadingRegex = new RegExp(
    `^(${labelPattern})\\.?\\s+(?:No\\.?\\s*)?(${ordinalPattern})(?![\\w'’])(\\.?\\s*(?:[—–:-]+\\s*)?)(.*)$`
);
const wordLabelHeadingRegex = new RegExp(`^((?:${theWord})?${wordNumber})\\s+(${labelPattern})(?![\\w'’])\\.?\\s*(?:[—–:.-]+\\s*(.*))?$`);
const ordinalHeadingRegex = /^(\d{1,3}|[IVXLCDM]{1,7})(?:(?:\.?\s*[—–]|\.|:)\s*(.*))?$/;
const denseHeadingRegex = new RegExp(`(?:^|\\s)(?:${labelPattern})\\.?\\s+(?:\\d{1,3}|[IVXLCDM]{1,7})(?![\\w'’])`, 'g');
const matterHeadingRegex = new RegExp('^((?:THE |The )?(?:AUTHOR[\'’]S |Author[\'’]s )?(?:PROLOGUE|EPILOGUE|PREFACE|INTRODUCTION|' +
    'FOREWORD|AFTERWORD|CONCLUSION|APPENDIX|POSTSCRIPT|DEDICATION|INDUCTION|PREAMBLE|Prologue|Epilogue|Preface|' +
    'Introduction|Foreword|Afterword|Conclusion|Appendix|Postscript|Dedication|Induction|Preamble))(?![\\w\'’])\\.?\\s*' +
    '(?:[—–:.-]+\\s*)?(.*)$');
const creditRegex = new RegExp('^(?:(?:Produced|Prepared|Transcribed|Scanned|Proofread|Digitized|Converted)\\s+(?:by|from|for)\\b|' +
    'E-?(?:text|book) (?:prepared|produced) by|This e-?(?:text|book) was (?:prepared|produced|created)|HTML version by)', 'i');
const romanRegex = /^M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$/;
const contentsRegex = /^(?:TABLE OF CONTENTS|Table of Contents|CONTENTS|Contents)[.:]?$/;
const illustrationsRegex = new RegExp('^(?:LIST OF (?:THE )?(?:ILLUSTRATIONS|PLATES|COLOU?RED PLATES)|' +
    'List of (?:the )?(?:Illustrations|Plates)|ILLUSTRATIONS|Illustrations)[.:]?$');


// A heading's ordinal as a number ('XIV' -> 14, 'the Second' -> 2, 'twenty-one' -> 21), or null
function ordinalNumber(ordinal) {
    if (/^\d+$/.test(ordinal)) {
        return Number(ordinal);
    }
    if (/^[IVXLCDM]+$/.test(ordinal)) {
        if (!romanRegex.test(ordinal)) {
            return null;
        }
        const values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000};
        let value = 0;
        for (let ix = 0; ix < ordinal.length; ix++) {
            const digit = values[ordinal[ix]];
            value += (ix + 1 < ordinal.length && values[ordinal[ix + 1]] > digit) ? -digit : digit;
        }
        return value;
    }
    let value = 0;
    for (const word of ordinal.toLowerCase().replace(/^the\s+/, '').split(/[- ]/)) {
        if (tensWords.includes(word)) {
            value += 20 + 10 * tensWords.indexOf(word);
        } else if (numberWords.includes(word)) {
            value += numberWords.indexOf(word) + 1;
        } else if (ordinalWords.includes(word)) {
            value += ordinalWords.indexOf(word) + 1;
        } else if (word === 'last' || word === 'final') {
            return null;
        }
    }
    return value;
}


// Match a line as a heading - {label, ordinal, number, title, rank, matter} or null
function matchHeading(line) {
    if (line.length > 90 || !/^[A-Z0-9]/.test(line)) {
        return null;
    }
    // A label and ordinal ('CHAPTER I.', 'Book the First—Recalled to Life') - a title after a word ordinal needs a
    // separator ('Part One of the...' isn't a heading)
    let match = line.match(labelHeadingRegex);
    if (match && (match[4] === '' || /[.:—–-]/.test(match[3]) || (/^[\dIVXLCDM]+$/.test(match[2]) && /^[\p{Lu}\d“‘"'(]/u.test(match[4])))) {
        const number = ordinalNumber(match[2]);
        if (number !== null || /^(?:the\s+)?(?:last|final)$/i.test(match[2])) {
            const heading = {'label': match[1], 'ordinal': match[2], number, 'title': match[4].trim(),
                'rank': labelRanks[match[1].toLowerCase()], 'matter': false};
            // A nested heading in the title ('ACT I. SCENE 1. Rousillon.')
            const nested = heading.title.match(labelHeadingRegex);
            if (nested && labelRanks[nested[1].toLowerCase()] > heading.rank && ordinalNumber(nested[2]) !== null) {
                heading.nested = {'label': nested[1], 'ordinal': nested[2], 'number': ordinalNumber(nested[2]), 'title': nested[4].trim(),
                    'rank': labelRanks[nested[1].toLowerCase()]};
            }
            return heading;
        }
    }
    match = line.match(wordLabelHeadingRegex);
    if (match) {
        return {'label': match[2], 'ordinal': match[1], 'number': ordinalNumber(match[1]), 'title': (match[3] || '').trim(),
            'rank': labelRanks[match[2].toLowerCase()], 'matter': false, 'ordinalFirst': true};
    }
    // An ordinal, alone or before a title ('I. The Period') - not a list item ('1. Take two eggs.')
    match = line.match(ordinalHeadingRegex);
    if (match && ordinalNumber(match[1]) !== null && !/^[LCDM]$/.test(match[1]) &&
        (!match[2] || (isTitleLike(match[2]) && !/[;,]\s*[IVXLC]+\.\s/.test(match[2])))) {
        const title = (match[2] || '').trim();
        return {'label': null, 'ordinal': match[1], 'number': ordinalNumber(match[1]), title, 'rank': null, 'matter': false};
    }
    match = line.match(matterHeadingRegex);
    if (match && match[2].length <= 60) {
        return {'label': match[1], 'ordinal': '', 'number': null, 'title': match[2].trim(), 'rank': 0, 'matter': true};
    }
    return null;
}


// A comparison key for a heading line - lower case letters and digits
function headingKey(text) {
    return text.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '');
}


// Is the text title-like - capitalized words, or all capitals
function isTitleLike(text) {
    if (!/^[\p{Lu}\d“‘"'(]/u.test(text) || /[,;]$/.test(text)) {
        return false;
    }
    const letters = text.replace(/[^\p{L}]/gu, '');
    if (letters.length > 1 && letters === letters.toUpperCase()) {
        return true;
    }
    const lastWord = text.toLowerCase().replace(/[^\p{L}\s]/gu, '').trim().split(/\s+/).at(-1);
    if (smallWords.has(lastWord)) {
        return false;
    }
    const words = text.split(/\s+/).map((word) => word.replace(/^[^\p{L}]+/u, '')).filter((word) => word.length >= 4);
    return words.length > 0 && words.filter((word) => /^\p{Lu}/u.test(word)).length >= 0.6 * words.length;
}


// Title case for an ALL-CAPS heading - small words lower case, roman numerals kept
const smallWords = new Set(['a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'from', 'in', 'into', 'nor', 'of', 'on',
    'or', 'the', 'to', 'with', 'upon', 'o\'er']);
const notRoman = new Set(['MIX', 'DIM', 'MID', 'LID', 'DID', 'MILD', 'VIM', 'DI', 'MI', 'LI', 'CI', 'LIV', 'DIV', 'MIC',
    'MIL', 'CIVIL', 'LIVID', 'VIVID', 'MIMIC', 'DIX']);
function titleCase(text) {
    // All capitals, or mostly ('JONATHAN HARKER'S JOURNAL—continued') - title case the all-caps words
    const letters = text.replace(/[^\p{L}]/gu, '');
    const capitals = letters.replace(/[^\p{Lu}]/gu, '');
    if (letters.length < 2 || capitals.length < 0.7 * letters.length) {
        return text;
    }
    let first = true;
    return text.replace(/[\p{L}\d'’]+/gu, (word, offset) => {
        if (word !== word.toUpperCase()) {
            first = false;
            return word;
        }
        const before = text.slice(0, offset);
        const isFirst = first || /[:.!?—(]\s*$/.test(before) || /^\s*$/.test(before);
        first = false;
        if (romanRegex.test(word) && word.length >= 2 && !notRoman.has(word)) {
            return word;
        }
        const lower = word.toLowerCase();
        if (!isFirst && smallWords.has(lower) && !/-\s*$/.test(before)) {
            return lower;
        }
        // O'Brien, D'Arcy - otherwise lower case after an apostrophe (Engineer's)
        return lower.replace(/^(\p{L})(?:(['’])(\p{L}))?/u, (match, initial, apostrophe, next) => (
            apostrophe && match.length === 3 && /^[od]$/.test(initial)
                ? `${initial.toUpperCase()}${apostrophe}${next.toUpperCase()}`
                : initial.toUpperCase() + (apostrophe ? apostrophe + next : '')
        ));
    });
}


// End a heading as a sentence - keep a final '.', '?', or '!' (inside closing quotes or parentheses)
function headingSentence(text) {
    const trimmed = text.replace(/[\s,;:—–-]+$/, '');
    return /[.!?]["'”’)]*$/.test(trimmed) ? trimmed : `${trimmed}.`;
}


// Prepare a raw Project Gutenberg text for reading aloud - {text, chapters, title, author}
export function prepareText(raw, book = {}) {
    let lines = raw.replace(/^\uFEFF/, '').replace(/\r\n?/g, '\n').replace(/[\u00A0\t]/g, ' ').split('\n');

    // Find the header and footer markers - keep everything if there are none
    let start = 0;
    let end = lines.length;
    for (let ix = 0; ix < lines.length; ix++) {
        const line = lines[ix];
        if (/^\s*\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG/i.test(line) || /^\s*\*END\*THE SMALL PRINT/i.test(line)) {
            // The marker may continue onto the next lines, to its closing '***'
            let markerEnd = ix;
            while (markerEnd < ix + 3 && markerEnd + 1 < lines.length && !/\*{3}\s*$|\*END\*\s*$/.test(lines[markerEnd]) &&
                lines[markerEnd + 1].trim() !== '') {
                markerEnd++;
            }
            start = markerEnd + 1;
        } else if (start > 0 && (/^\s*\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG/i.test(line) ||
            /^\s*End of (?:the |this )?Project Gutenberg/i.test(line))) {
            end = ix;
            break;
        }
    }
    if (start === 0) {
        const endIndex = lines.findIndex((line) => /^\s*(?:\*{3}\s*)?END OF (?:THE |THIS )?PROJECT GUTENBERG/i.test(line));
        if (endIndex >= 0) {
            end = endIndex;
        }
    }

    // The title and author from the header - 'Title:' and 'Author:' lines, with indented continuation lines
    const header = lines.slice(0, start);
    const headerField = (name) => {
        const index = header.findIndex((line) => line.startsWith(`${name}:`));
        if (index < 0) {
            return '';
        }
        const parts = [header[index].slice(name.length + 1).trim()];
        for (let ix = index + 1; ix < header.length && /^\s+\S/.test(header[ix]) && !/^\s*\w+:/.test(header[ix]); ix++) {
            parts.push(header[ix].trim());
        }
        return parts.filter((part) => part !== '').reduce((title, part) => {
            if (/^(?:or|and),?\s/i.test(part)) {
                return `${title.replace(/[;,]$/, '')}, ${part}`;
            }
            return /[;:,]$/.test(title) ? `${title} ${part}` : `${title}: ${part}`;
        });
    };
    const headerTitle = headerField('Title');
    const headerAuthor = headerField('Author');
    const title = (headerTitle || book.title || '').replace(/\s*;\s*(or|Or),?\s+/, ', or ').replace(/\s+/g, ' ').trim();
    const author = (headerAuthor || book.author || '').replace(/\s*\([^)]*\)/g, '').replace(/\s+/g, ' ').trim();

    // Drop bracketed illustration, footnote, and transcriber blocks (possibly several lines) and the content notes
    lines = lines.slice(start, end);
    const kept = [];
    for (let ix = 0; ix < lines.length; ix++) {
        let line = lines[ix];
        if (/^\s*\*\s*CONTENT NOTE/.test(line)) {
            while (ix < lines.length && !/^\s*\*\s*END CONTENT NOTE/.test(lines[ix])) {
                ix++;
            }
            continue;
        }
        const blockMatch = line.match(/^(\s*)\[(?:Illustration|Footnote|Transcriber|Sidenote|Decoration|Image|Note:)/i);
        if (blockMatch) {
            // Skip to the closing bracket
            let depth = 0;
            let rest = '';
            let jx = ix;
            for (; jx < lines.length && jx < ix + 60; jx++) {
                const text = lines[jx];
                let closed = -1;
                for (let cx = (jx === ix ? blockMatch[1].length : 0); cx < text.length; cx++) {
                    if (text[cx] === '[') {
                        depth++;
                    } else if (text[cx] === ']') {
                        depth--;
                        if (depth === 0) {
                            closed = cx;
                            break;
                        }
                    }
                }
                if (closed >= 0) {
                    rest = text.slice(closed + 1);
                    break;
                }
            }
            if (jx < lines.length && jx < ix + 60) {
                ix = jx;
                line = rest;
                if (line.trim() === '') {
                    // Keep a paragraph break where the block was
                    kept.push('');
                    continue;
                }
            }
        }
        // Act and scene markers ('ACT_3|SC_1') separate paragraphs
        if (/^\s*ACT_\d+\|SC_\d+\s*$/.test(line)) {
            kept.push('');
            continue;
        }
        kept.push(line.replace(/\[(?:Illustration|Decoration|Image)[^\]]*\]/gi, ''));
    }

    // Split into paragraphs - lines with their indentation, and the blank lines before
    const paragraphs = [];
    let blanks = 0;
    let current = null;
    for (const line of kept) {
        const text = line.trimEnd();
        if (text.trim() === '') {
            blanks++;
            current = null;
            continue;
        }
        if (current === null) {
            current = {'lines': [], blanks};
            paragraphs.push(current);
            blanks = 0;
        }
        current.lines.push({'text': cleanLine(text.trim()), 'indent': text.length - text.trimStart().length, 'length': text.length});
    }

    // Mark the paragraphs to drop - separators, credits, license notes, footnotes, and author lines ('by Bram Stoker')
    const authorWords = new Set([...wordsOf(author), ...wordsOf(headerAuthor), ...wordsOf(book.author || '')]);
    const titleWords = new Set([...wordsOf(title), ...wordsOf(book.title || ''), ...authorWords,
        'by', 'the', 'a', 'an', 'of', 'and', 'or', 'with', 'sir', 'mrs', 'mr', 'dr', 'illustrated', 'illustrations',
        'complete', 'author', 'written', 'unabridged']);
    for (const paragraph of paragraphs) {
        const text = paragraph.lines.map((line) => line.text).join(' ').replace(/\s+/g, ' ').trim();
        paragraph.text = text;
        paragraph.key = paragraph.lines.length === 1 ? headingKey(text) : null;
        if (text === '' || /^[\s*.·•~=_#+—…-]+$/.test(text)) {
            paragraph.drop = true;
        } else if (text.length < 600 && creditRegex.test(text)) {
            paragraph.drop = true;
        } else if (/Project Gutenberg/i.test(text) && /e-?book|e-?text|license|copyright|www\.gutenberg/i.test(text)) {
            paragraph.drop = true;
        } else if (/^\[\d+\]\s/.test(text) || /^Footnote \d+:/i.test(text) || /^(?:FOOTNOTES?|Footnotes?)[.:]?$/.test(text)) {
            paragraph.drop = true;
        } else if (/^by\s/i.test(text) && text.length < 120 && wordsOf(text).every((word) => authorWords.has(word) || word === 'by')) {
            paragraph.drop = true;
        }
    }

    // Drop transcriber's notes - a notes heading and its paragraphs up to a wider gap or a heading (or, near the end, to
    // the end)
    for (let ix = 0; ix < paragraphs.length; ix++) {
        const paragraph = paragraphs[ix];
        if (/^\[?(?:Original )?Transcriber['’]?s['’]? Notes?\b/i.test(paragraph.text) ||
            (/\btranscriber\b/i.test(paragraph.text) && paragraph.text.length < 120 && /:$/.test(paragraph.text))) {
            paragraph.drop = true;
            if (paragraph.text.length < 120) {
                const atEnd = ix > 0.97 * paragraphs.length;
                const scanEnd = atEnd ? paragraphs.length : ix + 16;
                for (let jx = ix + 1; jx < paragraphs.length && jx < scanEnd; jx++) {
                    if (!atEnd && (paragraphs[jx].blanks >= 2 && jx > ix + 1 || matchHeading(paragraphs[jx].text))) {
                        break;
                    }
                    paragraphs[jx].drop = true;
                }
            }
        }
    }
    const live = paragraphs.filter((paragraph) => !paragraph.drop);

    // The last paragraph index of each short line, for confirming table of contents entries
    const lastIndex = new Map();
    for (let ix = 0; ix < live.length; ix++) {
        for (const line of live[ix].lines) {
            if (line.text.length <= 100) {
                lastIndex.set(headingKey(line.text), ix);
            }
        }
    }

    // Remove tables of contents - a contents heading and its entries (headings, or lines repeated later in the text);
    // the entries that aren't numbered headings are the book's own section titles ('THE SONNETS', 'Economy'), and the
    // numbered entries' titles name untitled chapters
    const tocKeys = new Map();
    const tocTitles = new Map();
    const stripPage = (text) => text.replace(/(?:\s{2,}|\s*\.{2,}\s*|\s+\.\s+)(?:\d+|[ivxlc]+)$/i, '').replace(/\s+\.?$/, '').trim();
    for (let ix = 0; ix < live.length; ix++) {
        const paragraph = live[ix];
        const headingLines = paragraph.lines.filter((line) => matchHeading(stripPage(line.text))).length;
        if (paragraph.lines.length >= 3 && headingLines >= 0.75 * paragraph.lines.length) {
            // A run of numbered headings in one paragraph
            paragraph.drop = true;
            continue;
        }

        // A list of illustrations - short lines up to a heading or prose
        if (illustrationsRegex.test(paragraph.text)) {
            paragraph.drop = true;
            for (let jx = ix + 1; jx < live.length; jx++) {
                const entry = live[jx];
                if (matchHeading(entry.lines[0].text) || contentsRegex.test(entry.text) || entry.text.length > 300 ||
                    entry.lines.some((line) => line.text.length > 80)) {
                    break;
                }
                entry.drop = true;
            }
            continue;
        }
        if (!contentsRegex.test(paragraph.text)) {
            continue;
        }
        paragraph.drop = true;
        const entries = [];
        let firstKey = null;
        let skipped = [];
        for (let jx = ix + 1; jx < live.length; jx++) {
            const entry = live[jx];
            const key = headingKey(stripPage(entry.lines[0].text));
            if (firstKey !== null && key === firstKey) {
                break;
            }

            // An entry is a heading, or a line repeated later - after a list of only numbered headings, a title
            // ('Dramatis Personæ') ends it
            const onlyNumbered = entries.length >= 3 && entries.every((item) => item.heading);
            const previousEntry = entries.at(-1);
            const titlePending = previousEntry && previousEntry.heading && !previousEntry.heading.matter &&
                previousEntry.heading.title === '';
            // Or a run of headings wrapped as prose ('ACT I Scene I. Elsinore. ... Scene II. ...')
            const dense = entry.text.length < 1000 && matchHeading(entry.lines[0].text.split(/\s+/).slice(0, 2).join(' ')) &&
                (entry.text.match(denseHeadingRegex) || []).length >= 2;
            const isEntry = dense || entry.lines.every((line) => {
                const text = stripPage(line.text);
                const lineKey = headingKey(text);
                const isRepeated = (!onlyNumbered || titlePending) && lineKey !== '' && (lastIndex.get(lineKey) || 0) > jx;
                return text.length <= 100 && (matchHeading(text) || isRepeated || /^(?:page|chap(?:ter)?\.?)$/i.test(text));
            });
            if (!isEntry) {
                // Allow a few short unconfirmed lines between entries
                if (entry.lines.length === 1 && entry.text.length < 100 && skipped.length < 3) {
                    skipped.push(entry);
                    continue;
                }
                break;
            }
            for (const skip of skipped) {
                skip.drop = true;
            }
            skipped = [];
            entry.drop = true;
            for (const line of (dense ? [entry.lines[0]] : entry.lines)) {
                const text = stripPage(line.text);
                if (/^(?:page|chap(?:ter)?\.?)$/i.test(text) || text === '') {
                    continue;
                }
                if (firstKey === null) {
                    firstKey = headingKey(text);
                }
                // A title line after an untitled numbered entry is its title ('CHAPTER I' / 'Jonathan Harker's Journal')
                const heading = matchHeading(text);
                const previous = entries.at(-1);
                if (!heading && previous && previous.heading && !previous.heading.matter && previous.heading.title === '') {
                    previous.heading.title = text;
                } else {
                    entries.push({text, heading, dense});
                }
            }
        }

        // Rank the section titles - above numbered headings (plays and their acts), beside them (Moby Dick's
        // 'Etymology'), and all-caps titles above mixed-case ones (Walden's books and chapters)
        const titles = entries.filter((entry) => !entry.heading && /\p{L}/u.test(entry.text));
        const hasNumbered = entries.some((entry) => entry.heading);
        const isCaps = (text) => text.replace(/[^\p{L}]/gu, '') === text.replace(/[^\p{L}]/gu, '').toUpperCase();
        const mixedCaps = titles.some((entry) => isCaps(entry.text)) && titles.some((entry) => !isCaps(entry.text));
        for (const entry of titles) {
            const key = headingKey(entry.text);
            if (key !== '' && !tocKeys.has(key)) {
                const rank = mixedCaps && isCaps(entry.text) ? -2 : -1;
                tocKeys.set(key, hasNumbered ? {'rank': 0, 'leaf': true} : {rank, 'leaf': false});
            }
        }
        for (const entry of entries) {
            if (entry.heading && !entry.dense && !entry.heading.nested && entry.heading.number !== null && entry.heading.title !== '' &&
                (entry.heading.rank ?? 1) === 1) {
                tocTitles.set(entry.heading.number, tocTitles.has(entry.heading.number) ? null : entry.heading.title);
            }
        }
    }
    let body = live.filter((paragraph) => !paragraph.drop);

    // Drop what follows the book's last 'THE END' (publishers' advertisements, printers' notes)
    for (let ix = body.length - 1; ix >= 0.9 * body.length; ix--) {
        if (/^(?:THE END|The End|FINIS|Finis)\.?$/.test(body[ix].text)) {
            body = body.slice(0, ix + 1);
            break;
        }
    }

    // Remove line numbers from the ends of verse lines ('...convey’d;     1192')
    for (const paragraph of body) {
        if (paragraph.lines.length >= 3 && paragraph.lines.some((line) => /\S\s{2,}\d{1,4}$/.test(line.text))) {
            for (const line of paragraph.lines) {
                const stripped = line.text.replace(/\s{2,}\d{1,4}$/, '');
                line.length -= line.text.length - stripped.length;
                line.text = stripped;
            }
            paragraph.text = paragraph.lines.map((line) => line.text).join(' ').replace(/\s+/g, ' ').trim();
        }
    }

    // Verse paragraphs - most line breaks come before a word that would have fit on the line (the wrap width is a
    // high percentile of the lengths of lines followed by another line)
    const lengths = [];
    for (let ix = 0; ix < body.length; ix += Math.max(1, Math.floor(body.length / 4000))) {
        for (const line of body[ix].lines.slice(0, -1)) {
            lengths.push(line.length);
        }
    }
    lengths.sort((first, second) => first - second);
    const wrapWidth = Math.min(100, Math.max(55, lengths[Math.floor(lengths.length * 0.8)] || 72));
    let verseCount = 0;
    let multiLineCount = 0;
    for (const paragraph of body) {
        const paragraphLines = paragraph.lines;
        const breaks = paragraphLines.slice(0, -1)
            .filter((line, ix) => line.length + 1 + paragraphLines[ix + 1].text.split(' ')[0].length <= wrapWidth - 3).length;
        // Or every line capitalized, with a break or several lines (Whitman's long lines)
        // Prose continues on lower case lines, verse lines are capitalized
        const capitals = paragraphLines.slice(1).filter((line) => /^[“‘"'(]?\p{Lu}/u.test(line.text)).length;
        const lineCount = paragraphLines.length;
        paragraph.verse = lineCount >= 2 && capitals >= 0.6 * (lineCount - 1) &&
            (breaks >= 0.5 * (lineCount - 1) || (capitals === lineCount - 1 && (breaks > 0 || lineCount >= 4)));
        if (paragraphLines.length >= 2) {
            multiLineCount++;
            verseCount += paragraph.verse ? 1 : 0;
        }
    }

    // Find the headings
    const isPlay = body.some((paragraph) => /^(?:ACT|Act)\s+(?:I|1|One|ONE)\b/.test(paragraph.text));
    const isVerse = !isPlay && verseCount >= 0.5 * multiLineCount;
    for (let ix = 0; ix < body.length; ix++) {
        const paragraph = body[ix];
        const firstLine = paragraph.lines[0].text;
        if (paragraph.lines.length > 3 || firstLine.length > 90) {
            continue;
        }
        let heading = matchHeading(firstLine);
        if (heading && paragraph.lines.length > 1) {
            // A heading with its title on the following lines ('CHAPTER I.' / 'DOWN THE RABBIT-HOLE')
            const rest = paragraph.lines.slice(1).map((line) => line.text).join(' ');
            if (heading.nested && heading.nested.title === '' && rest.length <= 80) {
                heading.nested.title = rest;
            } else if (heading.title !== '' || rest.length > 80 || !isTitleLike(rest)) {
                heading = null;
            } else {
                heading.title = rest;
            }
        }
        if (!heading && paragraph.key !== null && tocKeys.has(paragraph.key)) {
            const toc = tocKeys.get(paragraph.key);
            heading = {'label': null, 'ordinal': '', 'number': null, 'title': firstLine, 'rank': toc.rank, 'matter': toc.leaf,
                'section': true};
        }
        if (!heading && isVerse && paragraph.lines.length === 1 && firstLine.length <= 60 && !/[.,;:]$/.test(firstLine) &&
            isTitleLike(firstLine) && ix + 1 < body.length && body[ix + 1].verse &&
            paragraph.lines[0].indent <= Math.min(...body[ix + 1].lines.map((line) => line.indent))) {
            // A poem's title
            heading = {'label': null, 'ordinal': '', 'number': null, 'title': firstLine, 'rank': 1, 'matter': false, 'section': true,
                'poemTitle': true};
        }
        if (heading) {
            // A bare number is a heading only before a longer paragraph
            if (heading.label === null && !heading.section && heading.title === '' && /^\d+$/.test(heading.ordinal) &&
                !(ix + 1 < body.length && body[ix + 1].lines.length >= 2)) {
                continue;
            }
            paragraph.heading = heading;
        }
    }

    // Remove runs of three or more headings with the same label and nothing between them - contents lists without a
    // contents heading
    for (let ix = 0; ix < body.length;) {
        let jx = ix;
        while (jx < body.length && body[jx].heading && !body[jx].heading.section) {
            jx++;
        }
        if (jx - ix >= 3) {
            const counts = new Map();
            for (let kx = ix; kx < jx; kx++) {
                const label = (body[kx].heading.label || '#').toLowerCase();
                counts.set(label, (counts.get(label) || 0) + 1);
            }
            if (Math.max(...counts.values()) >= 3) {
                for (let kx = ix; kx < jx; kx++) {
                    body[kx].drop = true;
                }
            }
        }
        ix = Math.max(jx, ix + 1);
    }
    body = body.filter((paragraph) => !paragraph.drop);

    // Drop the book's own title page - short paragraphs of title and author words, and edition and printing notes,
    // before the first numbered heading or prose
    for (let ix = 0; ix < body.length && ix < 40; ix++) {
        const paragraph = body[ix];
        if ((paragraph.heading && !paragraph.heading.section) || paragraph.text.length >= 200) {
            break;
        }
        const words = wordsOf(paragraph.text);
        if ((words.length > 0 && words.every((word) => titleWords.has(word))) ||
            (paragraph.text.length < 160 && /\b(?:edition|printed|publishers?|copyright|all rights reserved)\b/i.test(paragraph.text))) {
            paragraph.drop = true;
        }
    }
    body = body.filter((paragraph) => !paragraph.drop);

    // Title untitled numbered headings - with a following title line ('CHAPTER I' / 'Down the Rabbit-Hole'), or the
    // table of contents' title for the number
    const labelOnlyRegex = new RegExp(`^(?:${labelPattern})\\.?$`);
    for (let ix = 0; ix + 1 < body.length; ix++) {
        const {heading} = body[ix];
        if (!heading || heading.section || heading.matter || heading.title !== '') {
            continue;
        }
        const next = body[ix + 1];
        if (ix + 2 < body.length && (!next.heading || next.heading.poemTitle) && next.lines.length === 1 && next.text.length <= 70 &&
            isTitleLike(next.text) && !labelOnlyRegex.test(next.text) && !/^(?:Enter|Exit|Exeunt|Re-enter)\b/.test(next.text)) {
            heading.title = next.text;
            next.drop = true;
        } else if ((heading.rank ?? 1) === 1 && tocTitles.get(heading.number)) {
            heading.title = tocTitles.get(heading.number);
        }
    }
    body = body.filter((paragraph) => !paragraph.drop);

    // Label, level, and format the headings - an unlabeled ordinal continues the numbering of an open heading, or starts
    // a numbering under it ('Chapter' under a book, 'Part' under a chapter, 'Sonnet' under 'The Sonnets')
    const stack = [];
    const out = [];
    const chapters = [];
    let textLength = 0;
    const addParagraph = (text) => {
        textLength += (out.length > 0 ? 2 : 0) + text.length;
        out.push(text);
    };
    if (title !== '') {
        addParagraph(headingSentence(title));
    }
    if (author !== '') {
        addParagraph(headingSentence(`By ${author}`));
    }
    for (const paragraph of body) {
        const {heading} = paragraph;
        if (!heading) {
            addParagraph(paragraph.verse ? verseText(paragraph, wrapWidth) : paragraph.text);
            continue;
        }

        // The rank and label
        let {rank, label, number, ordinal} = heading;
        if (label === null && !heading.section) {
            const continued = [...stack].reverse().find((entry) => entry.number !== null && entry.number + 1 === heading.number &&
                entry.ordinalStyle === ordinalStyle(heading.ordinal));
            if (continued) {
                ({rank, label} = continued);
            } else {
                const top = stack.filter((entry) => !entry.leaf).at(-1);
                const plural = top && top.section && top.title.match(/^(?:THE |The )?(\p{L}+?)[sS]$/u);
                if (top && top.rank >= 1) {
                    rank = top.rank + 1;
                    label = isVerse ? 'Poem' : 'Part';
                } else {
                    rank = 1;
                    label = plural ? plural[1] : (isVerse ? 'Part' : 'Chapter');
                }
            }
        } else if (heading.nested && stack.some((entry) => entry.label && entry.label.toLowerCase() === label.toLowerCase() &&
            entry.number === heading.number)) {
            // A later scene of an open act ('ACT I. SCENE 2.')
            ({rank, label, number, ordinal} = heading.nested);
        }

        // The level - the open headings above it
        // (front and back matter stays inside an open book or series)
        const popRank = heading.matter && !heading.section ? Math.max(rank, 1) : rank;
        while (stack.length > 0 && (stack.at(-1).leaf || stack.at(-1).rank >= popRank)) {
            stack.pop();
        }
        const level = stack.length + 1;
        stack.push({rank, label, number, 'ordinalStyle': ordinalStyle(ordinal), 'leaf': heading.matter,
            'section': Boolean(heading.section), 'title': heading.title});

        // The heading text - without a title that repeats the label ('EPILOGUE' / 'EPILOGUE.')
        if (heading.label && headingKey(heading.title).startsWith(headingKey(heading.label))) {
            heading.title = heading.title.slice(heading.label.length).replace(/^[\s.:—–-]+/, '');
        }
        let text;
        if (heading.section) {
            text = titleCase(heading.title);
        } else {
            const labelText = titleCase(label);
            const ordinalText = heading.ordinal ? titleCase(heading.ordinal) : '';
            text = heading.ordinalFirst ? `${ordinalText} ${labelText}` : `${labelText} ${ordinalText}`.trim();
            if (heading.nested) {
                const {nested} = heading;
                text = `${titleCase(heading.label)} ${heading.ordinal}. ${titleCase(nested.label)} ${nested.ordinal}`;
                if (nested.title !== '') {
                    text = `${text}. ${titleCase(nested.title)}`;
                }
            } else if (heading.title !== '') {
                text = `${text.replace(/[.:]$/, '')}. ${titleCase(heading.title)}`;
            }
        }
        text = headingSentence(text.replace(/\s+/g, ' '));
        if (level <= 2) {
            chapters.push({'title': text.replace(/\.$/, ''), 'char': textLength + (out.length > 0 ? 2 : 0), level});
        }
        addParagraph(text);
    }

    return {'text': out.join('\n\n'), chapters, 'title': displayTitle(title) || book.title || '', 'author': author || book.author || ''};
}


// Clean a line - italic underscores, =bold= markers, footnote markers, and '--' dashes
function cleanLine(text) {
    return text
        .replace(/_/g, '')
        .replace(/(^|[\s(“"‘'])=(?=\S)([^=]*?\S)=(?=$|[\s.,;:!?)”"’'])/g, '$1$2')
        .replace(/\s?\[(?:\d{1,3}|[*†‡§]|[A-Z])\]/g, '')
        .replace(/\s?\{\d{1,3}\}/g, '')
        .replace(/-{2,}/g, '—')
        .replace(/\*(?:\s*\*)+/g, '…')
        .replace(/(\S)\s?\*(?=\s|$)/g, '$1')
        .replace(/\s{2,}/g, '  ');
}


// A verse paragraph's text - lines kept, with indented continuations of long lines joined
function verseText(paragraph, wrapWidth) {
    const base = Math.min(...paragraph.lines.map((line) => line.indent));
    const lines = [];
    for (const line of paragraph.lines) {
        const previous = lines.at(-1);
        if (previous && line.indent >= base + 4 && previous.length >= 0.8 * wrapWidth - base) {
            lines[lines.length - 1] = `${previous} ${line.text}`;
        } else {
            lines.push(line.text);
        }
    }
    return lines.map((line) => line.replace(/\s+/g, ' ').trim()).join('\n');
}


// The kind of an ordinal - roman, arabic, or words - for continuing a numbering
function ordinalStyle(ordinal) {
    if (/^\d+$/.test(ordinal || '')) {
        return 'arabic';
    }
    return /^[IVXLCDM]+$/i.test(ordinal || '') ? 'roman' : 'word';
}


// The lower case words of a text
function wordsOf(text) {
    return (text || '').toLowerCase().match(/\p{L}+/gu) || [];
}
