// The text front end - normalization, tokenizing into phrases, and pronunciation. The pronouncer turns a word into
// ARPAbet phones with stress - the G2P network's (setPronouncer with lib/g2p.js's loadG2p): every word is pronounced by
// the network, with no dictionary at runtime (the voice model's training data is pronounced by it too - picovox_train/frontend.py).


// The pronouncer
let pronouncer = null;


// Set the pronouncer - pronounce(word, reading) returns the word's ARPAbet phones with stress in a reading (0, the
// default); with ambiguous(word) and chooseReading(items, index) (lib/g2p.js loadG2p), the reading of an ambiguous
// word is chosen from its context (null unsets it)
export function setPronouncer(pronounce) {
    pronouncer = pronounce;
}


// Get the pronouncer - the one set
function getPronouncer() {
    if (pronouncer === null) {
        throw new Error('No pronouncer - call setPronouncer(pronounce) first');
    }
    return pronouncer;
}


// Function words - unaccented, with reduced stress
const functionWords = new Set([
    'a', 'an', 'the', 'of', 'to', 'in', 'on', 'at', 'for', 'and', 'but', 'or', 'nor', 'is', 'are', 'was', 'were', 'be',
    'been', 'am', 'it', 'its', 'that', 'this', 'with', 'as', 'by', 'from', 'i', 'you', 'he', 'she', 'we', 'they', 'me',
    'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their', 'do', 'does', 'did', 'have', 'has', 'had', 'will',
    'would', 'can', 'could', 'shall', 'should', 'may', 'might', 'must', 'so', 'if', 'than', 'then', 'into', 'onto',
    'up', 'out', 'over', 'about', 'there', 'some', 'just'
]);


// A text's accented Latin letters without their accents - á as a, Ñ as N, and the ligatures as English says them
// (Cæsar, Œdipus; ß as s) - each letter one letter, so the text keeps its length (the front end's words are letters a
// to z)
const plainLetterMap = {'æ': 'e', 'Æ': 'E', 'œ': 'e', 'Œ': 'E', 'ø': 'o', 'Ø': 'O', 'ł': 'l', 'Ł': 'L', 'đ': 'd', 'Đ': 'D', 'ß': 's'};
function plainLetters(text) {
    return text.replace(/[\u00c0-\u024f]/g, (char) => plainLetterMap[char] ?? char.normalize('NFD').replace(/[\u0300-\u036f]/g, ''));
}


// Question words - a question starting with one ends with a fall
export const whWords = new Set(['who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom']);


// The letters whose names aren't the words the pronouncer says for them alone ("a" is the article, "i" a function word) -
// an initial's ("A. Lincoln"), and a spelled word's letters
const letterWords = {'a': 'ay', 'i': 'eye'};


// The dotted abbreviations read as words ("i.e.", "e.g.") - any other is its letters' names ("a.k.a.", "U.S.A.", "Ph.D.")
const dottedWords = {'ie': ['that', 'is'], 'eg': ['for', 'example']};


// Abbreviations
const abbreviations = {'mr': 'mister', 'mrs': 'missus', 'dr': 'doctor', 'st': 'saint', 'vs': 'versus', 'etc': 'etcetera'};

// Abbreviations only with their period - the months and days ("mon" is also French, so "Mon." only; "mar", "sat",
// "sun", and "wed" are words, so they aren't abbreviations), and the titles and the rest that don't end a sentence
// (lib/sentences.js - but not "Ms.", which is also MS., a manuscript, or "Gen." and "Ch.", also Genesis and Christ
// Church); "No." before a number is "number"
const periodAbbreviations = {
    'jan': 'january', 'feb': 'february', 'apr': 'april', 'jun': 'june', 'jul': 'july', 'aug': 'august', 'sep': 'september',
    'sept': 'september', 'oct': 'october', 'nov': 'november', 'dec': 'december', 'mon': 'monday', 'tue': 'tuesday',
    'tues': 'tuesday', 'thu': 'thursday', 'thur': 'thursday', 'thurs': 'thursday', 'fri': 'friday',
    'prof': 'professor', 'capt': 'captain', 'col': 'colonel', 'lt': 'lieutenant', 'rev': 'reverend',
    'esq': 'esquire', 'jr': 'junior', 'sr': 'senior', 'mt': 'mount', 'vol': 'volume', 'fig': 'figure'
};


// The units after a number, by their symbols as written (case matters - "MW", not "mW"; "g", not "G") - the unit's words,
// and its plural's (5 km as five kilometers, 1 km as one kilometer); "sq" and "cu" before one are "square" and "cubic"
const unitWords = {
    'km': ['kilometer', 'kilometers'], 'm': ['meter', 'meters'], 'cm': ['centimeter', 'centimeters'], 'mm': ['millimeter', 'millimeters'],
    'nm': ['nanometer', 'nanometers'], 'mi': ['mile', 'miles'], 'ft': ['foot', 'feet'], 'yd': ['yard', 'yards'],
    'kg': ['kilogram', 'kilograms'], 'g': ['gram', 'grams'], 'mg': ['milligram', 'milligrams'], 'lb': ['pound', 'pounds'],
    'lbs': ['pounds', 'pounds'], 'oz': ['ounce', 'ounces'], 'ml': ['milliliter', 'milliliters'], 'mL': ['milliliter', 'milliliters'],
    'ha': ['hectare', 'hectares'], 'mph': ['mile per hour', 'miles per hour'], 'kW': ['kilowatt', 'kilowatts'],
    'MW': ['megawatt', 'megawatts'], 'GW': ['gigawatt', 'gigawatts'], 'Hz': ['hertz', 'hertz'], 'kHz': ['kilohertz', 'kilohertz'],
    'MHz': ['megahertz', 'megahertz'], 'GHz': ['gigahertz', 'gigahertz'], 'kPa': ['kilopascal', 'kilopascals'],
    'ppm': ['part per million', 'parts per million'], 'psi': ['pound per square inch', 'pounds per square inch'],
    'AU': ['astronomical unit', 'astronomical units']
};
const unitPrefixes = {'sq': 'square', 'cu': 'cubic'};

// The units written with a slash or a power, anywhere (km/h, m/s, km2, m³) - the lengths, the times, and the powers
const unitLengths = {'km': 'kilometer', 'm': 'meter', 'cm': 'centimeter', 'mm': 'millimeter', 'mi': 'mile', 'ft': 'foot'};
const unitTimes = {'h': 'hour', 'hr': 'hour', 's': 'second', 'min': 'minute'};
const unitPowers = {'2': 'square', '²': 'square', '3': 'cubic', '³': 'cubic'};

// The symbols read as words - a minus sign before a number too
const symbolWords = {
    '&': ['and'], '+': ['plus'], '=': ['equals'], '×': ['times'], '±': ['plus', 'or', 'minus'], '≈': ['approximately'],
    '<': ['less', 'than'], '>': ['greater', 'than'], '~': ['about'], '−': ['minus'], '-': ['minus']
};


// A unit's plural - its last word's ("foot" as "feet")
function unitPlural(unit) {
    return unit === 'foot' ? 'feet' : `${unit}s`;
}


// Number words
const ones = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
    'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
const tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];


// Spell an integer in words
export function numberWords(value) {
    if (value < 20) {
        return [ones[value]];
    }
    if (value < 100) {
        return value % 10 === 0 ? [tens[Math.floor(value / 10)]] : [tens[Math.floor(value / 10)], ones[value % 10]];
    }
    for (const [size, name] of [[1e12, 'trillion'], [1e9, 'billion'], [1e6, 'million'], [1e3, 'thousand'], [100, 'hundred']]) {
        if (value >= size) {
            const rest = value % size;
            return [...numberWords(Math.floor(value / size)), name, ...(rest === 0 ? [] : numberWords(rest))];
        }
    }
    return [];
}


// Spell a year in words - by its hundreds (1972 as nineteen seventy-two, 1900 as nineteen hundred, 1905 as nineteen oh
// five), but a millennium's first years as a number (2000, 2001 as two thousand one)
export function yearWords(value) {
    const hundreds = Math.floor(value / 100);
    const rest = value % 100;
    if (hundreds % 10 === 0 && rest < 10) {
        return numberWords(value);
    }
    return [...numberWords(hundreds), ...(rest === 0 ? ['hundred'] : (rest < 10 ? ['oh', ones[rest]] : numberWords(rest)))];
}


// A number word as an ordinal (twenty as twentieth) or a plural (twenty as twenties)
const ordinalWords = {
    'one': 'first', 'two': 'second', 'three': 'third', 'five': 'fifth', 'eight': 'eighth', 'nine': 'ninth', 'twelve': 'twelfth'
};
function ordinalWord(word) {
    return ordinalWords[word] ?? (word.endsWith('y') ? `${word.slice(0, -1)}ieth` : `${word}th`);
}
function pluralWord(word) {
    return word.endsWith('y') ? `${word.slice(0, -1)}ies` : (word === 'six' ? 'sixes' : `${word}s`);
}


// Digits in words, digit by digit
function digitWords(digits) {
    return [...digits].map((digit) => ones[Number(digit)]);
}


// An integer's digits (commas grouping its thousands or not) in words - digit by digit past 15 digits (a number that
// long isn't read as one, and is past a double's exact integers)
function integerWords(digits) {
    const plain = digits.replace(/,/g, '').replace(/^0+(?=\d)/, '');
    return plain.length > 15 ? digitWords(plain) : numberWords(Number(plain));
}


// A decimal's fraction in words - digit by digit after the point
function fractionWords(fraction) {
    return ['point', ...digitWords(fraction)];
}


// A number token's words - its digits (commas grouping its thousands) in words (integerWords), a four-digit number as a
// year, a decimal's fraction digit by digit, a percentage, and an ordinal (1st, 22nd) or a plural (1970s, 70's) as its
// last word's
export function numberTokenWords(token) {
    const [, digits, fraction, suffix] = token.match(/^([\d,]+)(?:\.(\d+))?(\D*)$/);
    const isYear = /^[1-9]\d{3}$/.test(digits) && fraction === undefined && (suffix === '' || suffix.endsWith('s'));
    const words = isYear ? yearWords(Number(digits)) : integerWords(digits);
    if (fraction !== undefined) {
        words.push(...fractionWords(fraction));
    }
    if (suffix === '%') {
        words.push('percent');
    } else if (suffix.endsWith('s')) {
        words.push(pluralWord(words.pop()));
    } else if (suffix !== '') {
        words.push(ordinalWord(words.pop()));
    }
    return words;
}


// The currencies' words - the unit, its plural, the hundredth, its plural
const currencies = {
    '$': ['dollar', 'dollars', 'cent', 'cents'], '£': ['pound', 'pounds', 'penny', 'pence'], '€': ['euro', 'euros', 'cent', 'cents']
};


// An amount of money's words - $1,234.56 as one thousand two hundred thirty-four dollars and fifty-six cents, $0.50 as
// fifty cents, $1.5 as one point five dollars, and $5 million as five million dollars
export function moneyWords(token) {
    const [, symbol, digits, fraction, scale] = token.match(/^(\D)([\d,]+)(?:\.(\d+))?(?:\s+([a-z]+))?$/);
    const [unit, units, hundredth, hundredths] = currencies[symbol];
    const value = Number(digits.replace(/,/g, ''));
    if (fraction?.length === 2 && scale === undefined) {
        const cents = Number(fraction);
        const words = value === 0 && cents !== 0 ? [] : [...integerWords(digits), value === 1 ? unit : units];
        if (cents !== 0) {
            words.push(...(words.length === 0 ? [] : ['and']), ...numberWords(cents), cents === 1 ? hundredth : hundredths);
        }
        return words;
    }
    const words = integerWords(digits);
    if (fraction !== undefined) {
        words.push(...fractionWords(fraction));
    }
    if (scale !== undefined) {
        words.push(scale);
    }
    words.push(value === 1 && fraction === undefined && scale === undefined ? unit : units);
    return words;
}


// The letters of a.m. and p.m. as words
const meridiemWords = {'a': ['ay', 'em'], 'p': ['pee', 'em']};


// A time's words - 10:30 as ten thirty, 10:05 as ten oh five, 10:00 as ten o'clock (14:00 as fourteen hundred), and
// with a.m. or p.m. (10:00 p.m. and 10pm as ten pee em)
export function timeWords(token) {
    const [, hours, minutes, meridiem] = token.match(/^(\d+)(?::(\d+))?[ \t]*(?:([ap])\.?m\.?)?$/);
    const words = numberWords(Number(hours));
    if (minutes !== undefined && Number(minutes) !== 0) {
        words.push(...(Number(minutes) < 10 ? ['oh', ones[Number(minutes)]] : numberWords(Number(minutes))));
    } else if (meridiem === undefined) {
        words.push(Number(hours) >= 1 && Number(hours) <= 12 ? 'o\'clock' : 'hundred');
    }
    return meridiem === undefined ? words : [...words, ...meridiemWords[meridiem]];
}


// The context marks of the phrase ends - a phrase end's item in a text's items (the G2P's context network reads them)
const phraseMarks = {'comma': ',', 'period': '.', 'question': '?', 'exclamation': '!'};


// The text's tokens - a.m. or p.m., a dotted abbreviation (letters with their periods - "i.e.", "U.S.A." - and "Ph.D."), a
// unit with a slash or a power (km/h, m2), degrees (°, °C, °F), a word, a time (10:30, 10:30 a.m., 10pm), an amount of
// money (with its scale - $5 million), a number (its thousands grouped by commas or not; with a decimal fraction, a
// percent sign, or an ordinal or plural suffix), a symbol read as a word (&, +, =, ×, ±, ≈; <, >, and ~ before a number;
// a minus sign before a number - a hyphen only after a space or an opening parenthesis), or a punctuation mark (a
// parenthesis too)
// (matched without case - a word's capitals tell an initial and a word in capitals)
const digitsPattern = String.raw`(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)`;
const meridiemPattern = String.raw`[ap]\.?m\.?(?![a-z])`;
const tokenPattern = new RegExp([
    String.raw`[ap]\.m\.?(?![a-z])`,
    String.raw`(?:(?:[a-z]\.)+[a-z]|ph\.d)\.?(?![a-z])`,
    String.raw`(?:km|mi|cm|mm|ft|m)(?:\/(?:hr|h|s|min)|[23²³])(?![a-z0-9])`,
    String.raw`°(?:[ \t]?[cf](?![a-z]))?`,
    String.raw`[a-z][a-z']*\.?`,
    String.raw`\d{1,2}(?::\d{2}(?!\d)(?:[ \t]*${meridiemPattern})?|[ \t]*${meridiemPattern})`,
    String.raw`[$£€]${digitsPattern}(?:\.\d+)?(?:[ \t\r\n]+(?:thousand|million|billion|trillion)(?![a-z]))?`,
    String.raw`${digitsPattern}(?:\.\d+%?|%|(?:st|nd|rd|th|'?s)(?![a-z]))?`,
    String.raw`[&+=×±≈]|[<>~](?=[ \t]?\d)|−(?=\d)|(?<![^ \t\r\n(])-(?=\d)`,
    '[.,;:!?()]'
].join('|'), 'gi');


// Whether the token "I." at index is the pronoun, not an initial - after a lowercase word and not before an initial
// ("said I.", not "met I. M. Pei" - as lib/sentences.js ends a sentence)
function isPronoun(casedTokens, index) {
    return /^[a-z]/.test(casedTokens[index - 1] ?? '') && !/^[A-Z]\./.test(casedTokens[index + 1] ?? '');
}


// The words of a unit with a slash or a power, as written - "km/h" as kilometers per hour, "m2" as square meters (the
// plural, but after the number 1); null if it isn't written as a unit is (its letters in capitals - "M2")
function compoundUnitWords(cased, singular) {
    const [, length, time, power] = cased.match(/^([a-z]+)(?:\/([a-z]+)|(.))$/) ?? [];
    if (length === undefined) {
        return null;
    }
    const lengthWord = singular ? unitLengths[length] : unitPlural(unitLengths[length]);
    return time === undefined ? [unitPowers[power], lengthWord] : [lengthWord, 'per', unitTimes[time]];
}


// A spelled word's phones - its letters' names, the last with the word's -s ("CEOs", "IBM's")
function spelledPhones(pronounce, stem, suffix) {
    const names = [...stem].map((letter) => letterWords[letter] ?? letter);
    if (suffix) {
        names[names.length - 1] += '\'s';
    }
    return names.flatMap((name) => pronounce(name));
}


// Tokenize text into phrases - [{words: [{text, phones: [{ph, stress}]}], end}]
// The phrase end is 'comma' (a comma, semicolon, or colon), 'period', 'exclamation', 'question' (yes/no), or 'wh' (a
// question-word question). A parenthesis - opening or closing an aside - is a comma's break, unless a punctuation mark
// right after it ends the phrase ("(1818)." ends with a period). A word the pronouncer finds ambiguous (the G2P network's ambiguity head -
// pronounce.ambiguous) takes the reading its context network chooses from the text's items (pronounce.chooseReading).
// An initial - a capital letter with its period ("Dr. A. B. Jones"; not the pronoun - isPronoun) - is its letter's name
// and doesn't end a phrase, nor does a dotted abbreviation ("the U.S. economy", "i.e."); and a word in capitals the
// pronouncer finds spelled (the G2P network's spelled head - pronounce.spelled: "AI", "TOC" - not "NOT" or "NASA") is its
// letters' names - with its -s, the word's ("CEOs", "IBM's": the head is asked of the capitals) - asked as a word alone
// among lowercase words ("the US government"), or as one beside words in capitals that are read ("LEAVE US ALONE" - its
// neighbors in the phrase, both if it has two: not "the US NATO allies"). A unit after a number is
// its words ("5 km" as five kilometers, "93,000 sq mi", "20 °C" as twenty degrees celsius; "km/h" and "km2" anywhere).
export function textToPhrases(text) {
    const pronounce = getPronouncer();
    const casedTokens = plainLetters(text).replace(/[‘’]/g, '\'').match(tokenPattern) || [];
    const tokens = casedTokens.map((token) => token.toLowerCase());

    // The text's items - its words, and its phrase ends ({end}, only after a word - a parenthesis's break gives way to a
    // punctuation mark right after it) - and the items that are words in capitals (their capitals - the word without
    // its -s; '' for a capital letter alone - "A", "I")
    const items = [];
    const capitals = new Map();
    let parenthesisEnd = false;
    const endPhrase = (end, parenthesis = false) => {
        if (items.length !== 0 && typeof items[items.length - 1] === 'string') {
            items.push({end});
            parenthesisEnd = parenthesis;
        } else if (parenthesisEnd && !parenthesis) {
            items[items.length - 1] = {end};
            parenthesisEnd = false;
        }
    };
    const addWord = (word) => {
        items.push(word.replace(/^'+|'+$/g, ''));
    };

    // Whether the token at index is a number - a unit's ("5 km"; with "sq" or "cu" between, "5 sq mi") - and whether it is 1
    const isNumber = (index) => /^[\d,]+(?:\.\d+)?$/.test(tokens[index] ?? '');
    const afterNumber = (index) => isNumber(index - 1) ||
        (Object.hasOwn(unitPrefixes, casedTokens[index - 1] ?? '') && isNumber(index - 2));
    const isOne = (index) => tokens[index - 1] === '1';
    for (const [index, token] of tokens.entries()) {
        if (/^\d+(?::|[ \t]*[ap])/.test(token)) {
            timeWords(token).forEach(addWord);
        } else if (/^\d/.test(token)) {
            numberTokenWords(token).forEach(addWord);
        } else if (/^[$£€]/.test(token)) {
            moneyWords(token).forEach(addWord);
        } else if (/^[ap]\.m/.test(token)) {
            meridiemWords[token[0]].forEach(addWord);
        } else if (/^(?:(?:[a-z]\.)+[a-z]|ph\.d)\.?$/.test(token)) {
            const letters = token.replace(/\./g, '');
            (dottedWords[letters] ?? [...letters].map((letter) => letterWords[letter] ?? letter)).forEach(addWord);
        } else if (token.startsWith('°')) {
            addWord(isOne(index) ? 'degree' : 'degrees');
            if (token.length > 1) {
                addWord(token.endsWith('c') ? 'celsius' : 'fahrenheit');
            }
        } else if (Object.hasOwn(symbolWords, token)) {
            symbolWords[token].forEach(addWord);
        } else if (/^[a-z]+[/23²³]/.test(token)) {
            const words = compoundUnitWords(casedTokens[index], isOne(index));
            if (words !== null) {
                words.forEach(addWord);
            } else {
                // (not a unit - "M2": its letters as a word, and its number)
                const [, letters, digit] = token.match(/^([a-z]+)\/?(.*)$/);
                addWord(letters);
                const rest = /^\d$/.test(digit) ? numberWords(Number(digit)) : [digit];
                rest.filter((word) => /^[a-z]/.test(word)).forEach(addWord);
            }
        } else if (/^[a-z]/.test(token)) {
            const bare = token.replace(/\.$/, '');
            const casedBare = casedTokens[index].replace(/\.$/, '');
            if (Object.hasOwn(unitWords, casedBare) && afterNumber(index)) {
                unitWords[casedBare][isOne(index) ? 0 : 1].split(' ').forEach(addWord);
                if (token.endsWith('.')) {
                    endPhrase('period');
                }
            } else if (Object.hasOwn(unitPrefixes, casedBare) && isNumber(index - 1) &&
                Object.hasOwn(unitWords, (casedTokens[index + 1] ?? '').replace(/\.$/, ''))) {
                addWord(unitPrefixes[casedBare]);
            } else if (Object.hasOwn(abbreviations, bare)) {
                addWord(abbreviations[bare]);
            } else if (token.endsWith('.') && Object.hasOwn(periodAbbreviations, bare)) {
                addWord(periodAbbreviations[bare]);
            } else if (token === 'no.' && /^\d/.test(tokens[index + 1] ?? '')) {
                addWord('number');
            } else if (/^[A-Z]\.$/.test(casedTokens[index]) && !(bare === 'i' && isPronoun(casedTokens, index))) {
                addWord(letterWords[bare] ?? bare);
            } else {
                const capital = casedTokens[index].match(/^([A-Z]{2,})(?:'?s)?\.?$/);
                if (capital !== null) {
                    capitals.set(items.length, capital[1].toLowerCase());
                } else if (/^[A-Z]\.?$/.test(casedTokens[index])) {
                    capitals.set(items.length, '');
                }
                addWord(bare);
                if (token.endsWith('.')) {
                    endPhrase('period');
                }
            }
        } else if (token === '?') {
            endPhrase('question');
        } else if (token === '!') {
            endPhrase('exclamation');
        } else if (token === '.') {
            endPhrase('period');
        } else if (token === '(' || token === ')') {
            endPhrase('comma', true);
        } else {
            endPhrase('comma');
        }
    }
    endPhrase('period');

    // The words in capitals beside words in capitals that are read - a word's neighbors in its phrase (one, or both; a
    // capital letter alone - "A", "I" - is neither): "LEAVE US ALONE", "NOT US", and "TELL US A STORY", not "the US FDA",
    // "the US NATO allies", or "A US senator"
    const isRead = (index) => capitals.has(index) && pronounce.spelled?.(capitals.get(index), true) !== true;
    const isWord = (index) => typeof items[index] === 'string' && capitals.get(index) !== '';
    const besideCapitals = new Set([...capitals.keys()].filter((index) => {
        const neighbors = [index - 1, index + 1].filter(isWord);
        return capitals.get(index) !== '' && neighbors.length !== 0 && neighbors.every(isRead);
    }));

    // The phrases - each word pronounced (in the reading its context chooses, if ambiguous; a spelled word in capitals as
    // its letters' names)
    const context = items.map((item) => (typeof item === 'string' ? item : phraseMarks[item.end]));
    const phrases = [];
    let words = [];
    for (const [index, item] of items.entries()) {
        if (typeof item !== 'string') {
            const isQuestion = item.end === 'question';
            phrases.push({words, 'end': isQuestion && whWords.has(words[0].text) ? 'wh' : item.end});
            words = [];
            continue;
        }
        const stem = capitals.get(index);
        const spelled = Boolean(stem) && pronounce.spelled?.(stem, besideCapitals.has(index)) === true;
        const reading = !spelled && pronounce.ambiguous?.(item) ? pronounce.chooseReading(context, index) : 0;
        const wordPhones = spelled ? spelledPhones(pronounce, stem, item !== stem) : pronounce(item, reading);
        const phones = wordPhones.map((phone) => {
            const match = phone.match(/^([A-Z]+)(\d)?$/);
            return {'ph': match[1], 'stress': match[2] === undefined ? null : Number(match[2])};
        });
        if (functionWords.has(item) && !spelled) {
            for (const phone of phones) {
                if (phone.stress !== null) {
                    phone.stress = 0;
                }
            }
        }
        words.push({'text': item, phones});
    }
    return phrases;
}
