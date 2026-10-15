// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

import {decodeText, displayTitle, prepareText} from '../lib/gutenberg.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


// Check a prepared text's chapters - each at its heading paragraph
function checkChapters(result) {
    for (const chapter of result.chapters) {
        assert.equal(result.text.slice(chapter.char, chapter.char + chapter.title.length), chapter.title);
        assert.ok(chapter.char === 0 || result.text.slice(chapter.char - 2, chapter.char) === '\n\n');
    }
}


// A paragraph of prose, wrapped at about 70 characters
function prose(sentence, count = 3) {
    const words = Array(count).fill(sentence).join(' ').split(' ');
    const lines = [];
    let line = '';
    for (const word of words) {
        if (line !== '' && line.length + 1 + word.length > 70) {
            lines.push(line);
            line = word;
        } else {
            line = line === '' ? word : `${line} ${word}`;
        }
    }
    lines.push(line);
    return lines.join('\n');
}


// A sentence repeated three times, as prose() paragraphs are prepared
function repeated(sentence) {
    return [sentence, sentence, sentence].join(' ');
}


// Filler paragraphs, wrapped and as prepared
const fillerSentences = [...Array(8).keys()].map((ix) => `Paragraph ${ix + 1} of the story goes on and on about the day.`);
const fillerRaw = fillerSentences.map((sentence) => prose(sentence)).join('\n\n');
const fillerText = fillerSentences.map((sentence) => repeated(sentence)).join('\n\n');

// Short paragraphs, the same raw and prepared
const shortParagraphsText = [...Array(120).keys()].map((ix) => `Paragraph ${ix + 1}.`).join('\n\n');

// A novel - a byte order mark, CRLF line ends, a header with a continued title, a header marker continued onto the next
// line, a credit line, a title page, an illustration, contents and illustration lists, footnotes, and a footer marker
const novelText = `\uFEFFThe Project Gutenberg EBook of A Test Novel, by Jane Doe

This eBook is for the use of anyone anywhere at no cost.

Title: A Test Novel;
       or, The Long Day

Author: Jane Doe (Jane Q. Doe)

Release Date: 2001 [EBook #1]

*** START OF THIS PROJECT GUTENBERG EBOOK A TEST
NOVEL ***

Produced by Some Volunteer and the Online Distributed Proofreading Team

* * * * *

A TEST NOVEL

by Jane Doe

FIRST EDITION, PRINTED IN LONDON

[Illustration: The frontispiece
showing a house [with a garden]]

CONTENTS

CHAPTER I. The Morning ..... 1

CHAPTER II. The Afternoon ..... 9

CHAPTER III .... 20

Page

LIST OF ILLUSTRATIONS

The house ..... 3

The garden ..... 12

CHAPTER I.
THE MORNING

${prose('It was _early_ in the =morning= and the sun[1] rose -- slowly -- over the hill.')}

[Footnote 1: A footnote
over two lines.]

${fillerRaw}

CHAPTER II.

The Afternoon

${prose('The afternoon was warm.')}

[Illustration: A garden] And text after the illustration.

CHAPTER III

${prose('Night fell over the house and the garden.')}

${fillerRaw}

THE END

Advertisements for other books.

*** END OF THIS PROJECT GUTENBERG EBOOK A TEST NOVEL ***

This file should be named 1.txt. Project Gutenberg license and copyright notes.
`.replace(/\n/g, '\r\n');

// A play - no header marker, a content note, a contents list of acts and scenes, Dramatis Personae, speeches with line
// numbers, act and scene markers, and a footer marker
const playText = `THE TRAGEDY OF HAMLET, PRINCE OF DENMARK

by William Shakespeare

* CONTENT NOTE (added in 2017) *

This text has a content note.

* END CONTENT NOTE *

Contents

ACT I
Scene I. Elsinore. A platform before the Castle.
Scene II. Elsinore. A room of state in the Castle.

ACT II
Scene I. A room in Polonius’s house.

Dramatis Personæ

Dramatis Personæ

HAMLET, Prince of Denmark.
CLAUDIUS, King of Denmark.

ACT I

SCENE I. Elsinore. A platform before the Castle.

Enter Francisco and Barnardo, two Sentinels.

BARNARDO.
Who’s there?

FRANCISCO.
Nay, answer me. Stand and unfold yourself.

ACT_1|SC_2

SCENE II. Elsinore. A room of state in the Castle.

Enter the King.

KING.
Though yet of Hamlet our dear brother’s death     2
The memory be green, and that it us befitted     3
To bear our hearts in grief and our whole kingdom     4

ACT II

SCENE I. A room in Polonius’s house.

Enter Polonius.

POLONIUS.
Give him this money and these notes, Reynaldo.

*** END OF THE PROJECT GUTENBERG EBOOK HAMLET ***

The license.
`;

// A play with nested act and scene headings
const playNestedText = `ALL'S WELL THAT ENDS WELL

ACT I. SCENE 1.
Rousillon. The COUNT'S palace

Enter BERTRAM, the COUNTESS OF ROUSILLON, HELENA, and LAFEU, all in black

COUNTESS. In delivering my son from me, I bury a second husband.
BERTRAM. And I in going, madam, weep o'er my father's death anew;
  but I must attend his Majesty's command.

ACT I. SCENE 2.
Paris. The KING'S palace

Flourish of cornets. Enter the KING OF FRANCE, with letters, and divers attendants

ACT II. SCENE 1.

Paris. The KING'S palace

Enter the KING.

SCENE 2. ACT 3.

The play goes on.
`;

// Poems - a contents list of section titles, numbered sonnets under a plural section title, titled poems, and long
// lines with indented continuations
const poemsText = `CONTENTS

THE SONNETS
Song of the Open Road
A LOVER'S COMPLAINT

THE SONNETS

I.

From fairest creatures we desire increase,
That thereby beauty’s rose might never die,
But as the riper should by time decease,
His tender heir might bear his memory:

II.

When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty’s field,
Thy youth’s proud livery so gazed on now,
Will be a tatter’d weed of small worth held:

IV.

Unthrifty loveliness, why dost thou spend
Upon thyself thy beauty’s legacy?

Song of the Open Road

Afoot and light-hearted I take to the open road, healthy, free, the world
    before me, the long brown path before me leading wherever I choose.
Henceforth I ask not good-fortune, I myself am good-fortune,
Henceforth I whimper no more, postpone no more, need nothing,
Done with indoor complaints, libraries, querulous criticisms,
Strong and content I travel the open road.

A LOVER'S COMPLAINT

From off a hill whose concave womb reworded
A plaintful story from a sistering vale,
My spirits to attend this double voice accorded,
And down I laid to list the sad-tuned tale;

To the Reader

This figure that thou here seest put,
It was for gentle Shakespeare cut;
Wherein the graver had a strife
With Nature, to out-do the life.

The End.
`;

// Headings of many kinds, footnotes, and transcriber's notes
const headingsText = `The Project Gutenberg eBook, Various Headings, by A. Writer

Title: Various Headings
    A Collection of Essays

Author: A. Writer

*** START OF THE PROJECT GUTENBERG EBOOK VARIOUS HEADINGS ***

Transcriber's Note:

Obvious typos have been corrected.

Italics are shown with underscores.


THE AUTHOR'S PREFACE

${prose('The preface tells of the writing of the book.')}

[1] The first footnote.

Footnote 2: The second footnote.

FOOTNOTES:

This Project Gutenberg eBook is a reprint of the first edition.

BOOK THE FIRST—THE BEGINNING

CHAPTER 12

${prose('The twelfth chapter comes first.')}

CHAPTER IIII.

CHAPTER THE LAST

${prose('The last chapter is next.')}

Part One of the Story begins here

CHAPTER IV The Title Without Separator

1. Take two eggs.

I. Alpha, II. Beta

L

APPENDIX: A Very Long Appendix Title That Goes On And On For More Than Sixty Characters

FIRST BOOK

The Second Part—The Middle

CHAPTER I. JONATHAN HARKER'S JOURNAL—continued

${prose('The journal goes on.')}

CHAPTER II. THE MAN OF THE HOUR AND O'BRIEN AND D'ARCY THE ENGINEER'S MATE

${prose('The hour came.')}

CHAPTER V. WHO IS SHE?

${prose('Nobody knew.')}

I.

${prose('The first part of the fifth chapter.')}

II.

${prose('The second part of the fifth chapter.')}

12

A single line.

13

${prose('Thirteen.')}

EPILOGUE
EPILOGUE.

${prose('After it all.')}

Notes by the transcriber:

Some notes at the end.

More notes.
`;

// A text with no title, author, or markers - a heading first, and a contents list without a contents heading
const bareText = `CHAPTER I

${prose('The first chapter.')}

CHAPTER II.

CHAPTER III.

CHAPTER IV.

PREFACE

CHAPTER I.

APPENDIX

${prose('A paragraph.')}
`;

// A contents list of ordinal-only entries with titles, a section title beside numbered entries, a skipped line, and a
// header title continued after a comma
const contentsText = `Title: Poems,
       Second Series

Author: Emily Dickinson

*** START OF THIS PROJECT GUTENBERG EBOOK POEMS ***

CONTENTS

ETYMOLOGY

I. Down the Rabbit-Hole

an unconfirmed line

IV

II. The Pool of Tears

II. The Pool of Tears Again

CHAPTER THREE. The Twenty-first Day

CHAPTER THE FINAL

ACT I Scene I. Elsinore. A platform. Scene II. A room of state in the Castle.

Etymology

ETYMOLOGY

${prose('Etymology is the study of words.')}

I.

${prose('Alice was tired.')}

II.

${prose('The pool of tears.')}

CHAPTER ONE.
the first day

${prose('One day.')}

CHAPTER TWENTY-FIRST.
The Man Of

${prose('Another day.')}

CHAPTER XXII.
A very long title that goes on and on and on and on and on and on and on and on and on and on

${prose('More days.')}

CHAPTER XXIII. A Title
And More

${prose('Even more days.')}

PREFACE

CHAPTER I.

APPENDIX

${prose('Text between.')}

I.

II.

III.

${prose('Text after.')}

CHAPTER XXIV. L'AMOUR, THE END

${prose('The end.')}
`;

// Transcriber's notes at the end of a long text
const notesText = `${shortParagraphsText}

Transcriber's Notes

The first note.

The second note.
`;

// Verse - lines with breaks before words that would have fit, all-capitalized long lines, and indented continuations
const verseText = `I

Afoot and light-hearted I take to the open road, healthy, free, the world
    before me, the long brown path before me leading wherever I choose.
Henceforth I ask not good-fortune, I myself am good-fortune, now and then,
Henceforth I whimper no more, postpone no more, need nothing at all now,

II

Short line,
Another short line,
A third short line.

III

Done with indoor complaints, libraries, querulous criticisms, and all the rest
Strong and content I travel the open road, and travel it on, and on, and on
Still here I carry my old delicious burdens, I carry them, men and women,

IV

Prose continues here on a line that is quite long and goes all the way to the
end, and then it goes on to the next line, which starts with a lower case word,
and so on.

A Poem Title

The first line of the poem,
The second line of the poem.
`;

// A verse paragraph with an indented continuation of a long line
const whitmanText = `Out of the cradle endlessly rocking, and more so,
    out of the mocking-bird's throat,
Out of the Ninth-month midnight, it was so,
Over the sterile sands and the fields beyond,
`;


test('displayTitle', () => {
    assert.equal(displayTitle('Moby Dick; Or, The Whale'), 'Moby Dick; Or, The Whale');
    assert.equal(displayTitle('Frankenstein; Or, The Modern Prometheus'), 'Frankenstein; Or, The Modern Prometheus');
    assert.equal(displayTitle('Dracula'), 'Dracula');

    // Line breaks - after punctuation a space, otherwise a colon
    assert.equal(displayTitle('The Odyssey\r\nRendered into English prose'), 'The Odyssey: Rendered into English prose');
    assert.equal(displayTitle('Walden,\nand On The Duty Of Civil Disobedience'),
        'Walden, and On The Duty Of Civil Disobedience');

    // A generic subtitle
    assert.equal(displayTitle('Wuthering Heights: A Novel.'), 'Wuthering Heights');
    assert.equal(displayTitle('The Scarlet Letter; a romance'), 'The Scarlet Letter');

    // A long title is cut at its subtitle
    assert.equal(displayTitle('The Strange Case of Dr. Jekyll: And Mr. Hyde, and Other Stories of the Night'),
        'The Strange Case of Dr. Jekyll');
    assert.equal(displayTitle('A Very Long Title Without Any Subtitle Separator At All'),
        'A Very Long Title Without Any Subtitle Separator At All');
    assert.equal(displayTitle('Folk tales from Tibet — with illustrations by a Tibetan artist'), 'Folk tales from Tibet');
});


test('decodeText', () => {
    const utf8 = new TextEncoder().encode('Café — “quoted”');
    const latin1 = new Uint8Array([0x43, 0x61, 0x66, 0xe9, 0x20, 0x93, 0x71, 0x94]);
    assert.equal(decodeText(utf8), 'Café — “quoted”');
    assert.equal(decodeText(utf8, 'UTF-8'), 'Café — “quoted”');

    // Not UTF-8 - decoded as Windows-1252
    assert.equal(decodeText(latin1), 'Café “q”');
    assert.equal(decodeText(latin1, 'utf-8'), 'Café “q”');

    // Windows-1252
    assert.equal(decodeText(latin1, 'windows-1252'), 'Café “q”');
});


test('prepareText, novel', () => {
    const result = prepareText(novelText);
    assert.equal(result.title, 'A Test Novel, or, The Long Day');
    assert.equal(result.author, 'Jane Doe');
    assert.equal(result.text, `\
A Test Novel, or, The Long Day.

By Jane Doe.

Chapter I. The Morning.

${repeated('It was early in the morning and the sun rose — slowly — over the hill.')}

${fillerText}

Chapter II. The Afternoon.

${repeated('The afternoon was warm.')}

And text after the illustration.

Chapter III.

${repeated('Night fell over the house and the garden.')}

${repeated('Paragraph 1 of the story goes on and on about the day.')}

${repeated('Paragraph 2 of the story goes on and on about the day.')}

${repeated('Paragraph 3 of the story goes on and on about the day.')}

${repeated('Paragraph 4 of the story goes on and on about the day.')}

${repeated('Paragraph 5 of the story goes on and on about the day.')}

${repeated('Paragraph 6 of the story goes on and on about the day.')}

${repeated('Paragraph 7 of the story goes on and on about the day.')}

${repeated('Paragraph 8 of the story goes on and on about the day.')}

THE END`);
    assert.deepEqual(result.chapters, [
        {'title': 'Chapter I. The Morning', 'char': 47, 'level': 1},
        {'title': 'Chapter II. The Afternoon', 'char': 1614, 'level': 1},
        {'title': 'Chapter III', 'char': 1749, 'level': 1}
    ]);
    checkChapters(result);
});


test('prepareText, play', () => {
    const result = prepareText(playText, {'title': 'Book Title', 'author': 'Book Author'});
    assert.equal(result.title, 'Book Title');
    assert.equal(result.author, 'Book Author');
    assert.equal(result.text, `\
Book Title.

By Book Author.

THE TRAGEDY OF HAMLET, PRINCE OF DENMARK

by William Shakespeare

Dramatis Personæ

HAMLET, Prince of Denmark.
CLAUDIUS, King of Denmark.

Act I.

Scene I. Elsinore. A platform before the Castle.

Enter Francisco and Barnardo, two Sentinels.

BARNARDO.
Who’s there?

FRANCISCO.
Nay, answer me. Stand and unfold yourself.

Scene II. Elsinore. A room of state in the Castle.

Enter the King.

KING.
Though yet of Hamlet our dear brother’s death
The memory be green, and that it us befitted
To bear our hearts in grief and our whole kingdom

Act II.

Scene I. A room in Polonius’s house.

Enter Polonius.

POLONIUS.
Give him this money and these notes, Reynaldo.`);
    assert.deepEqual(result.chapters, [
        {'title': 'Act I', 'char': 169, 'level': 1},
        {'title': 'Scene I. Elsinore. A platform before the Castle', 'char': 177, 'level': 2},
        {'title': 'Scene II. Elsinore. A room of state in the Castle', 'char': 352, 'level': 2},
        {'title': 'Act II', 'char': 569, 'level': 1},
        {'title': 'Scene I. A room in Polonius’s house', 'char': 578, 'level': 2}
    ]);
    checkChapters(result);
});


test('prepareText, play with nested headings', () => {
    const result = prepareText(playNestedText, {'title': 'Book Title', 'author': 'Book Author'});
    assert.equal(result.title, 'Book Title');
    assert.equal(result.author, 'Book Author');
    assert.equal(result.text, `\
Book Title.

By Book Author.

ALL'S WELL THAT ENDS WELL

Act I. Scene 1. Rousillon. The COUNT'S palace.

Enter BERTRAM, the COUNTESS OF ROUSILLON, HELENA, and LAFEU, all in black

COUNTESS. In delivering my son from me, I bury a second husband. BERTRAM. And I in going, madam, weep o'er my father's death \
anew; but I must attend his Majesty's command.

Act I. Scene 2. Paris. The KING'S palace.

Flourish of cornets. Enter the KING OF FRANCE, with letters, and divers attendants

Act II. Scene 1.

Paris. The KING'S palace

Enter the KING.

Scene 2. Act 3.

The play goes on.`);
    assert.deepEqual(result.chapters, [
        {'title': 'Act I. Scene 1. Rousillon. The COUNT\'S palace', 'char': 57, 'level': 1},
        {'title': 'Act I. Scene 2. Paris. The KING\'S palace', 'char': 353, 'level': 2},
        {'title': 'Act II. Scene 1', 'char': 480, 'level': 1},
        {'title': 'Scene 2. Act 3', 'char': 541, 'level': 2}
    ]);
    checkChapters(result);
});


test('prepareText, poems', () => {
    const result = prepareText(poemsText, {'title': 'Book Title', 'author': 'Book Author'});
    assert.equal(result.title, 'Book Title');
    assert.equal(result.author, 'Book Author');
    assert.equal(result.text, `\
Book Title.

By Book Author.

The Sonnets.

Sonnet I.

From fairest creatures we desire increase,
That thereby beauty’s rose might never die,
But as the riper should by time decease,
His tender heir might bear his memory:

Sonnet II.

When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty’s field,
Thy youth’s proud livery so gazed on now,
Will be a tatter’d weed of small worth held:

Poem IV.

Unthrifty loveliness, why dost thou spend
Upon thyself thy beauty’s legacy?

Song of the Open Road.

Afoot and light-hearted I take to the open road, healthy, free, the world before me, the long brown path before me leading \
wherever I choose. Henceforth I ask not good-fortune, I myself am good-fortune, Henceforth I whimper no more, postpone no \
more, need nothing, Done with indoor complaints, libraries, querulous criticisms, Strong and content I travel the open road.

A Lover's Complaint.

From off a hill whose concave womb reworded
A plaintful story from a sistering vale,
My spirits to attend this double voice accorded,
And down I laid to list the sad-tuned tale;

To the Reader.

This figure that thou here seest put,
It was for gentle Shakespeare cut;
Wherein the graver had a strife
With Nature, to out-do the life.

The End.`);
    assert.deepEqual(result.chapters, [
        {'title': 'The Sonnets', 'char': 30, 'level': 1},
        {'title': 'Sonnet I', 'char': 44, 'level': 2},
        {'title': 'Sonnet II', 'char': 223, 'level': 2},
        {'title': 'Song of the Open Road', 'char': 498, 'level': 2},
        {'title': 'A Lover\'s Complaint', 'char': 893, 'level': 1},
        {'title': 'To the Reader', 'char': 1094, 'level': 2}
    ]);
    checkChapters(result);
});


test('prepareText, verse', () => {
    const result = prepareText(verseText);
    assert.equal(result.title, '');
    assert.equal(result.author, '');
    assert.equal(result.text, `\
Chapter I.

Afoot and light-hearted I take to the open road, healthy, free, the world before me, the long brown path before me leading \
wherever I choose. Henceforth I ask not good-fortune, I myself am good-fortune, now and then, Henceforth I whimper no more, \
postpone no more, need nothing at all now,

Chapter II.

Short line,
Another short line,
A third short line.

Chapter III.

Done with indoor complaints, libraries, querulous criticisms, and all the rest Strong and content I travel the open road, \
and travel it on, and on, and on Still here I carry my old delicious burdens, I carry them, men and women,

Chapter IV.

Prose continues here on a line that is quite long and goes all the way to the end, and then it goes on to the next line, \
which starts with a lower case word, and so on.

A Poem Title

The first line of the poem,
The second line of the poem.`);
    assert.deepEqual(result.chapters, [
        {'title': 'Chapter I', 'char': 0, 'level': 1},
        {'title': 'Chapter II', 'char': 303, 'level': 1},
        {'title': 'Chapter III', 'char': 369, 'level': 1},
        {'title': 'Chapter IV', 'char': 613, 'level': 1}
    ]);
    checkChapters(result);
});


test('prepareText, verse continuation lines', () => {
    const result = prepareText(whitmanText, {'title': 'Book Title', 'author': 'Book Author'});
    assert.equal(result.title, 'Book Title');
    assert.equal(result.author, 'Book Author');
    assert.equal(result.text, `\
Book Title.

By Book Author.

Out of the cradle endlessly rocking, and more so, out of the mocking-bird's throat,
Out of the Ninth-month midnight, it was so,
Over the sterile sands and the fields beyond,`);
    assert.deepEqual(result.chapters, []);
    checkChapters(result);
});


test('prepareText, headings', () => {
    const result = prepareText(headingsText);
    assert.equal(result.title, 'Various Headings: A Collection of Essays');
    assert.equal(result.author, 'A. Writer');
    assert.equal(result.text, `\
Various Headings: A Collection of Essays.

By A. Writer.

The Author's Preface.

${repeated('The preface tells of the writing of the book.')}

The first footnote.

Book The First. The Beginning.

Chapter 12.

${repeated('The twelfth chapter comes first.')}

CHAPTER IIII.

Chapter The Last.

${repeated('The last chapter is next.')}

Part One of the Story begins here

Chapter IV. The Title Without Separator.

1. Take two eggs.

I. Alpha, II. Beta

L

APPENDIX: A Very Long Appendix Title That Goes On And On For More Than Sixty Characters

First Book.

The Second Part. The Middle.

Chapter I. Jonathan Harker's Journal—continued.

${repeated('The journal goes on.')}

Chapter II. The Man of the Hour and O'Brien and D'Arcy the Engineer's Mate.

${repeated('The hour came.')}

Chapter V. Who Is She?

${repeated('Nobody knew.')}

Part I.

${repeated('The first part of the fifth chapter.')}

Part II.

${repeated('The second part of the fifth chapter.')}

12

A single line.

13

${repeated('Thirteen.')}

Epilogue.

${repeated('After it all.')}`);
    assert.deepEqual(result.chapters, [
        {'title': 'The Author\'s Preface', 'char': 58, 'level': 1},
        {'title': 'Book The First. The Beginning', 'char': 241, 'level': 1},
        {'title': 'Chapter 12', 'char': 273, 'level': 2},
        {'title': 'Chapter The Last', 'char': 401, 'level': 2},
        {'title': 'Chapter IV. The Title Without Separator', 'char': 534, 'level': 2},
        {'title': 'First Book', 'char': 707, 'level': 1},
        {'title': 'The Second Part. The Middle', 'char': 720, 'level': 1},
        {'title': 'Chapter I. Jonathan Harker\'s Journal—continued', 'char': 750, 'level': 2},
        {'title': 'Chapter II. The Man of the Hour and O\'Brien and D\'Arcy the Engineer\'s Mate', 'char': 863, 'level': 2},
        {'title': 'Chapter V. Who Is She?', 'char': 986, 'level': 2},
        {'title': 'Epilogue', 'char': 1351, 'level': 2}
    ]);
    checkChapters(result);
});


test('prepareText, contents', () => {
    const result = prepareText(contentsText);
    assert.equal(result.title, 'Poems, Second Series');
    assert.equal(result.author, 'Emily Dickinson');
    assert.equal(result.text, `\
Poems, Second Series.

By Emily Dickinson.

Etymology.

Etymology.

${repeated('Etymology is the study of words.')}

Part I. Down the Rabbit-Hole.

${repeated('Alice was tired.')}

Part II.

${repeated('The pool of tears.')}

CHAPTER ONE. the first day

${repeated('One day.')}

CHAPTER TWENTY-FIRST.
The Man Of

${repeated('Another day.')}

CHAPTER XXII.
A very long title that goes on and on and on and on and on and on and on and on and on and on

${repeated('More days.')}

CHAPTER XXIII. A Title
And More

${repeated('Even more days.')}

Preface.

Chapter I. Down the Rabbit-Hole.

Appendix.

${repeated('Text between.')}

${repeated('Text after.')}

Chapter XXIV. L'amour, the End.

${repeated('The end.')}`);
    assert.deepEqual(result.chapters, [
        {'title': 'Etymology', 'char': 44, 'level': 1},
        {'title': 'Etymology', 'char': 56, 'level': 1},
        {'title': 'Part I. Down the Rabbit-Hole', 'char': 168, 'level': 1},
        {'title': 'Part II', 'char': 251, 'level': 1},
        {'title': 'Preface', 'char': 674, 'level': 1},
        {'title': 'Chapter I. Down the Rabbit-Hole', 'char': 684, 'level': 1},
        {'title': 'Appendix', 'char': 718, 'level': 1},
        {'title': 'Chapter XXIV. L\'amour, the End', 'char': 809, 'level': 1}
    ]);
    checkChapters(result);
});


test('prepareText, no header', () => {
    const result = prepareText(bareText);
    assert.equal(result.title, '');
    assert.equal(result.author, '');
    assert.equal(result.text, `\
Chapter I.

${repeated('The first chapter.')}

${repeated('A paragraph.')}`);
    assert.deepEqual(result.chapters, [
        {'title': 'Chapter I', 'char': 0, 'level': 1}
    ]);
    checkChapters(result);
});


test('prepareText, transcriber\'s notes at the end', () => {
    const result = prepareText(notesText);
    assert.equal(result.title, '');
    assert.equal(result.author, '');
    assert.equal(result.text, `\
${shortParagraphsText}`);
    assert.deepEqual(result.chapters, []);
    checkChapters(result);
});


test('prepareText, empty', () => {
    assert.deepEqual(prepareText(''), {'text': '', 'chapters': [], 'title': '', 'author': ''});
    assert.deepEqual(prepareText('Text.', {'title': 'Book Title'}), {
        'text': 'Book Title.\n\nText.',
        'chapters': [],
        'title': 'Book Title',
        'author': ''
    });
});
