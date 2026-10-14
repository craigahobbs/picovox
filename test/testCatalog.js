// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

import {findBook, listBooks, normalizeText, pageItems, searchBooks} from '../lib/catalog.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


// A small catalog - two sources, ranks that interleave, subtitles, several authors, and subjects
function testCatalog() {
    const book = (source, id, rank, title, author, subjects, subtitle = null) => ({
        source, id, title, ...(subtitle === null ? {} : {subtitle}), author, subjects, rank,
        'description': '', 'language': 'en', 'cover': null, 'words': null,
        'text': {'url': `https://example.com/${id}`, 'format': source === 'gutenberg' ? 'gutenberg-text' : 'epub'}
    });
    return {
        'version': 1,
        'generated': '2026-10-14T00:00:00.000Z',
        'sources': {
            'standardebooks': {'name': 'Standard Ebooks', 'url': 'https://standardebooks.org', 'description': 'SE'},
            'gutenberg': {'name': 'Project Gutenberg', 'url': 'https://www.gutenberg.org', 'description': 'PG'}
        },
        'books': [
            book('gutenberg', '2701', 2, 'Moby Dick', 'Herman Melville', ['Sea stories', 'Whaling'], 'Or, The Whale'),
            book('standardebooks', '/ebooks/jane-austen/pride-and-prejudice', 1, 'Pride and Prejudice', 'Jane Austen',
                ['Fiction', 'Romance']),
            book('gutenberg', '1342', 1, 'Pride and Prejudice', 'Jane Austen', ['Courtship', 'Fiction']),
            book('standardebooks', '/ebooks/arthur-conan-doyle/the-adventures-of-sherlock-holmes', 3,
                'The Adventures of Sherlock Holmes', 'Arthur Conan Doyle', ['Fiction', 'Mystery', 'Shorts']),
            book('standardebooks', '/ebooks/mary-shelley/frankenstein', 2, 'Frankenstein', 'Mary Shelley',
                ['Fiction', 'Horror', 'Science Fiction']),
            book('gutenberg', '45', 3, 'Anne of Green Gables', 'L. M. Montgomery', ['Orphans', 'Fiction']),
            book('gutenberg', '16328', 4, 'Beowulf', '', []),
            book('other', 'x1', 1, 'Émile', 'Jean-Jacques Rousseau', ['Education']),
            book('gutenberg', '1661', 4, 'Alice’s Adventures in Wonderland', 'Lewis Carroll', ['Fantasy'])
        ]
    };
}


// The source and id of each book
function bookKeys(books) {
    return books.map((book) => `${book.source}:${book.id}`);
}


test('normalizeText', () => {
    // Case and spaces
    assert.equal(normalizeText('  Moby   DICK  '), 'moby dick');
    assert.equal(normalizeText(''), '');
    assert.equal(normalizeText('   '), '');

    // Accents and letters without decompositions
    assert.equal(normalizeText('Émile Zola'), 'emile zola');
    assert.equal(normalizeText('Brontë'), 'bronte');
    assert.equal(normalizeText('Encyclopædia Œuvres Straße Søren Łódź þorn Đuro ð ı ŋ'),
        'encyclopaedia oeuvres strasse soren lodz thorn duro d i n');

    // Apostrophes join, ampersands are "and", and other punctuation separates words
    assert.equal(normalizeText('Alice’s Adventures'), 'alices adventures');
    assert.equal(normalizeText("Grimms' Fairy Tales"), 'grimms fairy tales');
    assert.equal(normalizeText('Tess of the d‘Urbervilles'), 'tess of the durbervilles');
    assert.equal(normalizeText('Dombey & Son'), 'dombey and son');
    assert.equal(normalizeText('Moby-Dick; Or, The Whale'), 'moby dick or the whale');
    assert.equal(normalizeText('H. G. Wells'), 'h g wells');
    assert.equal(normalizeText('Nineteen Eighty—Four (1984)!'), 'nineteen eighty four 1984');
});


test('listBooks', () => {
    const catalog = testCatalog();

    // A source's books, by rank
    assert.deepEqual(bookKeys(listBooks(catalog, 'gutenberg')), [
        'gutenberg:1342', 'gutenberg:2701', 'gutenberg:45', 'gutenberg:1661', 'gutenberg:16328'
    ]);
    assert.deepEqual(bookKeys(listBooks(catalog, 'standardebooks')), [
        'standardebooks:/ebooks/jane-austen/pride-and-prejudice', 'standardebooks:/ebooks/mary-shelley/frankenstein',
        'standardebooks:/ebooks/arthur-conan-doyle/the-adventures-of-sherlock-holmes'
    ]);

    // All sources, interleaved by rank in the catalog's source order (unlisted sources last), then by title
    assert.deepEqual(bookKeys(listBooks(catalog)), [
        'standardebooks:/ebooks/jane-austen/pride-and-prejudice', 'gutenberg:1342', 'other:x1',
        'standardebooks:/ebooks/mary-shelley/frankenstein', 'gutenberg:2701',
        'standardebooks:/ebooks/arthur-conan-doyle/the-adventures-of-sherlock-holmes', 'gutenberg:45',
        'gutenberg:1661', 'gutenberg:16328'
    ]);
    assert.deepEqual(bookKeys(listBooks(catalog, 'all')), bookKeys(listBooks(catalog)));

    // An unknown source
    assert.deepEqual(listBooks(catalog, 'unknown'), []);

    // The catalog's books aren't reordered
    assert.equal(catalog.books[0].id, '2701');

    // A catalog without sources
    const noSources = {'books': [catalog.books[2], catalog.books[1]]};
    assert.deepEqual(bookKeys(listBooks(noSources)), ['gutenberg:1342', 'standardebooks:/ebooks/jane-austen/pride-and-prejudice']);
});


test('searchBooks', () => {
    const catalog = testCatalog();

    // An empty query lists the source's books
    assert.deepEqual(bookKeys(searchBooks(catalog, '')), bookKeys(listBooks(catalog)));
    assert.deepEqual(bookKeys(searchBooks(catalog, ' -- ', 'gutenberg')), bookKeys(listBooks(catalog, 'gutenberg')));

    // Title words, by rank across sources
    assert.deepEqual(bookKeys(searchBooks(catalog, 'pride')), ['standardebooks:/ebooks/jane-austen/pride-and-prejudice', 'gutenberg:1342']);
    assert.deepEqual(bookKeys(searchBooks(catalog, 'pride', 'gutenberg')), ['gutenberg:1342']);

    // Every query word must match, as a word prefix, in any order, ignoring case and punctuation
    assert.deepEqual(bookKeys(searchBooks(catalog, 'HOLMES sher')), [
        'standardebooks:/ebooks/arthur-conan-doyle/the-adventures-of-sherlock-holmes'
    ]);
    assert.deepEqual(searchBooks(catalog, 'holmes pride'), []);
    assert.deepEqual(searchBooks(catalog, 'erlock'), []);

    // Subtitles and authors
    assert.deepEqual(bookKeys(searchBooks(catalog, 'whale')), ['gutenberg:2701']);
    assert.deepEqual(bookKeys(searchBooks(catalog, 'Melville')), ['gutenberg:2701']);
    assert.deepEqual(bookKeys(searchBooks(catalog, 'l m montgomery')), ['gutenberg:45']);

    // Accents and apostrophes
    assert.deepEqual(bookKeys(searchBooks(catalog, 'emile')), ['other:x1']);
    assert.deepEqual(bookKeys(searchBooks(catalog, 'Rousseau Émile')), ['other:x1']);
    assert.deepEqual(bookKeys(searchBooks(catalog, "alice's")), ['gutenberg:1661']);
    assert.deepEqual(bookKeys(searchBooks(catalog, 'alice')), ['gutenberg:1661']);

    // Subjects, ranked below title and author matches
    assert.deepEqual(bookKeys(searchBooks(catalog, 'whaling')), ['gutenberg:2701']);
    assert.deepEqual(bookKeys(searchBooks(catalog, 'fiction')), [
        'standardebooks:/ebooks/jane-austen/pride-and-prejudice', 'gutenberg:1342', 'standardebooks:/ebooks/mary-shelley/frankenstein',
        'standardebooks:/ebooks/arthur-conan-doyle/the-adventures-of-sherlock-holmes', 'gutenberg:45'
    ]);
    assert.deepEqual(bookKeys(searchBooks(catalog, 'science')), ['standardebooks:/ebooks/mary-shelley/frankenstein']);
    assert.deepEqual(bookKeys(searchBooks(catalog, 'adventures')), [
        'standardebooks:/ebooks/arthur-conan-doyle/the-adventures-of-sherlock-holmes', 'gutenberg:1661'
    ]);
    assert.deepEqual(bookKeys(searchBooks(catalog, 'austen fiction')), [
        'standardebooks:/ebooks/jane-austen/pride-and-prejudice', 'gutenberg:1342'
    ]);

    // A title match outranks a subject match of a more popular book
    const mixed = testCatalog();
    mixed.books.push({
        ...mixed.books[4], 'id': '/ebooks/h-g-wells/the-horror', 'rank': 9, 'title': 'The Horror', 'author': 'H. G. Wells', 'subjects': []
    });
    assert.deepEqual(bookKeys(searchBooks(mixed, 'horror')), [
        'standardebooks:/ebooks/h-g-wells/the-horror', 'standardebooks:/ebooks/mary-shelley/frankenstein'
    ]);

    // A book without subjects or an author
    assert.deepEqual(bookKeys(searchBooks(catalog, 'beowulf')), ['gutenberg:16328']);
    const noSubjects = {'sources': {}, 'books': [{'source': 'gutenberg', 'id': '1', 'rank': 1, 'title': 'Beowulf', 'author': ''}]};
    assert.deepEqual(bookKeys(searchBooks(noSubjects, 'beo')), ['gutenberg:1']);
    assert.deepEqual(searchBooks(noSubjects, 'fiction'), []);

    // Repeated searches use the cached book words
    assert.deepEqual(bookKeys(searchBooks(catalog, 'pride')), ['standardebooks:/ebooks/jane-austen/pride-and-prejudice', 'gutenberg:1342']);
});


test('findBook', () => {
    const catalog = testCatalog();
    assert.equal(findBook(catalog, 'gutenberg', '1342').title, 'Pride and Prejudice');
    assert.equal(findBook(catalog, 'standardebooks', '/ebooks/mary-shelley/frankenstein').title, 'Frankenstein');
    assert.equal(findBook(catalog, 'standardebooks', '1342'), null);
    assert.equal(findBook(catalog, 'gutenberg', '99999'), null);
});


test('pageItems', () => {
    const items = Array.from({'length': 50}, (_item, index) => index);

    // The first page, by default
    assert.deepEqual(pageItems(items), {
        'items': items.slice(0, 24), 'page': 1, 'pageCount': 3, 'total': 50, 'hasMore': true
    });

    // Middle and last pages
    assert.deepEqual(pageItems(items, 2), {
        'items': items.slice(24, 48), 'page': 2, 'pageCount': 3, 'total': 50, 'hasMore': true
    });
    assert.deepEqual(pageItems(items, 3), {
        'items': [48, 49], 'page': 3, 'pageCount': 3, 'total': 50, 'hasMore': false
    });

    // Past the end
    assert.deepEqual(pageItems(items, 4), {
        'items': [], 'page': 4, 'pageCount': 3, 'total': 50, 'hasMore': false
    });

    // Page sizes
    assert.deepEqual(pageItems(items, 5, 10), {
        'items': [40, 41, 42, 43, 44, 45, 46, 47, 48, 49], 'page': 5, 'pageCount': 5, 'total': 50, 'hasMore': false
    });
    assert.deepEqual(pageItems(items, 1, 50).hasMore, false);

    // Pages before the first, fractional, and not numbers
    assert.equal(pageItems(items, 0).page, 1);
    assert.equal(pageItems(items, -3).page, 1);
    assert.equal(pageItems(items, 2.7).page, 2);
    assert.equal(pageItems(items, Number.NaN).page, 1);
    assert.equal(pageItems(items, '2').page, 1);

    // No items
    assert.deepEqual(pageItems([]), {'items': [], 'page': 1, 'pageCount': 0, 'total': 0, 'hasMore': false});
});
