// The book catalog (data/catalog.json) - search, source listings, and paging. Pure functions over the parsed catalog:
//
//   {'version': 1, 'generated': '<ISO time>', 'sources': {'<sourceId>': {'name', 'url', 'description'}, ...},
//    'books': [{'source', 'id', 'title', 'subtitle'?, 'author', 'description', 'subjects', 'language', 'cover',
//               'rank', 'words', 'text': {'url', 'format', 'encoding'?}}, ...]}


// Letters that Unicode normalization (NFD) doesn't decompose to ASCII
const letterFolds = {
    'æ': 'ae', 'œ': 'oe', 'ß': 'ss', 'ø': 'o', 'đ': 'd', 'ð': 'd', 'þ': 'th', 'ł': 'l', 'ı': 'i', 'ŋ': 'n'
};
const letterFoldRegex = new RegExp(`[${Object.keys(letterFolds).join('')}]`, 'g');


// Text normalized for searching and comparison - lowercase, without accents, apostrophes removed ("Alice's" is
// "alices"), "&" as "and", and other punctuation as word breaks, with single spaces between words
export function normalizeText(text) {
    return text.normalize('NFD').toLowerCase()
        .replace(/[̀-ͯ]/g, '')
        .replace(letterFoldRegex, (letter) => letterFolds[letter])
        .replace(/['’ʼ‘`]/g, '')
        .replace(/&/g, ' and ')
        .replace(/[^a-z0-9]+/g, ' ')
        .trim();
}


// A book's normalized search words - title, subtitle, and author words, and subject words (cached per book)
const bookWordsCache = new WeakMap();
function bookWords(book) {
    let words = bookWordsCache.get(book);
    if (words === undefined) {
        const splitWords = (text) => {
            const normal = normalizeText(text);
            return normal === '' ? [] : normal.split(' ');
        };
        words = {
            'primary': splitWords([book.title, book.subtitle ?? '', book.author].join(' ')),
            'subjects': splitWords((book.subjects ?? []).join(' '))
        };
        bookWordsCache.set(book, words);
    }
    return words;
}


// Does any of the words start with the query word?
function wordsMatch(words, queryWord) {
    return words.some((word) => word.startsWith(queryWord));
}


// Compare books by rank, then by the catalog order of their sources, then by title
function compareRank(catalog) {
    const sourceIds = Object.keys(catalog.sources ?? {});
    const sourceOrder = (book) => {
        const index = sourceIds.indexOf(book.source);
        return index === -1 ? sourceIds.length : index;
    };
    return (book1, book2) => (book1.rank - book2.rank) || (sourceOrder(book1) - sourceOrder(book2)) ||
        book1.title.localeCompare(book2.title);
}


// A source's books by popularity rank - for 'all', every source's books interleaved by rank (the most popular book of
// each source, then the second of each source, and so on)
export function listBooks(catalog, source = 'all') {
    const books = source === 'all' ? [...catalog.books] : catalog.books.filter((book) => book.source === source);
    return books.sort(compareRank(catalog));
}


// Search a source's books (or 'all' sources') by title, subtitle, author, and subjects - every query word must start a
// word of the book's text. Books matched by title, subtitle, and author come first, then books that needed their
// subjects to match, each by rank as in listBooks. An empty query lists the source's books.
export function searchBooks(catalog, query, source = 'all') {
    const books = listBooks(catalog, source);
    const normalQuery = normalizeText(query);
    if (normalQuery === '') {
        return books;
    }
    const queryWords = normalQuery.split(' ');
    const primaryMatches = [];
    const subjectMatches = [];
    for (const book of books) {
        const words = bookWords(book);
        let subjectMatch = false;
        let match = true;
        for (const queryWord of queryWords) {
            if (!wordsMatch(words.primary, queryWord)) {
                if (!wordsMatch(words.subjects, queryWord)) {
                    match = false;
                    break;
                }
                subjectMatch = true;
            }
        }
        if (match) {
            (subjectMatch ? subjectMatches : primaryMatches).push(book);
        }
    }
    return [...primaryMatches, ...subjectMatches];
}


// A book by its source and id, or null if it isn't in the catalog
export function findBook(catalog, source, id) {
    return catalog.books.find((book) => book.source === source && book.id === id) ?? null;
}


// A page of items - page is 1-based (less than 1, or not a number, is the first page)
export function pageItems(items, page = 1, pageSize = 24) {
    const pageNumber = Number.isFinite(page) ? Math.max(1, Math.floor(page)) : 1;
    const start = (pageNumber - 1) * pageSize;
    return {
        'items': items.slice(start, start + pageSize),
        'page': pageNumber,
        'pageCount': Math.ceil(items.length / pageSize),
        'total': items.length,
        'hasMore': start + pageSize < items.length
    };
}
