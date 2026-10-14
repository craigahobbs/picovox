// Build the book catalog, data/catalog.json - every Standard Ebooks book and the most-downloaded English Project
// Gutenberg books whose plain text is in the GITenberg mirror on GitHub (the app fetches texts from their sources with
// CORS at play time, so the catalog holds only metadata and each book's text URL).
//
// node tools/catalog.js [--gutenberg <count>] [--standardebooks <count>] [--max-age <days>] [--cache-only] [-o <file>]
//
//   --gutenberg <count>       the number of top Gutenberg books (by downloads) to consider (default 1000)
//   --standardebooks <count>  only the most popular Standard Ebooks books (default all)
//   --max-age <days>          refetch cached listing and book pages older than this (default 30)
//   --cache-only              no network requests - build from the cache alone (uncached pages and unresolved
//                             Gutenberg texts are left out)
//   -o <file>                 the output file (default data/catalog.json)
//
// Every remote page is cached under out/catalog/, so a rerun is fast and an interrupted build resumes. Requests are
// throttled per site (Standard Ebooks about 2 a second) and retried with backoff. GITenberg repositories are found
// in the gitberg project's published repository list; the GitHub search API (10 searches a minute without a
// GITHUB_TOKEN) is used only for books missing from the list.

import {dirname, join} from 'node:path';
import {existsSync, mkdirSync, readFileSync, renameSync, statSync, writeFileSync} from 'node:fs';
import {JSDOM} from 'jsdom';
import {fileURLToPath} from 'node:url';
import {normalizeText} from '../lib/catalog.js';
import process from 'node:process';


const root = fileURLToPath(new URL('..', import.meta.url));
const cacheDir = join(root, 'out', 'catalog');
const userAgent = 'voice-synth-catalog/1.0 (+https://github.com/craigahobbs/voice-synth)';

const seURL = 'https://standardebooks.org';
const gutendexURL = 'https://gutendex.com/books/';
const gitenbergListURL = 'https://raw.githubusercontent.com/gitenberg-dev/gitberg/master/gitenberg/data/GITenberg_repo_list.tsv';
const gitenbergRawURL = 'https://raw.githubusercontent.com/GITenberg/';
const githubSearchURL = 'https://api.github.com/search/repositories';

// The minimum milliseconds between requests to a host
const hostIntervals = {
    'standardebooks.org': 500,
    'gutendex.com': 1000,
    'raw.githubusercontent.com': 100,
    'api.github.com': 1000
};

// The catalog's sources
const sources = {
    'standardebooks': {
        'name': 'Standard Ebooks',
        'url': seURL,
        'description': 'Carefully edited, beautifully typeset public-domain classics, freely available.'
    },
    'gutenberg': {
        'name': 'Project Gutenberg',
        'url': 'https://www.gutenberg.org',
        'description': 'The oldest digital library - tens of thousands of free public-domain books, read from its GITenberg mirror.'
    }
};

// Description and subject limits
const descriptionLength = 600;
const subjectCount = 8;


// Parse the command line
function parseArgs(args) {
    const options = {
        'gutenberg': 1000, 'standardebooks': Infinity, 'maxAge': 30, 'cacheOnly': false, 'output': join(root, 'data', 'catalog.json')
    };
    for (let ix = 0; ix < args.length; ix++) {
        const arg = args[ix];
        if (arg === '--gutenberg') {
            options.gutenberg = Number(args[++ix]);
        } else if (arg === '--standardebooks') {
            options.standardebooks = Number(args[++ix]);
        } else if (arg === '--max-age') {
            options.maxAge = Number(args[++ix]);
        } else if (arg === '--cache-only') {
            options.cacheOnly = true;
        } else if (arg === '-o') {
            options.output = args[++ix];
        } else {
            throw new Error(`Unknown argument "${arg}"`);
        }
    }
    for (const name of ['gutenberg', 'standardebooks', 'maxAge']) {
        if (Number.isNaN(options[name]) || options[name] < 0) {
            throw new Error(`Invalid ${name} "${options[name]}"`);
        }
    }
    return options;
}


//
// Logging
//


const startTime = Date.now();

// Write a progress line with the elapsed time
function log(message) {
    const seconds = Math.round((Date.now() - startTime) / 1000);
    process.stdout.write(`[${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}] ${message}\n`);
}


//
// Fetching - throttled per host, retried with backoff
//


const hostNextTimes = new Map();

// Sleep for milliseconds
function sleep(milliseconds) {
    return new Promise((resolve) => {
        setTimeout(resolve, milliseconds);
    });
}


// Wait for a host's turn
async function throttle(url) {
    const {host} = new URL(url);
    const now = Date.now();
    const nextTime = Math.max(now, hostNextTimes.get(host) ?? 0);
    hostNextTimes.set(host, nextTime + (hostIntervals[host] ?? 500));
    if (nextTime > now) {
        await sleep(nextTime - now);
    }
}


// Fetch a URL - network errors, timeouts, 429s, and 5xx responses are retried with exponential backoff (or the
// response's Retry-After)
async function fetchRetry(url, {method = 'GET', headers = {}, timeout = 60000, attempts = 6} = {}) {
    for (let attempt = 1; ; attempt++) {
        await throttle(url);
        let retryAfter = 2 ** attempt;
        try {
            const response = await fetch(url, {
                method,
                'headers': {'User-Agent': userAgent, ...headers},
                'signal': AbortSignal.timeout(timeout)
            });
            if (response.status !== 429 && response.status < 500) {
                return response;
            }
            await response.body?.cancel();
            retryAfter = Math.max(retryAfter, Number(response.headers.get('Retry-After')) || 0);
            if (attempt >= attempts) {
                throw new Error(`${url}: error ${response.status}`);
            }
            log(`  ${url}: error ${response.status} - retrying in ${retryAfter} s`);
        } catch (error) {
            if (attempt >= attempts || error.message.startsWith(url)) {
                throw error;
            }
            log(`  ${url}: ${error.cause?.code ?? error.name} - retrying in ${retryAfter} s`);
        }
        await sleep(retryAfter * 1000);
    }
}


// Fetch a URL's text, cached in a file under out/catalog/ - null for a 404 (cached as well). Cached files older than
// maxAge days are refetched (Infinity keeps them forever). With --cache-only, any cached file is used and an
// uncached URL is null.
let cacheOnly = false;
async function fetchCached(url, cacheName, maxAge, timeout = 60000) {
    const cachePath = join(cacheDir, cacheName);
    const missingPath = `${cachePath}.404`;
    for (const path of [cachePath, missingPath]) {
        if (existsSync(path) && (cacheOnly || (Date.now() - statSync(path).mtimeMs) < maxAge * 86400000)) {
            return path === cachePath ? readFileSync(path, 'utf-8') : null;
        }
    }
    if (cacheOnly) {
        return null;
    }
    const response = await fetchRetry(url, {timeout});
    if (response.status === 404) {
        await response.body?.cancel();
        writeFileAtomic(missingPath, '');
        return null;
    }
    if (!response.ok) {
        throw new Error(`${url}: error ${response.status}`);
    }
    const text = await response.text();
    writeFileAtomic(cachePath, text);
    return text;
}


// Write a file, creating its directory, so an interrupted write never leaves a partial file
function writeFileAtomic(path, text) {
    mkdirSync(dirname(path), {'recursive': true});
    writeFileSync(`${path}.tmp`, text);
    renameSync(`${path}.tmp`, path);
}


// Read a JSON cache file (an empty object if there isn't one)
function readJSONCache(cacheName) {
    const path = join(cacheDir, cacheName);
    return existsSync(path) ? JSON.parse(readFileSync(path, 'utf-8')) : {};
}


// Parse HTML into a document - with one window's DOMParser (a JSDOM window per page holds gigabytes over a crawl)
const domParser = new (new JSDOM('').window.DOMParser)();
function parseHTML(html) {
    return domParser.parseFromString(html, 'text/html');
}


//
// Text helpers
//


// Collapse whitespace
function collapseSpace(text) {
    return text.replace(/\s+/g, ' ').trim();
}


// A description trimmed to at most descriptionLength characters, at a sentence end if there is one past a third of
// the limit, else at a word break with an ellipsis
function trimDescription(text) {
    const description = collapseSpace(text);
    if (description.length <= descriptionLength) {
        return description;
    }
    const head = description.slice(0, descriptionLength + 1);
    const sentenceEnd = Math.max(...[...head.matchAll(/[.!?][”"’)]?(?=\s)/g)].map((match) => match.index + match[0].length));
    if (sentenceEnd > descriptionLength / 3) {
        return head.slice(0, sentenceEnd);
    }
    return `${head.slice(0, head.lastIndexOf(' ')).replace(/[\s,;:—–-]+$/, '')}…`;
}


// A subject list - cleaned, deduplicated (normalized), at most subjectCount
function uniqueSubjects(subjects) {
    const seen = new Set();
    const result = [];
    for (const subject of subjects.map(collapseSpace)) {
        const key = normalizeText(subject);
        if (key !== '' && !seen.has(key) && result.length < subjectCount) {
            seen.add(key);
            result.push(subject);
        }
    }
    return result;
}


// A key for matching the same work across sources - the normalized title without a leading article or word breaks
// ("The Case-Book of Sherlock Holmes" matches "The Casebook of Sherlock Holmes")
function titleKey(title) {
    return normalizeText(title).replace(/^(?:the|a|an) /, '').replace(/ /g, '');
}


// The surname keys of a display author list ('Mary Shelley, Percy Bysshe Shelley') - each name's last word, first
// four letters, so spelling variants of transliterated names match (Dostoevsky, Dostoyevsky); anonymous is ''
function surnameKeys(author) {
    const keys = author.split(/\s*,\s*/).map((name) => normalizeText(name).split(' ').pop().slice(0, 4)).filter((key) => key !== '');
    return keys.length > 0 ? keys : [''];
}


//
// Standard Ebooks
//


// A Standard Ebooks book id - the book's path on the site
const seIdPattern = /^\/ebooks(\/[a-z0-9][a-z0-9_-]*){2,4}$/;


// Every Standard Ebooks book, by popularity - from the catalog listing pages and each book's page
async function standardEbooks(options) {
    // The listing pages, in popularity order
    const listed = [];
    const listedIds = new Set();
    for (let page = 1; listed.length < options.standardebooks; page++) {
        const url = `${seURL}/ebooks?sort=popularity&per-page=48&page=${page}`;
        const html = await fetchCached(url, `standardebooks/list-${page}.html`, options.maxAge);
        if (html === null) {
            break;
        }
        const doc = parseHTML(html);

        // A page past the end shows the last page
        const nav = doc.querySelector('nav.pagination');
        const currentPage = nav?.querySelector('a[aria-current="page"]')?.textContent.trim() ?? '1';
        if (currentPage !== String(page)) {
            break;
        }
        for (const item of doc.querySelectorAll('li[typeof="schema:Book"]')) {
            const id = item.getAttribute('about') ?? '';
            if (!seIdPattern.test(id) || listedIds.has(id) || listed.length >= options.standardebooks) {
                continue;
            }
            listedIds.add(id);
            const coverSrc = item.querySelector('img')?.getAttribute('src');
            listed.push({
                id,
                'title': collapseSpace(item.querySelector('p:not(.author) [property="schema:name"]')?.textContent ?? ''),
                'authors': [...item.querySelectorAll('p.author [property="schema:name"]')].map((span) => collapseSpace(span.textContent)),
                'cover': coverSrc ? new URL(coverSrc, seURL).href : null
            });
        }
        log(`Standard Ebooks: listing page ${page} - ${listed.length} books`);
        if (!nav?.querySelector('a[rel="next"][href]')) {
            break;
        }
    }

    // Each book's page
    const books = [];
    let missing = 0;
    for (const [index, item] of listed.entries()) {
        const cacheName = `standardebooks/book${item.id.slice('/ebooks'.length).replace(/\//g, '_')}.html`;
        const html = await fetchCached(`${seURL}${item.id}`, cacheName, options.maxAge);
        const book = html === null ? null : standardEbook(item, parseHTML(html));
        if (book === null) {
            log(`  Standard Ebooks: ${item.id} - no book page or EPUB download`);
            missing++;
        } else {
            book.rank = books.length + 1;
            books.push(book);
        }
        if ((index + 1) % 100 === 0) {
            log(`Standard Ebooks: ${index + 1} of ${listed.length} book pages`);
        }
    }
    log(`Standard Ebooks: ${books.length} books (${missing} without a book page or EPUB)`);
    return books;
}


// A Standard Ebooks catalog book from its listing item and book page, or null if it has no EPUB download
function standardEbook(item, doc) {
    const article = doc.querySelector('article[typeof="schema:Book"]') ?? doc;
    const meta = (property) => article.querySelector(`meta[property="schema:${property}"]`)?.getAttribute('content') ?? null;

    // The EPUB download - the compatible EPUB, not the Kobo or advanced ones
    const links = [...doc.querySelectorAll('a[href*="/downloads/"]')].map((link) => link.getAttribute('href'));
    const epubHref = links.find((link) => /\.epub$/.test(link) && !/(\.kepub|_advanced)\.epub$/.test(link)) ??
        links.find((link) => /\.epub$/.test(link) && !/\.kepub\.epub$/.test(link));
    if (!epubHref) {
        return null;
    }

    // Title, subtitle, and authors - the book page's, falling back to the listing's
    const hgroup = article.querySelector('hgroup');
    const title = collapseSpace(hgroup?.querySelector('h1')?.textContent ?? '') || item.title;
    const subtitle = collapseSpace(hgroup?.querySelector('[property="schema:alternativeHeadline"]')?.textContent ?? '');
    const pageAuthors = [...(hgroup?.querySelectorAll('[property="schema:author"] [property="schema:name"]') ?? [])]
        .map((span) => collapseSpace(span.textContent));
    const authors = (pageAuthors.length > 0 ? pageAuthors : item.authors).filter((name) => name !== '');

    // The short blurb, else the long description's first paragraph
    const description = meta('abstract') ?? article.querySelector('[property="schema:description"] p')?.textContent ?? '';
    const words = Number(meta('wordCount'));
    return {
        'source': 'standardebooks',
        'id': item.id,
        title,
        ...(subtitle === '' ? {} : {subtitle}),
        'author': authors.join(', '),
        'description': trimDescription(description),
        'subjects': uniqueSubjects([...article.querySelectorAll('ul.tags a')].map((link) => link.textContent)),
        'language': (meta('inLanguage') ?? 'en').split('-')[0],
        'cover': item.cover,
        'rank': 0,
        'words': words > 0 ? words : null,
        'text': {'url': `${new URL(epubHref, seURL).href}?source=download`, 'format': 'epub'}
    };
}


//
// Project Gutenberg
//


// The top Gutenberg books by downloads from Gutendex - English texts that aren't under copyright
async function gutendexBooks(options) {
    const results = [];
    for (let page = 1; results.length < options.gutenberg; page++) {
        const url = `${gutendexURL}?languages=en&mime_type=text%2Fplain&page=${page}`;
        const json = await fetchCached(url, `gutendex/page-${page}.json`, options.maxAge, 180000);
        if (json === null) {
            break;
        }
        const data = JSON.parse(json);
        for (const result of data.results) {
            if (result.media_type === 'Text' && result.copyright !== true && results.length < options.gutenberg) {
                results.push(result);
            }
        }
        log(`Gutendex: page ${page} - ${results.length} books`);
        if (!data.next) {
            break;
        }
    }
    return results;
}


// A Gutendex title as a title and subtitle - one line, split at the first colon, semicolon, or dash outside
// parentheses, without a generic subtitle ("A Novel", "Complete"), and a long subtitle shortened
const subtitleLength = 100;
function gutenbergTitle(rawTitle) {
    const oneLine = rawTitle.replace(/\s*[\r\n]+\s*/g, (match, offset) => (/[;:,]$/.test(rawTitle.slice(0, offset)) ? ' ' : ': ')).trim();
    const separator = [...oneLine.matchAll(/\s*(?:[:;]|\s[—–]\s|\s-{1,2}\s|—|--)\s*/g)].find((match) => {
        const prefix = oneLine.slice(0, match.index);
        return match.index >= 3 && prefix.split('(').length === prefix.split(')').length;
    });
    if (!separator) {
        return {'title': titleCase(oneLine), 'subtitle': ''};
    }
    const title = oneLine.slice(0, separator.index).replace(/[,.]$/, '');
    let subtitle = oneLine.slice(separator.index + separator[0].length).trim();
    if (/^(?:(?:a|an)\s+(?:novel|romance|tale|story|play|poem)|complete)\.?$/i.test(subtitle)) {
        subtitle = '';
    } else if (subtitle.length > subtitleLength) {
        subtitle = `${subtitle.slice(0, subtitle.lastIndexOf(' ', subtitleLength)).replace(/[\s,;:—–-]+$/, '')}…`;
    }
    return {'title': titleCase(title), 'subtitle': titleCase(subtitle)};
}


// Title-case a title in sentence case or lowercase ("The three musketeers" is "The Three Musketeers", "or, the modern
// prometheus" is "Or, the Modern Prometheus") - a title with mostly capitalized words is left as it is
const smallWords = new Set([
    'a', 'an', 'and', 'as', 'at', 'but', 'by', 'de', 'du', 'for', 'from', 'in', 'into', 'la', 'le', 'nor', 'of', 'on', 'or',
    'the', 'to', 'van', 'von', 'with'
]);
function titleCase(text) {
    const capitalize = (word) => `${word.charAt(0).toUpperCase()}${word.slice(1)}`;
    const words = text.split(' ');
    const isSmall = (word) => smallWords.has(word.replace(/\W/g, '')) || /^d['’]/.test(word);
    const majorWords = words.slice(1).filter((word) => /^[a-zA-Z]/.test(word) && !isSmall(word));
    const lowerCount = majorWords.filter((word) => /^[a-z]/.test(word)).length;
    if (lowerCount === 0 || lowerCount < majorWords.length - lowerCount) {
        return capitalize(text);
    }
    return words.map((word, index) => (index > 0 && isSmall(word) ? word : capitalize(word))).join(' ');
}


// A Gutendex author name ('Last, First (Full Names), suffix') in display order ('First Last')
const epithetRegex = /^(?:Emperor|Empress|King|Queen|Saint|St\.|Pope|Prince|Princess|Duke|Earl|Lord|Lady|Baron|Count|Countess)\b/;
function gutenbergAuthor(name) {
    const parts = name.replace(/\s*\([^)]*\)/g, '').split(/\s*,\s*/).filter((part) => part !== '');
    if (parts.length < 2) {
        return parts.join('');
    }
    // An epithet ('Marcus Aurelius, Emperor of Rome') isn't a first name
    if (epithetRegex.test(parts[1])) {
        return parts[0];
    }
    // A lowercase particle ('Augustine, of Hippo, Saint') follows the name
    if (/^[a-z]/.test(parts[1])) {
        return `${parts[0]} ${parts[1]}`;
    }
    const suffix = parts.slice(2).find((part) => /^(?:Jr\.?|Sr\.?|[IVX]+)$/.test(part));
    return `${parts[1]} ${parts[0]}${suffix ? ` ${suffix}` : ''}`;
}


// A Gutendex summary as a description - without the "(This is an automatically generated summary.)" note, and
// without the leading '"Title" by Author is' phrase (the title and author are shown with it)
function gutenbergDescription(summary) {
    let description = collapseSpace(summary).replace(/\s*\(This is an automatically generated summary\.?\)\s*$/i, '');
    const match = description.match(/^["“][^"”]{1,300}["”],?\s+(?:\([^)]{0,100}\)\s+)?(?:by\s+.{1,200}?\s+)?(?:is|are)\s+(an?|the)\s+(\S)/);
    if (match) {
        description = `${match[1].charAt(0).toUpperCase()}${match[1].slice(1)} ${match[2]}${description.slice(match[0].length)}`;
    }
    return trimDescription(description);
}


// Gutenberg subjects - the bookshelf categories, then the Library of Congress subjects (their topic, without
// subdivisions; characters as "First Last"), with "Fiction" for fiction
function gutenbergSubjects(result) {
    const shelves = result.bookshelves
        .filter((shelf) => /^(?:Category|Browsing):\s/.test(shelf))
        .map((shelf) => shelf.replace(/^[^:]+:\s*/, ''));
    const topics = result.subjects.map((subject) => {
        const [topic] = subject.split(' -- ');
        const plain = topic.replace(/\s*\([^)]*\)/g, '');
        const name = plain.replace(/,\s*(?:[a-z.]+\s*)?\d[^,]*$/, '');
        if ((name !== plain || topic !== plain) && name.includes(',')) {
            return gutenbergAuthor(name);
        }
        return plain;
    });
    const fiction = result.subjects.some((subject) => /\bfiction\b/i.test(subject)) ? ['Fiction'] : [];
    return uniqueSubjects([...fiction, ...shelves, ...topics]);
}


// Approximate words from a Gutenberg plain-text file's size - about 5.8 bytes a word, less the license (about 18 KB)
function gutenbergWords(bytes) {
    return bytes > 0 ? Math.max(100, Math.round((bytes - 18000) / 5.8 / 100) * 100) : null;
}


// The GITenberg repository names by Gutenberg id - from the gitberg project's published list
async function gitenbergRepositories(maxAge) {
    const tsv = await fetchCached(gitenbergListURL, 'gitenberg/GITenberg_repo_list.tsv', maxAge, 300000);
    const repositories = new Map();
    for (const line of (tsv ?? '').split('\n')) {
        const [id, name] = line.trim().split('\t');
        if (/^\d+$/.test(id) && name && name.endsWith(`_${id}`) && !repositories.has(id)) {
            repositories.set(id, name);
        }
    }
    return repositories;
}


// Find a GITenberg repository with the GitHub search API, honoring its rate limit - the name or null (cached)
const searchCacheName = 'gitenberg/search.json';
async function searchRepository(id, searchCache) {
    if (searchCache.has(id)) {
        return searchCache.get(id);
    }
    const headers = {'Accept': 'application/vnd.github+json'};
    if (process.env.GITHUB_TOKEN) {
        headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
    }
    const params = new URLSearchParams({'q': `_${id} in:name org:GITenberg`, 'per_page': '100'});
    for (;;) {
        const response = await fetchRetry(`${githubSearchURL}?${params}`, {headers});
        if (response.status === 403 || response.status === 429) {
            const resetTime = Number(response.headers.get('X-RateLimit-Reset')) * 1000 || (Date.now() + 60000);
            const wait = Math.max(1000, resetTime - Date.now() + 1000);
            log(`  GitHub search rate limit - waiting ${Math.round(wait / 1000)} s`);
            await response.body?.cancel();
            await sleep(wait);
            continue;
        }
        if (!response.ok) {
            throw new Error(`GitHub search: error ${response.status}`);
        }
        const data = await response.json();
        const name = data.items.find((item) => item.name.endsWith(`_${id}`))?.name ?? null;
        searchCache.set(id, name);
        writeFileAtomic(join(cacheDir, searchCacheName), JSON.stringify(Object.fromEntries(searchCache), null, 1));

        // Wait out an exhausted limit before the next search
        if (response.headers.get('X-RateLimit-Remaining') === '0') {
            const wait = Number(response.headers.get('X-RateLimit-Reset')) * 1000 - Date.now() + 1000;
            if (wait > 0) {
                log(`  GitHub search rate limit - waiting ${Math.round(wait / 1000)} s`);
                await sleep(wait);
            }
        }
        return name;
    }
}


// Resolve a Gutenberg book's text in its GITenberg repository - the UTF-8 file (<id>-0.txt), the plain file (<id>.txt,
// UTF-8 if it decodes as such, else Windows-1252), or the Latin-1 file (<id>-8.txt) - with HEAD requests
async function resolveText(id, repository) {
    for (const file of [`${id}-0.txt`, `${id}.txt`, `${id}-8.txt`]) {
        const url = `${gitenbergRawURL}${repository}/master/${file}`;
        const response = await fetchRetry(url, {'method': 'HEAD', 'headers': {'Accept-Encoding': 'identity'}});
        if (response.status === 404) {
            continue;
        }
        if (!response.ok) {
            throw new Error(`${url}: error ${response.status}`);
        }
        const bytes = Number(response.headers.get('Content-Length')) || 0;
        let encoding = file.endsWith('-8.txt') ? 'windows-1252' : 'utf-8';
        if (file === `${id}.txt`) {
            const body = await fetchRetry(url);
            try {
                new TextDecoder('utf-8', {'fatal': true}).decode(await body.arrayBuffer());
            } catch {
                encoding = 'windows-1252';
            }
        }
        return {'url': response.url || url, encoding, bytes};
    }
    return null;
}


// Is a Gutendex result a reference work (a dictionary, encyclopedia, index, or the like) - not for reading aloud?
const referenceTitleWords = [
    'dictionary', 'dictionaries', 'encyclop(?:a)?edia', 'cyclop(?:a)?edia', 'thesaurus', 'glossary', 'concordance', 'gazetteer', 'index of'
];
const referenceSubjects = [
    'dictionaries', 'encyclopedias and dictionaries', 'indexes', 'catalogs', 'glossaries, vocabularies', 'concordances', 'gazetteers'
];
const referenceTitleRegex = new RegExp(`\\b(?:${referenceTitleWords.join('|')})\\b`, 'i');
const referenceSubjectRegex = new RegExp(`^(?:${referenceSubjects.join('|')})\\b`, 'i');
function isReferenceWork(result) {
    return referenceTitleRegex.test(result.title) ||
        result.subjects.some((subject) => subject.split(' -- ').some((part) => referenceSubjectRegex.test(part.trim())));
}


// The top Gutenberg books, without Standard Ebooks duplicates or reference works, with texts in GITenberg
async function gutenbergBooks(options, seBooks) {
    const results = await gutendexBooks(options);

    // Standard Ebooks' works, by title and author surname
    const seWorks = new Set(seBooks.flatMap((book) => surnameKeys(book.author).map((surname) => `${titleKey(book.title)}|${surname}`)));

    // The candidates - not duplicating a Standard Ebooks book, or a more popular Gutenberg edition
    const candidates = [];
    const fullTitles = new Set();
    let seDuplicates = 0;
    let gutenbergDuplicates = 0;
    let referenceWorks = 0;
    for (const result of results) {
        if (isReferenceWork(result)) {
            referenceWorks++;
            continue;
        }
        const {title, subtitle} = gutenbergTitle(result.title);
        const author = result.authors.map((person) => gutenbergAuthor(person.name)).filter((name) => name !== '').join(', ');
        const surnames = surnameKeys(author);
        if (surnames.some((surname) => seWorks.has(`${titleKey(title)}|${surname}`))) {
            seDuplicates++;
            continue;
        }
        const fullTitle = `${normalizeText(`${title} ${subtitle}`)}|${normalizeText(author)}`;
        if (fullTitles.has(fullTitle)) {
            gutenbergDuplicates++;
            continue;
        }
        fullTitles.add(fullTitle);
        candidates.push({result, title, subtitle, author});
    }
    log(`Gutenberg: ${candidates.length} candidates (${seDuplicates} Standard Ebooks duplicates, ` +
        `${gutenbergDuplicates} duplicate editions, ${referenceWorks} reference works)`);

    // Resolve the texts - results are cached (GITenberg is a static mirror; unresolved books are retried after maxAge)
    const repositories = await gitenbergRepositories(options.maxAge);
    const maxListedId = Math.max(...[...repositories.keys()].map(Number));
    const textCacheName = 'gitenberg/texts.json';
    const textCache = readJSONCache(textCacheName);
    const searchCache = new Map(Object.entries(readJSONCache(searchCacheName)));
    const books = [];
    let unresolved = 0;
    let pending = 0;
    for (const [index, {result, title, subtitle, author}] of candidates.entries()) {
        const id = String(result.id);
        let cached = textCache[id];
        if (!cached && cacheOnly) {
            pending++;
            continue;
        }
        if (!cached || (cached.text === null && !cacheOnly && (Date.now() - cached.checked) > options.maxAge * 86400000)) {
            let repository = repositories.get(id) ?? null;
            if (repository === null && Number(id) <= maxListedId) {
                repository = await searchRepository(id, searchCache);
            }
            cached = {repository, 'text': repository === null ? null : await resolveText(id, repository), 'checked': Date.now()};
            textCache[id] = cached;
            writeFileAtomic(join(cacheDir, textCacheName), JSON.stringify(textCache, null, 1));
        }
        if ((index + 1) % 100 === 0) {
            log(`Gutenberg: ${index + 1} of ${candidates.length} texts resolved`);
        }
        if (cached.text === null) {
            unresolved++;
            continue;
        }
        books.push({
            'source': 'gutenberg',
            id,
            title,
            ...(subtitle === '' ? {} : {subtitle}),
            author,
            'description': gutenbergDescription(result.summaries?.[0] ?? ''),
            'subjects': gutenbergSubjects(result),
            'language': result.languages.includes('en') ? 'en' : result.languages[0],
            'cover': result.formats['image/jpeg'] || null,
            'rank': books.length + 1,
            'words': gutenbergWords(cached.text.bytes),
            'text': {'url': cached.text.url, 'format': 'gutenberg-text', 'encoding': cached.text.encoding}
        });
    }
    log(`Gutenberg: ${books.length} books (${unresolved} not in GITenberg${pending > 0 ? `, ${pending} not yet resolved` : ''})`);
    return books;
}


//
// The catalog
//


// The catalog JSON - one book per line, so changes diff well
function catalogJSON(catalog) {
    const {books, ...header} = catalog;
    const headerJSON = JSON.stringify(header, null, 4).replace(/\n\}$/, '');
    return `${headerJSON},\n    "books": [\n${books.map((book) => `        ${JSON.stringify(book)}`).join(',\n')}\n    ]\n}\n`;
}


// Build the catalog
async function main() {
    const options = parseArgs(process.argv.slice(2));
    ({cacheOnly} = options);
    const seBooks = await standardEbooks(options);
    const gutenberg = options.gutenberg > 0 ? await gutenbergBooks(options, seBooks) : [];
    const catalog = {
        'version': 1,
        'generated': new Date().toISOString(),
        sources,
        'books': [...seBooks, ...gutenberg]
    };
    const json = catalogJSON(catalog);
    JSON.parse(json);
    writeFileAtomic(options.output, json);
    log(`${options.output}: ${seBooks.length} Standard Ebooks and ${gutenberg.length} Gutenberg books, ${json.length} bytes`);
}


main().catch((error) => {
    process.stderr.write(`error: ${error.message}\n`);
    process.exitCode = 1;
});
