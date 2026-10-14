// EPUB to text for reading aloud - the package (container.xml, then the OPF's metadata, manifest, and spine), and the
// spine's XHTML converted to paragraphs and chapters using the semantic markup (epub:type) that every Standard Ebooks
// production carries
//
// The text is paragraphs separated by blank lines, verse lines separated by single newlines: a title paragraph, a
// "By <author>." paragraph, then the book. Headings are paragraphs ending in a period, and each is a chapter,
// {title, char, level} - the heading without its period, its character offset in the text, and its level (1 or 2).
// Title pages, publishing information, tables of contents, notes, indexes, images, and note references are skipped.
//
// XML is parsed with the global DOMParser - browsers have one; in Node, install one (e.g. jsdom's).

import {readZipText} from './zip.js';


// Is the ZIP entry needed for an EPUB's text - not images, styles, fonts, or other media?
export function isEpubTextEntry(name) {
    return /\.(xhtml|html|htm|xml|opf|ncx)$/i.test(name);
}


// Namespaces
const nsContainer = 'urn:oasis:names:tc:opendocument:xmlns:container';
const nsOpf = 'http://www.idpf.org/2007/opf';
const nsDc = 'http://purl.org/dc/elements/1.1/';
const nsEpub = 'http://www.idpf.org/2007/ops';


// Convert an EPUB (a zip reader, from lib/zip.js) to {text, chapters, title, author}
export async function convertEpub(zip) {
    // META-INF/container.xml names the package (OPF) file
    const container = parseXml(await readZipText(zip, 'META-INF/container.xml'));
    const opfPath = container.getElementsByTagNameNS(nsContainer, 'rootfile')[0]?.getAttribute('full-path');
    if (!opfPath) {
        throw new Error('EPUB package file not found');
    }
    const opf = parseXml(await readZipText(zip, opfPath));
    const {title, fullTitle, author} = readMetadata(opf);

    // The spine's documents, in reading order
    const manifest = new Map();
    for (const item of opf.getElementsByTagNameNS(nsOpf, 'item')) {
        manifest.set(item.getAttribute('id'), item);
    }
    const spinePaths = [];
    for (const itemref of opf.getElementsByTagNameNS(nsOpf, 'itemref')) {
        const item = manifest.get(itemref.getAttribute('idref'));
        if (item && itemref.getAttribute('linear') !== 'no' && /html/.test(item.getAttribute('media-type') ?? '')) {
            spinePaths.push(resolvePath(opfPath, item.getAttribute('href')));
        }
    }
    const documents = await Promise.all(spinePaths.map((path) => readZipText(zip, path)));

    // The table of contents titles, for sections without a heading (e.g. a preface)
    const navItem = [...manifest.values()].find((item) => /(^|\s)nav(\s|$)/.test(item.getAttribute('properties') ?? ''));
    const navPath = navItem ? resolvePath(opfPath, navItem.getAttribute('href')) : null;
    const tocTitles = navPath && zip.entries.has(navPath)
        ? readTocTitles(parseXml(await readZipText(zip, navPath), true), navPath)
        : new Map();

    // Convert - title and author paragraphs, then each spine document
    const builder = createTextBuilder();
    builder.paragraph(sentence(fullTitle));
    if (author !== '') {
        builder.paragraph(sentence(`By ${author}`));
    }
    const sectionDepths = new Map();
    for (const [ix, documentText] of documents.entries()) {
        convertDocument(parseXml(documentText, true), spinePaths[ix], builder, sectionDepths, tocTitles);
    }
    return {'text': builder.text(), 'chapters': builder.chapters, title, author};
}


// Read the table of contents (the EPUB navigation document) - returns a Map of "path" or "path#id" to title
function readTocTitles(nav, navPath) {
    const tocTitles = new Map();
    const navs = [...nav.getElementsByTagName('nav')];
    const toc = navs.find((element) => epubTypes(element).includes('toc')) ?? navs[0];
    for (const link of toc?.getElementsByTagName('a') ?? []) {
        const href = link.getAttribute('href') ?? '';
        const hashIndex = href.indexOf('#');
        const path = resolvePath(navPath, href);
        const key = hashIndex === -1 ? path : `${path}${href.slice(hashIndex)}`;
        const title = collapseSpace(cleanText(link.textContent));
        if (title !== '' && !tocTitles.has(key)) {
            tocTitles.set(key, title);
        }
    }
    return tocTitles;
}


// Read the title and author from the OPF metadata - returns {title, fullTitle, author}
function readMetadata(opf) {
    // Refinements - "#id" to {property: value}
    const refines = new Map();
    for (const meta of opf.getElementsByTagNameNS(nsOpf, 'meta')) {
        const refinesId = meta.getAttribute('refines');
        if (refinesId) {
            if (!refines.has(refinesId)) {
                refines.set(refinesId, {});
            }
            refines.get(refinesId)[meta.getAttribute('property')] = collapseSpace(meta.textContent);
        }
    }
    const titleType = (element) => refines.get(`#${element.getAttribute('id')}`)?.['title-type'] ?? '';

    // Titles - the main title, and the expanded title (or main title with subtitle) for the title paragraph
    const titles = [...opf.getElementsByTagNameNS(nsDc, 'title')];
    const titleText = (type) => collapseSpace(titles.find((element) => titleType(element) === type)?.textContent ?? '');
    const title = titleText('main') || collapseSpace(titles[0]?.textContent ?? '');
    const subtitle = titleText('subtitle');
    const fullTitle = titleText('expanded') || (subtitle !== '' ? `${title}: ${subtitle}` : title);

    // Authors
    const authors = [...opf.getElementsByTagNameNS(nsDc, 'creator')].map((element) => collapseSpace(element.textContent));
    return {title, fullTitle, 'author': joinNames(authors.filter((name) => name !== ''))};
}


// Resolve a manifest href relative to the OPF file's directory
function resolvePath(basePath, href) {
    const parts = basePath.split('/').slice(0, -1);
    for (const part of decodeURIComponent(href.split('#')[0]).split('/')) {
        if (part === '..') {
            parts.pop();
        } else if (part !== '.' && part !== '') {
            parts.push(part);
        }
    }
    return parts.join('/');
}


// The text builder - paragraphs joined by blank lines, and the chapter list
function createTextBuilder() {
    const paragraphs = [];
    const chapters = [];
    let length = 0;

    // Add a paragraph - returns its character offset
    const paragraph = (text) => {
        const char = length === 0 ? 0 : length + 2;
        length = char + text.length;
        paragraphs.push(text);
        return char;
    };
    return {
        paragraph,
        chapters,

        // Add a heading paragraph and its chapter
        heading(title, level) {
            chapters.push({title, 'char': paragraph(sentence(title)), level});
        },

        // The text
        text() {
            return paragraphs.join('\n\n');
        }
    };
}


// Sections that aren't read - title pages, publishing information, tables of contents, notes, and indexes
const skipTypes = new Set([
    'titlepage', 'halftitlepage', 'imprint', 'colophon', 'copyright-page', 'uncopyright', 'toc', 'loi', 'lot',
    'landmarks', 'endnotes', 'footnotes', 'rearnotes', 'footnote', 'endnote', 'index', 'glossary', 'bibliography',
    'z3998:frontispiece', 'cover', 'seriespage', 'errata', 'se:transcriber-note'
]);

// Elements whose content isn't read
const skipElements = new Set([
    'head', 'img', 'svg', 'math', 'figure', 'picture', 'audio', 'video', 'object', 'iframe', 'script', 'style',
    'template', 'noscript', 'canvas', 'map', 'button', 'input', 'select', 'textarea'
]);

// Block elements - each starts and ends a paragraph
const blockElements = new Set([
    'p', 'div', 'blockquote', 'header', 'footer', 'aside', 'nav', 'main', 'address', 'pre', 'ol', 'ul', 'li', 'dl',
    'dt', 'dd', 'table', 'thead', 'tbody', 'tfoot', 'caption', 'figcaption', 'hr', 'body', 'html', 'details',
    'summary', 'center'
]);

// Section labels for headings that are only an ordinal, by section epub:type - "I" in a chapter is "Chapter I"
const sectionLabels = [
    ['volume', 'Volume'], ['part', 'Part'], ['division', 'Part'], ['z3998:book', 'Book'], ['book', 'Book'],
    ['chapter', 'Chapter'], ['z3998:act', 'Act'], ['z3998:scene', 'Scene'], ['z3998:canto', 'Canto'],
    ['z3998:letter', 'Letter'], ['z3998:poem', 'Poem'], ['z3998:song', 'Song'], ['z3998:hymn', 'Hymn'],
    ['z3998:subchapter', 'Section']
];


// Convert one spine document, adding its paragraphs and headings to the builder
//
// sectionDepths maps section ids to their nesting depth, so a section in a later file that names its parent
// with data-parent (e.g. a chapter in a part) is nested under it.
function convertDocument(doc, path, builder, sectionDepths, tocTitles) {
    let buffer = '';
    let headerDepth = 0;
    let isFirstSection = true;
    let cellStart = null;

    // End the current paragraph
    const flush = () => {
        const lines = cleanText(buffer).split('\n').map(collapseSpace).filter((line) => line !== '');
        buffer = '';
        if (lines.length !== 0) {
            // A paragraph in a section's header (e.g. "Extracts") ends with a period, like a heading
            const text = lines.join('\n');
            builder.paragraph(headerDepth !== 0 ? sentence(text) : text);
        }
    };

    // Walk an element's children - depth is the section nesting depth, types the nearest section's epub:type
    const walk = (element, depth, types) => {
        for (const node of element.childNodes) {
            // Text
            if (node.nodeType === 3 || node.nodeType === 4) {
                buffer += node.data.replace(/[ \t\r\n]+/g, ' ');
                continue;
            }
            if (node.nodeType !== 1) {
                continue;
            }

            // Skipped elements, sections, and note references
            const name = node.localName.toLowerCase();
            const nodeTypes = epubTypes(node);
            if (skipElements.has(name) || isNoteReference(node, nodeTypes) || node.hasAttribute('hidden')) {
                continue;
            }
            if (nodeTypes.some((type) => skipTypes.has(type))) {
                flush();
                continue;
            }

            // Line breaks (verse lines)
            if (name === 'br') {
                buffer += '\n';
                continue;
            }

            // Headings
            if (/^h[1-6]$/.test(name) || name === 'hgroup') {
                flush();
                const {heading, subtitle} = readHeading(node, types);
                if (heading !== '') {
                    builder.heading(heading, Math.min(Math.max(depth, 1), 2));
                }
                if (subtitle !== '') {
                    builder.paragraph(sentence(subtitle));
                }
                continue;
            }

            // Sections - nested one deeper than their parent section
            if (name === 'section' || name === 'article') {
                flush();
                const parentId = node.getAttribute('data-parent');
                const sectionDepth = parentId && sectionDepths.has(parentId) ? sectionDepths.get(parentId) + 1 : depth + 1;
                if (node.id) {
                    sectionDepths.set(node.id, sectionDepth);
                }

                // A section without a heading takes its heading from the table of contents
                const tocTitle = tocTitles.get(`${path}#${node.id}`) ?? (isFirstSection ? tocTitles.get(path) : undefined);
                isFirstSection = false;
                if (tocTitle && !hasHeading(node)) {
                    builder.heading(tocTitle.replace(/\.$/, ''), Math.min(sectionDepth, 2));
                }
                walk(node, sectionDepth, nodeTypes);
                flush();
                continue;
            }

            // Table rows are paragraphs - in plays, "Persona: speech"
            if (name === 'tr') {
                flush();
                for (const cell of node.children) {
                    const isPersona = epubTypes(cell).includes('z3998:persona');
                    const before = buffer.trim();
                    if (before !== '') {
                        buffer = /[.,;:!?—]$/.test(before) ? `${before} ` : `${before}, `;
                    }
                    cellStart = buffer.length;
                    walk(cell, depth, types);
                    cellStart = null;
                    if (isPersona && buffer.trim() !== '') {
                        buffer = `${buffer.trim()}: `;
                    }
                }
                flush();
                continue;
            }

            // Blocks in a table cell (verse speeches in plays) are lines of the row's paragraph
            if (blockElements.has(name) && cellStart !== null) {
                if (buffer.slice(cellStart).trim() !== '') {
                    buffer += '\n';
                }
                walk(node, depth, types);
                continue;
            }

            // Blocks - a header's paragraphs end with a period
            if (blockElements.has(name)) {
                flush();
                headerDepth += name === 'header' ? 1 : 0;
                walk(node, depth, types);
                flush();
                headerDepth -= name === 'header' ? 1 : 0;
                continue;
            }

            // Inline
            walk(node, depth, types);
        }
    };

    // Convert the body, unless it's a skipped section (e.g. <body epub:type="backmatter z3998:colophon">)
    const body = doc.getElementsByTagName('body')[0] ?? doc.documentElement;
    if (!epubTypes(body).some((type) => skipTypes.has(type))) {
        walk(body, 0, []);
        flush();
    }
}


// Does a section have a heading - an h1-h6 or hgroup child, or one in its header?
function hasHeading(section) {
    return [...section.children].some((child) => {
        const name = child.localName.toLowerCase();
        return /^(h[1-6]|hgroup)$/.test(name) ||
            (name === 'header' && [...child.children].some((headerChild) => /^(h[1-6]|hgroup)$/i.test(headerChild.localName)));
    });
}


// Read a heading (h1-h6 or hgroup) - returns {heading, subtitle}
//
// Standard Ebooks headings mark their parts - "<span epub:type="se:label">Book</span> <span
// epub:type="z3998:ordinal">I</span>", or an hgroup of an ordinal heading and a title paragraph.
function readHeading(element, sectionTypes) {
    const parts = {'label': '', 'ordinal': '', 'title': '', 'subtitle': '', 'other': ''};

    // Collect the heading's text by part
    const collect = (node, part) => {
        for (const child of node.childNodes) {
            if (child.nodeType === 3 || child.nodeType === 4) {
                parts[part] += child.data;
            } else if (child.nodeType === 1) {
                const childTypes = epubTypes(child);
                const name = child.localName.toLowerCase();
                if (skipElements.has(name) || isNoteReference(child, childTypes)) {
                    continue;
                }
                if (name === 'br') {
                    parts[part] += ' ';
                    continue;
                }
                collect(child, headingPart(child, childTypes) ?? part);
                if (/^(h[1-6]|p|div)$/.test(name)) {
                    parts[part] += ' ';
                }
            }
        }
    };
    collect(element, headingPart(element, epubTypes(element)) ?? 'other');
    for (const part of Object.keys(parts)) {
        parts[part] = collapseSpace(cleanText(parts[part]));
    }

    // An ordinal alone takes its label from the section - "I" in a chapter is "Chapter I"
    let {label} = parts;
    if (parts.ordinal !== '' && label === '') {
        label = sectionLabels.find(([type]) => sectionTypes.includes(type))?.[1] ?? '';
    }
    const lead = [label, parts.ordinal].filter((text) => text !== '').join(' ');
    const name = [parts.title, parts.other].filter((text) => text !== '').join(' ');
    const heading = lead !== '' && name !== '' ? `${sentence(lead)} ${name}` : lead || name;
    return {'heading': heading.replace(/\.$/, ''), 'subtitle': parts.subtitle};
}


// The part of a heading an element marks - label, ordinal, title, subtitle, or null
function headingPart(element, types) {
    if (types.includes('se:label')) {
        return 'label';
    }
    if (types.includes('z3998:ordinal')) {
        return 'ordinal';
    }
    if (types.includes('subtitle') || element.getAttribute('role') === 'doc-subtitle') {
        return 'subtitle';
    }
    if (types.includes('title') || types.includes('fulltitle')) {
        return 'title';
    }
    return null;
}


// An element's epub:type tokens
function epubTypes(element) {
    const value = element.getAttributeNS(nsEpub, 'type') ?? element.getAttribute('epub:type') ?? '';
    return value.split(/\s+/).filter((type) => type !== '');
}


// Is the element a note reference (an endnote or footnote marker)?
function isNoteReference(element, types) {
    return types.includes('noteref') || element.getAttribute('role') === 'doc-noteref';
}


//
// Text utilities
//


// Parse XML (an EPUB's container, OPF, or XHTML) - XHTML that isn't well-formed is parsed as HTML
function parseXml(text, isXhtml = false) {
    const parser = new globalThis.DOMParser();
    const doc = parser.parseFromString(text, isXhtml ? 'application/xhtml+xml' : 'application/xml');
    if (doc.getElementsByTagName('parsererror').length !== 0) {
        if (isXhtml) {
            return parser.parseFromString(text, 'text/html');
        }
        throw new Error('Invalid XML');
    }
    return doc;
}


// Remove invisible characters (word joiners, soft hyphens) and make unusual spaces plain spaces
function cleanText(text) {
    return text.replace(/[\u00AD\u200B\u2060\uFEFF]/g, '').replace(/[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]/g, ' ');
}


// Collapse runs of spaces and trim
function collapseSpace(text) {
    return text.replace(/\s+/g, ' ').trim();
}


// End text with a period, unless it already ends a sentence ("What Is Man?")
function sentence(text) {
    return /[.!?…]['"’”)\]]*$/.test(text) ? text : `${text}.`;
}


// Join names - "A", "A and B", "A, B, and C"
function joinNames(names) {
    if (names.length <= 2) {
        return names.join(' and ');
    }
    return `${names.slice(0, -1).join(', ')}, and ${names.at(-1)}`;
}
