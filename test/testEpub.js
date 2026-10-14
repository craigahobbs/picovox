// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

import {convertEpub, isEpubTextEntry} from '../lib/epub.js';
import {JSDOM} from 'jsdom';
import {strict as assert} from 'node:assert';
import {openZip} from '../lib/zip.js';
import test from 'node:test';


// The EPUB parser uses the global DOMParser, as in browsers
globalThis.DOMParser = new JSDOM('').window.DOMParser;


// Build a zip reader of stored (uncompressed) entries from an object of name to text
function buildZipReader(files) {
    const encoder = new TextEncoder();
    const parts = [];
    const centrals = [];
    let offset = 0;
    for (const [name, text] of Object.entries(files)) {
        const nameBytes = encoder.encode(name);
        const data = encoder.encode(text);
        const local = new Uint8Array(30 + nameBytes.length + data.length);
        const localView = new DataView(local.buffer);
        localView.setUint32(0, 0x04034b50, true);
        localView.setUint32(18, data.length, true);
        localView.setUint32(22, data.length, true);
        localView.setUint16(26, nameBytes.length, true);
        local.set(nameBytes, 30);
        local.set(data, 30 + nameBytes.length);
        parts.push(local);
        const central = new Uint8Array(46 + nameBytes.length);
        const centralView = new DataView(central.buffer);
        centralView.setUint32(0, 0x02014b50, true);
        centralView.setUint32(20, data.length, true);
        centralView.setUint32(24, data.length, true);
        centralView.setUint16(28, nameBytes.length, true);
        centralView.setUint32(42, offset, true);
        central.set(nameBytes, 46);
        centrals.push(central);
        offset += local.length;
    }
    const directorySize = centrals.reduce((sum, central) => sum + central.length, 0);
    const end = new Uint8Array(22);
    const endView = new DataView(end.buffer);
    endView.setUint32(0, 0x06054b50, true);
    endView.setUint16(8, centrals.length, true);
    endView.setUint16(10, centrals.length, true);
    endView.setUint32(12, directorySize, true);
    endView.setUint32(16, offset, true);
    const allParts = [...parts, ...centrals, end];
    const bytes = new Uint8Array(allParts.reduce((sum, part) => sum + part.length, 0));
    let position = 0;
    for (const part of allParts) {
        bytes.set(part, position);
        position += part.length;
    }
    return openZip(bytes);
}


// META-INF/container.xml naming the package file
function containerXml(opfPath) {
    return `<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles><rootfile full-path="${opfPath}" media-type="application/oebps-package+xml"/></rootfiles>
</container>`;
}


// A package (OPF) file from metadata, manifest, and spine markup
function packageOpf(metadata, manifest, spine) {
    return `<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0" unique-identifier="uid">
    <metadata>${metadata}</metadata>
    <manifest>${manifest}</manifest>
    <spine>${spine}</spine>
</package>`;
}


// An XHTML document from body markup
function xhtml(body, bodyType = 'bodymatter') {
    return `<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>Test</title></head>
<body epub:type="${bodyType}">${body}</body>
</html>`;
}


// Convert a one-document EPUB with the given metadata
function convertBody(body, {metadata = '<dc:title>T</dc:title>', bodyType = 'bodymatter'} = {}) {
    return convertEpub(buildZipReader({
        'META-INF/container.xml': containerXml('content.opf'),
        'content.opf': packageOpf(
            metadata,
            '<item id="a" href="a.xhtml" media-type="application/xhtml+xml"/>',
            '<itemref idref="a"/>'
        ),
        'a.xhtml': xhtml(body, bodyType)
    }));
}


test('isEpubTextEntry', () => {
    assert.equal(isEpubTextEntry('epub/text/chapter-1.xhtml'), true);
    assert.equal(isEpubTextEntry('epub/content.opf'), true);
    assert.equal(isEpubTextEntry('META-INF/container.xml'), true);
    assert.equal(isEpubTextEntry('epub/toc.ncx'), true);
    assert.equal(isEpubTextEntry('index.HTML'), true);
    assert.equal(isEpubTextEntry('page.htm'), true);
    assert.equal(isEpubTextEntry('mimetype'), false);
    assert.equal(isEpubTextEntry('epub/images/cover.jpg'), false);
    assert.equal(isEpubTextEntry('epub/css/core.css'), false);
    assert.equal(isEpubTextEntry('epub/fonts/font.woff2'), false);
});


// A Standard Ebooks-like EPUB - metadata with refinements, a navigation document, front matter, parts and chapters
// in separate files, verse, a play, and back matter
const bookMetadata = `
    <dc:identifier id="uid">url:https://standardebooks.org/ebooks/test</dc:identifier>
    <dc:title id="title">Moby Dick</dc:title>
    <meta property="title-type" refines="#title">main</meta>
    <meta property="display-seq" refines="#title">1</meta>
    <dc:title id="subtitle">Or, The Whale</dc:title>
    <meta property="title-type" refines="#subtitle">subtitle</meta>
    <dc:creator id="author">Herman  Melville</dc:creator>
    <meta property="file-as" refines="#author">Melville, Herman</meta>
    <meta property="se:word-count">123</meta>`;

const bookManifest = `
    <item href="toc.xhtml" id="toc.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item href="text/titlepage.xhtml" id="titlepage.xhtml" media-type="application/xhtml+xml" properties="svg"/>
    <item href="text/preface.xhtml" id="preface.xhtml" media-type="application/xhtml+xml"/>
    <item href="text/part-1.xhtml" id="part-1.xhtml" media-type="application/xhtml+xml"/>
    <item href="text/chapter-1.xhtml" id="chapter-1.xhtml" media-type="application/xhtml+xml"/>
    <item href="./text/../text/chapter%202.xhtml" id="chapter-2.xhtml" media-type="application/xhtml+xml"/>
    <item href="text/play.xhtml" id="play.xhtml" media-type="application/xhtml+xml"/>
    <item href="text/endnotes.xhtml" id="endnotes.xhtml" media-type="application/xhtml+xml"/>
    <item href="text/colophon.xhtml" id="colophon.xhtml" media-type="application/xhtml+xml"/>
    <item href="css/core.css" id="core.css" media-type="text/css"/>
    <item href="images/cover.svg" id="cover.svg" media-type="image/svg+xml"/>`;

const bookSpine = `
    <itemref idref="titlepage.xhtml"/>
    <itemref idref="preface.xhtml"/>
    <itemref idref="part-1.xhtml"/>
    <itemref idref="chapter-1.xhtml"/>
    <itemref idref="chapter-2.xhtml"/>
    <itemref idref="play.xhtml"/>
    <itemref idref="core.css"/>
    <itemref idref="missing.xhtml"/>
    <itemref idref="endnotes.xhtml" linear="no"/>
    <itemref idref="colophon.xhtml"/>`;

const bookToc = xhtml(`
    <nav epub:type="landmarks"><ol><li><a href="text/titlepage.xhtml">Landmark Title</a></li></ol></nav>
    <nav id="toc" epub:type="toc">
        <h2 epub:type="title">Table of Contents</h2>
        <ol>
            <li><a href="text/titlepage.xhtml">Titlepage</a></li>
            <li><a href="text/preface.xhtml">Preface.</a></li>
            <li><a href="text/preface.xhtml">Preface Again</a></li>
            <li><a href="text/part-1.xhtml#part-1">Part I</a></li>
            <li><a href="text/chapter%202.xhtml#chapter-2">Chapter II</a></li>
            <li><a href="text/chapter%202.xhtml#interlude">The\u00A0Interlude</a></li>
            <li><a>No Link</a></li>
            <li><a href="text/play.xhtml"> </a></li>
        </ol>
    </nav>`, 'frontmatter');

const bookFiles = {
    'META-INF/container.xml': containerXml('epub/content.opf'),
    'epub/content.opf': packageOpf(bookMetadata, bookManifest, bookSpine),
    'epub/toc.xhtml': bookToc,
    'epub/text/titlepage.xhtml': xhtml(`
        <section id="titlepage" epub:type="titlepage"><h1 epub:type="title">Moby Dick</h1></section>`, 'frontmatter'),
    'epub/text/preface.xhtml': xhtml(`
        <section id="preface" epub:type="preface">
            <p>It was a <i>dark</i> night.<a href="endnotes.xhtml#note-1" id="noteref-1" epub:type="noteref">1</a></p>
            <!-- a comment -->
            <p hidden="hidden">Hidden text.</p>
            <p>Soft\u00ADhy\u200Bphen\u2060ated\uFEFF and\u2003spaced<a role="doc-noteref" href="#n">2</a>.</p>
            <figure><img src="../images/map.png" alt="A map"/><figcaption>A map.</figcaption></figure>
            <p><![CDATA[Character data.]]></p>
        </section>`, 'frontmatter'),
    'epub/text/part-1.xhtml': xhtml(`
        <section id="part-1" epub:type="part">
            <hgroup>
                <h2><span epub:type="se:label">Part</span> <span epub:type="z3998:ordinal">I</span></h2>
                <p epub:type="title">The Beginning</p>
            </hgroup>
        </section>`),
    'epub/text/chapter-1.xhtml': xhtml(`
        <section id="chapter-1" epub:type="chapter" data-parent="part-1">
            <h3 epub:type="z3998:ordinal">I</h3>
            <header><p>Extracts</p><p>Supplied by a sub-sub-librarian!</p></header>
            <p>Call me Ishmael.   Some years ago—never mind
                how long precisely—having little money.</p>
            <blockquote><p>A quotation.</p></blockquote>
            <div epub:type="z3998:poem">
                <p><span>Line one,</span><br/><span>Line two.</span></p>
            </div>
        </section>`),
    'epub/text/chapter 2.xhtml': xhtml(`
        <section id="chapter-2" epub:type="chapter" data-parent="no-such-part">
            <header><h2 epub:type="ordinal">II</h2><p epub:type="bridgehead">A bridgehead</p></header>
            <p>The second chapter.</p>
            <section id="interlude" epub:type="z3998:fiction">
                <p>An interlude without a heading.</p>
                <section epub:type="z3998:subchapter">
                    <h4 epub:type="z3998:ordinal">1</h4>
                    <p>A subsection.</p>
                </section>
            </section>
        </section>`),
    'epub/text/play.xhtml': xhtml(`
        <article id="act-1" epub:type="z3998:act">
            <h2 epub:type="ordinal z3998:roman">I</h2>
            <section id="scene-1" epub:type="z3998:scene">
                <h3><span epub:type="z3998:ordinal">1</span><span epub:type="title">A Room in the Castle</span></h3>
                <table>
                    <tbody>
                        <tr><td epub:type="z3998:persona">Hamlet</td><td>To be, or not to be.</td></tr>
                        <tr>
                            <td epub:type="z3998:persona">Ophelia</td>
                            <td><p>Good my lord,</p><p>How does your honour?</p></td>
                        </tr>
                        <tr><td epub:type="z3998:persona"><img src="x.png" alt=""/></td><td>Alone.</td></tr>
                        <tr><td><i epub:type="z3998:stage-direction">Enter Polonius</i></td><td>My lord.</td></tr>
                        <tr><td>First</td><td>Second.</td></tr>
                    </tbody>
                </table>
            </section>
        </article>`),
    'epub/text/endnotes.xhtml': xhtml(`
        <section id="endnotes" epub:type="endnotes"><ol><li id="note-1"><p>A note.</p></li></ol></section>`, 'backmatter'),
    'epub/text/colophon.xhtml': xhtml(`
        <section id="colophon" epub:type="colophon"><p>Standard Ebooks colophon.</p></section>`, 'backmatter'),
    'epub/css/core.css': 'body { margin: 0; }'
};


test('convertEpub', async () => {
    const result = await convertEpub(buildZipReader(bookFiles));
    assert.equal(result.title, 'Moby Dick');
    assert.equal(result.author, 'Herman Melville');
    assert.equal(result.text, `\
Moby Dick: Or, The Whale.

By Herman Melville.

Preface.

It was a dark night.

Softhyphenated and spaced.

Character data.

Part I. The Beginning.

Chapter I.

Extracts.

Supplied by a sub-sub-librarian!

Call me Ishmael. Some years ago—never mind how long precisely—having little money.

A quotation.

Line one,
Line two.

II.

A bridgehead.

The second chapter.

The Interlude.

An interlude without a heading.

Section 1.

A subsection.

I.

Scene 1. A Room in the Castle.

Hamlet: To be, or not to be.

Ophelia: Good my lord,
How does your honour?

Alone.

Enter Polonius, My lord.

First, Second.`);
    assert.deepEqual(result.chapters, [
        {'title': 'Preface', 'char': 48, 'level': 1},
        {'title': 'Part I. The Beginning', 'char': 125, 'level': 1},
        {'title': 'Chapter I', 'char': 149, 'level': 2},
        {'title': 'II', 'char': 325, 'level': 1},
        {'title': 'The Interlude', 'char': 366, 'level': 2},
        {'title': 'Section 1', 'char': 415, 'level': 2},
        {'title': 'I', 'char': 442, 'level': 1},
        {'title': 'Scene 1. A Room in the Castle', 'char': 446, 'level': 2}
    ]);

    // Each chapter is at its heading paragraph
    for (const chapter of result.chapters) {
        assert.equal(result.text.slice(chapter.char, chapter.char + chapter.title.length + 1), `${chapter.title}.`);
    }
});


test('convertEpub, missing package file', async () => {
    const zip = buildZipReader({'META-INF/container.xml': '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"/>'});
    await assert.rejects(convertEpub(zip), {'message': 'EPUB package file not found'});
});


test('convertEpub, invalid package file', async () => {
    const zip = buildZipReader({'META-INF/container.xml': containerXml('content.opf'), 'content.opf': '<package><metadata></package>'});
    await assert.rejects(convertEpub(zip), {'message': 'Invalid XML'});
});


test('convertEpub, missing spine document', async () => {
    const zip = buildZipReader({
        'META-INF/container.xml': containerXml('content.opf'),
        'content.opf': packageOpf('<dc:title>T</dc:title>', '<item id="a" href="a.xhtml" media-type="application/xhtml+xml"/>',
            '<itemref idref="a"/>')
    });
    await assert.rejects(convertEpub(zip), {'message': 'ZIP entry not found: a.xhtml'});
});


test('convertEpub, metadata', async () => {
    // No title-type refinements - the first title; an expanded title; three authors; an item without a media type
    const zip = buildZipReader({
        'META-INF/container.xml': containerXml('content.opf'),
        'content.opf': packageOpf(
            `<dc:title id="t1">What Is Man?</dc:title><dc:title id="t2">And Other Essays</dc:title>
             <meta property="title-type" refines="#t2">expanded</meta>
             <dc:creator>A. Author</dc:creator><dc:creator> </dc:creator>
             <dc:creator>B. Author</dc:creator><dc:creator>C. Author</dc:creator>`,
            '<item id="a" href="a.xhtml" media-type="application/xhtml+xml"/><item id="b" href="b.xhtml"/>',
            '<itemref idref="a"/><itemref idref="b"/><itemref/>'
        ),
        'a.xhtml': xhtml('<p>Text.</p>')
    });
    const result = await convertEpub(zip);
    assert.deepEqual(result, {
        'text': 'And Other Essays.\n\nBy A. Author, B. Author, and C. Author.\n\nText.',
        'chapters': [],
        'title': 'What Is Man?',
        'author': 'A. Author, B. Author, and C. Author'
    });

    // No title or author; two authors
    assert.deepEqual(await convertBody('<p>Text.</p>', {'metadata': ''}), {
        'text': '.\n\nText.',
        'chapters': [],
        'title': '',
        'author': ''
    });
    const twoAuthors = '<dc:title>Why?</dc:title><dc:creator>A</dc:creator><dc:creator>B</dc:creator>';
    assert.deepEqual(await convertBody('<p>Text.</p>', {'metadata': twoAuthors}), {
        'text': 'Why?\n\nBy A and B.\n\nText.',
        'chapters': [],
        'title': 'Why?',
        'author': 'A and B'
    });
});


test('convertEpub, navigation document', async () => {
    // A navigation document without a "toc" nav - the first nav; and a navigation item missing from the zip
    const files = (navBody) => ({
        'META-INF/container.xml': containerXml('content.opf'),
        'content.opf': packageOpf(
            '<dc:title>T</dc:title>',
            '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav scripted"/>' +
                '<item id="a" href="a.xhtml" media-type="application/xhtml+xml"/>',
            '<itemref idref="a"/>'
        ),
        ...(navBody !== null ? {'nav.xhtml': xhtml(navBody)} : {}),
        'a.xhtml': xhtml('<section id="s1"><p>One.</p></section><section id="s2"><p>Two.</p></section>')
    });
    const result = await convertEpub(buildZipReader(files('<nav><ol><li><a href="a.xhtml#s2">Second</a></li></ol></nav>')));
    assert.equal(result.text, 'T.\n\nOne.\n\nSecond.\n\nTwo.');
    assert.deepEqual(result.chapters, [{'title': 'Second', 'char': 10, 'level': 1}]);

    // No nav element
    const resultNoNav = await convertEpub(buildZipReader(files('<p>Nothing</p>')));
    assert.equal(resultNoNav.text, 'T.\n\nOne.\n\nTwo.');

    // The navigation document is missing
    const resultMissing = await convertEpub(buildZipReader(files(null)));
    assert.equal(resultMissing.text, 'T.\n\nOne.\n\nTwo.');
});


test('convertEpub, headings', async () => {
    const result = await convertBody(`
        <h1>A Body <b>Heading</b><a epub:type="noteref" href="#n">1</a><img src="x.png" alt=""/></h1>
        <section id="s1" epub:type="chapter">
            <h2 epub:type="z3998:ordinal">IV</h2>
            <p>Text.</p>
            <section id="s2" epub:type="z3998:letter">
                <h3><span epub:type="z3998:ordinal">II</span><br/><span epub:type="fulltitle">To My Sister</span></h3>
                <section id="s3">
                    <hgroup>
                        <h4 epub:type="z3998:ordinal">III</h4>
                        <p epub:type="subtitle">A Subtitle</p>
                        <p role="doc-subtitle">Another <i>Subtitle</i></p>
                    </hgroup>
                    <h4><img src="x.png" alt=""/></h4>
                    <h5>Plain <span epub:type="title">Title</span> and <div>more</div> text</h5>
                    <h6><span epub:type="se:label">Book</span><span epub:type="z3998:ordinal">X.</span></h6>
                    <h6><span epub:type="se:label">Book</span> the Last.</h6>
                </section>
            </section>
        </section>`);
    assert.equal(result.text, `\
T.

A Body Heading.

Chapter IV.

Text.

Letter II. To My Sister.

III.

A SubtitleAnother Subtitle.

Title Plain and more text.

Book X.

Book. the Last.`);
    assert.deepEqual(result.chapters.map((chapter) => [chapter.title, chapter.level]), [
        ['A Body Heading', 1],
        ['Chapter IV', 1],
        ['Letter II. To My Sister', 2],
        ['III', 2],
        ['Title Plain and more text', 2],
        ['Book X', 2],
        ['Book. the Last', 2]
    ]);
});


test('convertEpub, skipped body', async () => {
    const result = await convertBody('<p>The uncopyright.</p>', {'bodyType': 'backmatter uncopyright'});
    assert.equal(result.text, 'T.');
});


test('convertEpub, not well-formed XHTML', async () => {
    // Parsed as HTML - epub:type is a plain attribute
    const zip = buildZipReader({
        'META-INF/container.xml': containerXml('content.opf'),
        'content.opf': packageOpf('<dc:title>T</dc:title>', '<item id="a" href="a.xhtml" media-type="text/html"/>', '<itemref idref="a"/>'),
        'a.xhtml': `<html><body><section id="c1" epub:type="chapter"><h2 epub:type="z3998:ordinal">V</h2>
            <p>Non-well-formed&nbsp;text<br>and a line.</p><section epub:type="footnotes"><p>Note.</p></section></section></body></html>`
    });
    const result = await convertEpub(zip);
    assert.equal(result.text, 'T.\n\nChapter V.\n\nNon-well-formed text\nand a line.');
    assert.deepEqual(result.chapters, [{'title': 'Chapter V', 'char': 4, 'level': 1}]);
});


test('convertEpub, no body', async () => {
    const zip = buildZipReader({
        'META-INF/container.xml': containerXml('content.opf'),
        'content.opf': packageOpf('<dc:title>T</dc:title>', '<item id="a" href="a.xml" media-type="application/xhtml+xml"/>',
            '<itemref idref="a"/>'),
        'a.xml': '<?xml version="1.0"?><doc xmlns="http://www.w3.org/1999/xhtml"><p>Text without a body.</p></doc>'
    });
    const result = await convertEpub(zip);
    assert.equal(result.text, 'T.\n\nText without a body.');
});
