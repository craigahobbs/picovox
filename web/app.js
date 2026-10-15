// The audiobook player - a library (continue listening, your books, and finding books in the catalog of public-domain
// books - data/catalog.json, built offline by tools/catalog.js; each book's text is fetched from its source) and a
// player (play/pause, back and forward, with contents, bookmarks, and the narrator's voice and speed). The listener's
// place in every book is kept in browser storage, so reloading or leaving and coming back resumes where they were.

import {deleteCachedBook, getCachedBook, putCachedBook} from './bookcache.js';
import {findBook, listBooks, pageItems, searchBooks} from '../lib/catalog.js';
import {loadState, newId, saveState} from './store.js';
import {Engine} from './engine.js';
import {source as gutenberg} from './sources/gutenberg.js';
import {source as standardEbooks} from './sources/standardebooks.js';


// The book sources' loaders, and the catalog (loaded at startup)
const loaders = Object.fromEntries([standardEbooks, gutenberg].map((source) => [source.id, source]));
let catalog = null;
const catalogPromise = fetch(new URL('../data/catalog.json', import.meta.url)).then((response) => {
    if (!response.ok) {
        throw new Error(`The book catalog couldn't be loaded (${response.status}).`);
    }
    return response.json();
});


// The skip distance (s), the narrator speeds, and the narrator voices
const skipSeconds = 30;
const speeds = [0.8, 1, 1.2, 1.4, 1.6];
const voiceChoices = [['neural-male', 'Male'], ['neural-female', 'Female']];


// Elements
const element = (id) => document.getElementById(id);
const libraryView = element('libraryView');
const playerView = element('playerView');
const miniPlayer = element('miniPlayer');
const miniPlayerInner = element('miniPlayerInner');
const continueSection = element('continueSection');
const continueCard = element('continueCard');
const shelfSection = element('shelfSection');
const shelf = element('shelf');
const sourceTabs = element('sourceTabs');
const searchForm = element('searchForm');
const searchInput = element('search');
const sourceNote = element('sourceNote');
const results = element('results');
const resultsStatus = element('resultsStatus');
const moreButton = element('more');
const playerCover = element('playerCover');
const bookTitle = element('bookTitle');
const bookAuthor = element('bookAuthor');
const chapterTitle = element('chapterTitle');
const sentence = element('sentence');
const progressBar = element('progressBar');
const progressFill = element('progressFill');
const percentLabel = element('percent');
const remainingLabel = element('remaining');
const playPauseButton = element('playPause');
const playerStatus = element('playerStatus');
const sheet = element('sheet');
const sheetTitle = element('sheetTitle');
const sheetBody = element('sheetBody');


// State - the persisted state, the open book ({key, chapters, loading, progress, error}), the browse results, and the
// sheet being shown
const state = loadState();
// The narrator is a neural voice - listeners who chose a classic (rule) voice get its neural counterpart
const classicVoices = {'male': 'neural-male', 'female': 'neural-female'};
if (!voiceChoices.some(([voice]) => voice === state.settings.voice)) {
    state.settings.voice = classicVoices[state.settings.voice] || voiceChoices[0][0];
}
delete state.settings.neuralDefault;
state.sourceTab ||= 'all';
let current = null;
const browseState = {'query': '', 'page': 1, 'books': [], 'hasMore': false, 'total': 0, 'error': null};
let openSheetName = null;
let lastSave = 0;
// (?slowdown=N stretches synthesis N times - to reproduce a device slower than real time)
const slowdown = Number(new URLSearchParams(window.location.search).get('slowdown')) || 1;
const engine = new Engine({'onChange': onEngineChange, slowdown});


// The startup timeline overlay (?timing in the URL) - the engine's events since the last play, for diagnosing slow
// starts on a device. With &log=<url>, the timeline, page errors, and the user agent are also posted to <url>/log every
// two seconds (tools/devlog.js receives them).
const debugParameters = new URLSearchParams(window.location.search);
if (debugParameters.has('timing')) {
    const overlay = document.createElement('pre');
    overlay.className = 'timing-overlay';
    document.body.append(overlay);
    window.addEventListener('error', (event) => engine.mark(`page error: ${event.message} ${event.filename}:${event.lineno}`));
    window.addEventListener('unhandledrejection', (event) => engine.mark(`page rejection: ${event.reason?.stack || event.reason}`));
    const logUrl = debugParameters.get('log');
    let logged = '';
    setInterval(() => {
        const text = engine.timeline.map(({ms, event}) => `${String(ms).padStart(6)} ms  ${event}`).join('\n');
        overlay.textContent = text;
        if (logUrl && text !== logged) {
            logged = text;
            navigator.sendBeacon(`${logUrl}/log`, JSON.stringify({'userAgent': navigator.userAgent, 'href': window.location.href, text}));
        }
    }, 2000);
}


// Create an element with a class and text
function make(tag, className = '', text = '') {
    const created = document.createElement(tag);
    if (className !== '') {
        created.className = className;
    }
    if (text !== '') {
        created.textContent = text;
    }
    return created;
}


// An SVG icon (from the page's symbols)
function icon(name, className = 'icon') {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', className);
    svg.setAttribute('aria-hidden', 'true');
    const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
    use.setAttribute('href', `#i-${name}`);
    svg.append(use);
    return svg;
}


// A book's key - its source and id
function bookKey(book) {
    return `${book.source}:${book.id}`;
}


// A book's cover - its image, or a generated cover (a color from the title) if there is none or it fails to load
function coverElement(book, className = '') {
    const cover = make('div', `cover ${className}`.trim());
    const generated = () => {
        let hash = 0;
        for (const char of book.title || '?') {
            hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
        }
        const label = make('div', 'generated', book.title || '');
        label.style.background = `hsl(${hash % 360} 38% 34%)`;
        cover.replaceChildren(label);
    };
    if (book.cover) {
        const image = make('img');
        image.alt = '';
        image.loading = 'lazy';
        image.decoding = 'async';
        image.referrerPolicy = 'no-referrer';
        image.src = book.cover;
        image.onerror = generated;
        cover.append(image);
    } else {
        generated();
    }
    return cover;
}


// Format the time left
function formatRemaining(seconds) {
    if (!(seconds > 0)) {
        return '';
    }
    const minutes = Math.round(seconds / 60);
    if (minutes < 1) {
        return 'Less than a minute left';
    }
    if (minutes < 60) {
        return `${minutes} min left`;
    }
    return `${Math.floor(minutes / 60)} h ${minutes % 60} min left`;
}


// Save the persisted state - throttled unless forced
function persist(force = true) {
    const now = Date.now();
    if (force || now - lastSave > 3000) {
        lastSave = now;
        saveState(state);
    }
}


// The chapter at a character offset - the last chapter starting at or before it
function chapterAt(char) {
    if (current === null) {
        return null;
    }
    let found = null;
    for (const chapter of current.chapters) {
        if (chapter.char > char) {
            break;
        }
        found = chapter;
    }
    return found;
}


// Remember the place in the open book
function savePosition(force = false) {
    if (current === null || current.loading || current.error !== null || !engine.loaded) {
        return;
    }
    const engineState = engine.state();
    const entry = state.books[current.key];
    if (entry === undefined) {
        return;
    }
    const chapter = chapterAt(engineState.char);
    entry.char = engineState.sentenceChar;
    entry.percent = engineState.status === 'ended' ? 100 : engineState.percent;
    entry.remaining = engineState.status === 'ended' ? 0 : engineState.remainingSeconds;
    entry.chapter = chapter === null ? '' : chapter.title;
    entry.updated = Date.now();
    persist(force);
}


// Engine changes - a new sentence, play, pause, buffering, or the end
function onEngineChange(engineState) {
    savePosition(engineState.status !== 'playing');
    renderPlayer();
    renderMiniPlayer();
    updateMediaSession();
}


// Open a book - download it (or use the cached copy) and load it at the listener's place. To play, call from a user
// gesture (it unlocks audio).
async function openBook(book, {play = false, show = true} = {}) {
    const key = bookKey(book);
    const saved = state.books[key] || {};
    state.books[key] = {
        'char': 0, 'percent': 0, ...saved, 'source': book.source, 'id': book.id, 'title': book.title || saved.title || '',
        'author': book.author || saved.author || '', 'cover': book.cover || saved.cover || null, 'updated': Date.now()
    };
    state.current = key;
    persist();
    if (show) {
        showPlayer();
    }
    if (play) {
        engine.ensureAudio();
    }

    // Already open?
    if (current !== null && current.key === key && current.error === null) {
        if (play && !current.loading) {
            engine.play();
        }
        current.playWhenLoaded ||= play;
        renderAll();
        return;
    }
    engine.pause();
    current = {key, 'chapters': [], 'loading': true, 'progress': null, 'error': null, 'playWhenLoaded': play};
    renderAll();
    const opening = current;
    try {
        let prepared = await getCachedBook(key);
        if (prepared === null) {
            const loader = loaders[book.source];
            const entry = findBook(await catalogPromise, book.source, book.id);
            if (loader === undefined || entry === null) {
                throw new Error('This book is no longer in the catalog.');
            }
            prepared = await loader.load(entry, (loaded, total) => {
                opening.progress = total ? loaded / total : null;
                opening.loaded = loaded;
                renderPlayer();
            });
            const {text, chapters, title, author} = prepared;
            await putCachedBook({key, text, chapters, title, author});
        }
        if (current !== opening) {
            return;
        }
        opening.chapters = prepared.chapters || [];
        opening.loading = false;
        const entry = state.books[key];
        entry.title ||= prepared.title || '';
        entry.author ||= prepared.author || '';
        engine.load(prepared.text, state.settings, entry.char || 0);
        persist();
        renderAll();
        if (opening.playWhenLoaded) {
            engine.play();
        }
    } catch (error) {
        if (current === opening) {
            opening.loading = false;
            opening.error = error.message || String(error);
            renderAll();
        }
    }
}


// Play or pause the open book (from a user gesture)
function togglePlay() {
    if (current === null) {
        return;
    }
    if (current.error !== null) {
        // Retry a failed download
        const book = state.books[current.key];
        current = null;
        openBook(book, {'play': true, 'show': !playerView.hidden});
        return;
    }
    if (current.loading) {
        current.playWhenLoaded = !current.playWhenLoaded;
        engine.ensureAudio();
        renderAll();
        return;
    }
    engine.toggle();
}


// Skip back or forward
function skip(seconds) {
    if (current !== null && !current.loading && engine.loaded) {
        engine.skip(seconds);
        savePosition(true);
    }
}


// Show the library
function showLibrary() {
    closeSheet();
    state.view = 'library';
    persist();
    playerView.hidden = true;
    libraryView.hidden = false;
    renderLibrary();
    renderMiniPlayer();
}


// Show the player (with a history entry, so the back gesture returns to the library)
function showPlayer() {
    if (playerView.hidden && history.state?.view !== 'player') {
        history.pushState({'view': 'player'}, '');
    }
    state.view = 'player';
    persist();
    libraryView.hidden = true;
    miniPlayer.hidden = true;
    playerView.hidden = false;
    window.scrollTo(0, 0);
    renderPlayer();
}


// Render everything
function renderAll() {
    renderPlayer();
    renderMiniPlayer();
    if (!libraryView.hidden) {
        renderLibrary();
    }
}


// Render the player
function renderPlayer() {
    if (current === null || playerView.hidden) {
        return;
    }
    const book = state.books[current.key];
    if (playerCover.dataset.key !== current.key) {
        playerCover.dataset.key = current.key;
        playerCover.replaceChildren(...coverElement(book).childNodes);
        bookTitle.textContent = book.title;
        bookAuthor.textContent = book.author;
        document.title = `${book.title} - Audiobooks`;
    }
    const engineState = engine.state();
    const ready = !current.loading && current.error === null && engine.loaded;
    const percent = ready ? engineState.percent : (book.percent || 0);
    const chapter = ready ? chapterAt(engineState.char) : null;
    chapterTitle.textContent = chapter === null ? (book.chapter || '') : chapter.title;
    sentence.textContent = ready ? engineState.sentence : '';
    progressFill.style.width = `${Math.min(100, percent).toFixed(2)}%`;
    progressBar.setAttribute('aria-valuenow', Math.floor(percent));
    percentLabel.textContent = `${Math.floor(percent)}%`;
    remainingLabel.textContent = ready ? formatRemaining(engineState.remainingSeconds) : formatRemaining(book.remaining);

    // The play button and the status
    let status = '';
    let isError = false;
    let buttonIcon = 'play';
    let label = 'Play';
    if (current.error !== null) {
        status = `${current.error} Press play to try again.`;
        isError = true;
    } else if (current.loading) {
        status = current.progress === null ? 'Downloading the book...' : `Downloading the book... ${Math.round(100 * current.progress)}%`;
        if (current.playWhenLoaded) {
            buttonIcon = 'busy';
            label = 'Pause';
        }
    } else if (engine.error) {
        status = engine.error;
        isError = true;
    } else if (engineState.status === 'buffering') {
        buttonIcon = 'busy';
        label = 'Pause';
    } else if (engineState.status === 'playing') {
        buttonIcon = 'pause';
        label = 'Pause';
    } else if (engineState.status === 'loading') {
        status = 'Loading the voice...';
    } else if (engineState.status === 'ended') {
        status = 'The end. Press play to listen again.';
    }
    setPlayIcon(playPauseButton, buttonIcon, label);
    playerStatus.textContent = status;
    playerStatus.classList.toggle('error', isError);
}


// Set a play button's icon and label
function setPlayIcon(button, name, label) {
    if (button.dataset.icon !== name) {
        button.dataset.icon = name;
        button.replaceChildren(icon(name, name === 'busy' ? 'icon spinner' : 'icon'));
    }
    button.setAttribute('aria-label', label);
}


// The open book's play button state - 'pause' while playing (or about to), else 'play'
function isPlayingOrStarting() {
    if (current === null) {
        return false;
    }
    if (current.loading) {
        return current.playWhenLoaded;
    }
    const {status} = engine.state();
    return status === 'playing' || status === 'buffering';
}


// Render the mini player (in the library, while a book is open)
function renderMiniPlayer() {
    const visible = current !== null && !libraryView.hidden;
    miniPlayer.hidden = !visible;
    if (!visible) {
        return;
    }
    const book = state.books[current.key];
    if (miniPlayerInner.dataset.key !== current.key) {
        miniPlayerInner.dataset.key = current.key;
        const open = make('button', 'book-open');
        open.type = 'button';
        open.setAttribute('aria-label', `Open the player for ${book.title}`);
        const text = make('div', 'book-text');
        text.append(make('div', 'book-title', book.title), make('div', 'book-meta'));
        open.append(coverElement(book), text);
        open.addEventListener('click', showPlayer);
        const play = make('button', 'round-play');
        play.type = 'button';
        play.addEventListener('click', togglePlay);
        miniPlayerInner.replaceChildren(open, play);
    }
    const playing = isPlayingOrStarting();
    const busy = current.loading && current.playWhenLoaded || engine.state().status === 'buffering';
    setPlayIcon(miniPlayerInner.querySelector('.round-play'), busy ? 'busy' : (playing ? 'pause' : 'play'), playing ? 'Pause' : 'Play');
    const engineState = engine.loaded && !current.loading ? engine.state() : null;
    const chapter = engineState === null ? book.chapter : (chapterAt(engineState.char) || {}).title;
    const percent = engineState === null ? book.percent || 0 : engineState.percent;
    miniPlayerInner.querySelector('.book-meta').textContent = [chapter, `${Math.floor(percent)}%`].filter(Boolean).join(' · ');
}


// A book list row - open the book on click; with an optional remove button and progress
function bookRow(book, {progress = false, removable = false, showSource = false} = {}) {
    const row = make('li', 'book-row');
    const open = make('button', 'book-open');
    open.type = 'button';
    const text = make('div', 'book-text');
    text.append(make('div', 'book-title', book.title));
    const meta = [book.author];
    if (showSource && catalog !== null && catalog.sources[book.source]) {
        meta.push(catalog.sources[book.source].name);
    }
    if (progress) {
        meta.push(book.percent >= 99.5 ? 'Finished' : `${Math.floor(book.percent || 0)}%`);
    }
    text.append(make('div', 'book-meta', meta.filter(Boolean).join(' · ')));
    if (progress) {
        const bar = make('div', 'mini-progress');
        const fill = make('div');
        fill.style.width = `${Math.min(100, book.percent || 0)}%`;
        bar.append(fill);
        text.append(bar);
    }
    open.append(coverElement(book), text);
    open.addEventListener('click', () => openBook(book, {'play': false}));
    row.append(open);
    if (removable) {
        const remove = make('button', 'icon-button');
        remove.type = 'button';
        remove.setAttribute('aria-label', `Remove ${book.title} from your books`);
        remove.title = 'Remove';
        remove.append(icon('delete'));
        remove.addEventListener('click', () => removeBook(bookKey(book)));
        row.append(remove);
    }
    return row;
}


// Remove a book from your books - its place, bookmarks, and downloaded copy
function removeBook(key) {
    const book = state.books[key];
    if (book !== undefined) {
        openSheetName = 'confirm';
        confirmInSheet(`Remove "${book.title}" from your books? Your place and bookmarks in it are removed too.`, 'Remove',
            () => removeBookNow(key), closeSheet, 'Remove book');
        if (!sheet.open) {
            sheet.showModal();
        }
    }
}


// Remove a book now
function removeBookNow(key) {
    closeSheet();
    if (current !== null && current.key === key) {
        engine.pause();
        current = null;
        state.current = null;
    }
    delete state.books[key];
    delete state.bookmarks[key];
    deleteCachedBook(key);
    persist();
    renderLibrary();
    renderMiniPlayer();
}


// Render the library - continue listening, your books, and the source tabs
function renderLibrary() {
    const currentBook = state.current === null ? undefined : state.books[state.current];
    continueSection.hidden = currentBook === undefined;
    if (currentBook !== undefined) {
        const open = make('button', 'book-open');
        open.type = 'button';
        const text = make('div', 'book-text');
        const bar = make('div', 'mini-progress');
        const fill = make('div');
        fill.style.width = `${Math.min(100, currentBook.percent || 0)}%`;
        bar.append(fill);
        const where = [currentBook.chapter, `${Math.floor(currentBook.percent || 0)}%`, formatRemaining(currentBook.remaining)];
        text.append(
            make('div', 'book-title', currentBook.title),
            make('div', 'book-meta', currentBook.author),
            make('div', 'book-meta', where.filter(Boolean).join(' · ')),
            bar
        );
        open.append(coverElement(currentBook), text);
        open.addEventListener('click', showPlayer);
        if (current === null || current.key !== state.current) {
            open.addEventListener('click', () => openBook(currentBook, {'play': false}), {'once': true});
        }
        const play = make('button', 'round-play');
        play.type = 'button';
        play.id = 'continuePlay';
        play.addEventListener('click', () => {
            if (current !== null && current.key === state.current) {
                togglePlay();
            } else {
                openBook(currentBook, {'play': true, 'show': false});
            }
            renderLibrary();
        });
        continueCard.replaceChildren(open, play);
        renderContinuePlay();
    }
    const others = Object.entries(state.books).filter(([key]) => key !== state.current).
        sort(([, first], [, second]) => (second.updated || 0) - (first.updated || 0));
    shelfSection.hidden = others.length === 0;
    shelf.replaceChildren(...others.map(([, book]) => bookRow(book, {'progress': true, 'removable': true})));
}


// Render the continue-listening card's play button for the open book's state
function renderContinuePlay() {
    const play = document.getElementById('continuePlay');
    if (play !== null) {
        const isOpen = current !== null && current.key === state.current;
        const busy = isOpen && (current.loading && current.playWhenLoaded || engine.state().status === 'buffering');
        const playing = isOpen && isPlayingOrStarting();
        setPlayIcon(play, busy ? 'busy' : (playing ? 'pause' : 'play'), playing ? 'Pause' : 'Play');
    }
}


// Render the source tabs - all books, or one source's
function renderTabs() {
    if (catalog === null) {
        return;
    }
    // Short tab labels - they fit on a phone ('Project Gutenberg' -> 'Gutenberg')
    const tabs = [['all', 'All'], ...Object.entries(catalog.sources).map(([id, source]) => [id, source.name.replace(/^Project /, '')])];
    if (!tabs.some(([id]) => id === state.sourceTab)) {
        state.sourceTab = 'all';
    }
    sourceTabs.replaceChildren(...tabs.map(([id, name]) => {
        const tab = make('button', '', name);
        tab.type = 'button';
        tab.setAttribute('role', 'tab');
        tab.setAttribute('aria-selected', String(id === state.sourceTab));
        tab.addEventListener('click', () => {
            if (state.sourceTab !== id) {
                state.sourceTab = id;
                persist();
                renderTabs();
                browse(searchInput.value.trim(), 1);
            }
        });
        return tab;
    }));
    const source = catalog.sources[state.sourceTab];
    const sourceNames = Object.values(catalog.sources).map((other) => other.name).join(' and ');
    sourceNote.textContent = source === undefined ?
        `${catalog.books.length.toLocaleString()} public-domain books from ${sourceNames}.` : source.description;
}


// Browse the catalog - the selected source's books by popularity, or search results; page 1 replaces, later pages
// append
function browse(query, page) {
    browseState.query = query;
    browseState.page = page;
    if (catalog === null) {
        renderResults();
        return;
    }
    const books = query === '' ? listBooks(catalog, state.sourceTab) : searchBooks(catalog, query, state.sourceTab);
    const paged = pageItems(books, page, 24);
    browseState.books = page === 1 ? paged.items : [...browseState.books, ...paged.items];
    browseState.hasMore = paged.hasMore;
    browseState.total = paged.total;
    renderResults();
}


// Render the browse results
function renderResults() {
    results.replaceChildren(...browseState.books.map((book) => bookRow(book, {'showSource': state.sourceTab === 'all'})));
    let status = '';
    if (browseState.error !== null) {
        status = browseState.error;
    } else if (catalog === null) {
        status = 'Loading the catalog...';
    } else if (browseState.books.length === 0) {
        status = browseState.query === '' ? 'No books found.' : `No books match "${browseState.query}".`;
    } else if (browseState.query !== '') {
        const matches = browseState.total === 1 ? 'book matches' : 'books match';
        status = `${browseState.total.toLocaleString()} ${matches} "${browseState.query}".`;
    }
    resultsStatus.textContent = status;
    resultsStatus.classList.toggle('error', browseState.error !== null);
    moreButton.hidden = !browseState.hasMore;
}


// Sheets - contents, bookmarks, and settings
function openSheet(name) {
    if (current === null || current.loading || current.error !== null) {
        return;
    }
    openSheetName = name;
    renderSheet();
    if (!sheet.open) {
        sheet.showModal();
    }
}

function closeSheet() {
    openSheetName = null;
    if (sheet.open) {
        sheet.close();
    }
}

function renderSheet() {
    if (openSheetName === 'contents') {
        renderContents();
    } else if (openSheetName === 'bookmarks') {
        renderBookmarks();
    } else if (openSheetName === 'settings') {
        renderSettings();
    }
}


// The contents sheet - the chapters, the current one marked
function renderContents() {
    sheetTitle.textContent = 'Contents';
    if (current.chapters.length === 0) {
        sheetBody.replaceChildren(make('p', 'sheet-empty', 'This book has no chapter headings.'));
        return;
    }
    const currentChapter = chapterAt(engine.state().char);
    const list = make('ul', 'sheet-list');
    let currentButton = null;
    for (const chapter of current.chapters) {
        const item = make('li', 'sheet-item');
        const go = make('button', 'sheet-go');
        go.type = 'button';
        go.style.paddingLeft = `${10 + 16 * Math.max(0, (chapter.level || 1) - 1)}px`;
        go.append(make('span', 'label', chapter.title));
        if (chapter === currentChapter) {
            go.setAttribute('aria-current', 'true');
            currentButton = go;
        }
        go.addEventListener('click', () => {
            engine.seekChar(chapter.char);
            savePosition(true);
            closeSheet();
        });
        item.append(go);
        list.append(item);
    }
    sheetBody.replaceChildren(list);
    if (currentButton !== null) {
        requestAnimationFrame(() => currentButton.scrollIntoView({'block': 'center'}));
    }
}


// The bookmarks sheet - add one here, and go to, rename, or delete bookmarks
function renderBookmarks(editingId = null) {
    sheetTitle.textContent = 'Bookmarks';
    const {key} = current;
    const bookmarks = (state.bookmarks[key] || []).slice().sort((first, second) => first.char - second.char);
    const add = make('button', 'button primary add-bookmark');
    add.type = 'button';
    add.append(icon('bookmark'), document.createTextNode(' Bookmark this spot'));
    add.style.display = 'inline-flex';
    add.style.alignItems = 'center';
    add.style.justifyContent = 'center';
    add.style.gap = '8px';
    add.addEventListener('click', addBookmark);
    const nodes = [add];
    if (bookmarks.length === 0) {
        nodes.push(make('p', 'sheet-empty', 'No bookmarks yet. Bookmarks keep a spot to come back to - name them to remember why.'));
    }
    const list = make('ul', 'sheet-list');
    for (const bookmark of bookmarks) {
        const item = make('li', 'sheet-item');
        if (bookmark.id === editingId) {
            const input = make('input');
            input.type = 'text';
            input.value = bookmark.label;
            input.setAttribute('aria-label', 'Bookmark name');
            input.maxLength = 120;
            let done = false;
            const finish = (save) => {
                if (done) {
                    return;
                }
                done = true;
                if (save && input.value.trim() !== '') {
                    bookmark.label = input.value.trim();
                    persist();
                }
                renderBookmarks();
            };
            input.addEventListener('keydown', (event) => {
                if (event.key === 'Enter') {
                    event.preventDefault();
                    finish(true);
                } else if (event.key === 'Escape') {
                    event.preventDefault();
                    event.stopPropagation();
                    finish(false);
                }
            });
            input.addEventListener('blur', () => finish(true));
            item.append(input);
            requestAnimationFrame(() => {
                input.focus();
                input.select();
            });
        } else {
            const go = make('button', 'sheet-go');
            go.type = 'button';
            const date = new Date(bookmark.created).toLocaleDateString(undefined, {'month': 'short', 'day': 'numeric'});
            const detail = `${Math.floor(bookmark.percent)}% · ${date} · ${bookmark.snippet}`;
            go.append(make('span', 'label', bookmark.label), make('span', 'detail', detail));
            go.addEventListener('click', () => {
                engine.seekChar(bookmark.char);
                savePosition(true);
                closeSheet();
            });
            const rename = make('button', 'icon-button');
            rename.type = 'button';
            rename.setAttribute('aria-label', `Rename the bookmark "${bookmark.label}"`);
            rename.title = 'Rename';
            rename.append(icon('edit'));
            rename.addEventListener('click', () => renderBookmarks(bookmark.id));
            const remove = make('button', 'icon-button');
            remove.type = 'button';
            remove.setAttribute('aria-label', `Delete the bookmark "${bookmark.label}"`);
            remove.title = 'Delete';
            remove.append(icon('delete'));
            remove.addEventListener('click', () => {
                confirmInSheet(`Delete the bookmark "${bookmark.label}"?`, 'Delete', () => {
                    state.bookmarks[key] = state.bookmarks[key].filter((other) => other.id !== bookmark.id);
                    persist();
                }, renderBookmarks);
            });
            item.append(go, rename, remove);
        }
        list.append(item);
    }
    nodes.push(list);
    sheetBody.replaceChildren(...nodes);
}


// Ask for confirmation in the sheet - onConfirm runs on the action button, then onDone (also on cancel)
function confirmInSheet(message, actionLabel, onConfirm, onDone, title = sheetTitle.textContent) {
    sheetTitle.textContent = title;
    const cancel = make('button', 'button', 'Cancel');
    cancel.type = 'button';
    cancel.addEventListener('click', onDone);
    const action = make('button', 'button primary', actionLabel);
    action.type = 'button';
    action.addEventListener('click', () => {
        onConfirm();
        onDone();
    });
    const buttons = make('div', 'confirm-buttons');
    buttons.append(cancel, action);
    sheetBody.replaceChildren(make('p', 'confirm-message', message), buttons);
    action.focus();
}


// Bookmark the current sentence
function addBookmark() {
    const engineState = engine.state();
    const chapter = chapterAt(engineState.char);
    const snippet = engineState.sentence.length > 100 ? `${engineState.sentence.slice(0, 97)}...` : engineState.sentence;
    const bookmarks = state.bookmarks[current.key] || [];
    if (bookmarks.some((bookmark) => bookmark.char === engineState.sentenceChar)) {
        return;
    }
    const bookmark = {
        'id': newId(), 'char': engineState.sentenceChar, 'percent': engineState.percent, snippet,
        'label': chapter === null ? snippet : chapter.title, 'created': Date.now()
    };
    state.bookmarks[current.key] = [...bookmarks, bookmark];
    persist();
    renderBookmarks(bookmark.id);
}


// The settings sheet - the narrator's voice and speed
function renderSettings() {
    sheetTitle.textContent = 'Narrator';
    const choice = (title, options, selected, onSelect) => {
        const section = make('section', 'setting');
        section.append(make('h3', '', title));
        const tabs = make('div', 'tabs');
        tabs.setAttribute('role', 'radiogroup');
        tabs.setAttribute('aria-label', title);
        for (const [value, label] of options) {
            const button = make('button', '', label);
            button.type = 'button';
            button.setAttribute('role', 'radio');
            button.setAttribute('aria-selected', String(value === selected));
            button.setAttribute('aria-checked', String(value === selected));
            button.addEventListener('click', () => onSelect(value));
            tabs.append(button);
        }
        section.append(tabs);
        return section;
    };
    const apply = (settings) => {
        state.settings = {...state.settings, ...settings};
        persist();
        engine.setVoice(state.settings);
        renderSettings();
    };
    const note = make('p', 'muted', 'The voices are synthesized in your browser - no recordings. A small neural network, ' +
        'trained on 53 hours of 237 American audiobook readers, drives a model of the human voice.');
    sheetBody.replaceChildren(
        choice('Voice', voiceChoices, state.settings.voice, (voice) => apply({voice})),
        choice('Speed', speeds.map((speed) => [speed, `${speed}×`]), state.settings.rate, (rate) => apply({rate})),
        note
    );
}


// The lock screen and media keys
let mediaHandlersSet = false;
let mediaKey = '';
function updateMediaSession() {
    if (!('mediaSession' in navigator) || current === null) {
        return;
    }
    if (!mediaHandlersSet) {
        mediaHandlersSet = true;
        const handlers = {
            'play': () => engine.play(), 'pause': () => engine.pause(), 'seekbackward': () => skip(-skipSeconds),
            'seekforward': () => skip(skipSeconds), 'previoustrack': () => skip(-skipSeconds), 'nexttrack': () => skip(skipSeconds)
        };
        for (const [action, handler] of Object.entries(handlers)) {
            try {
                navigator.mediaSession.setActionHandler(action, handler);
            } catch {
                // An unsupported action
            }
        }
    }
    const book = state.books[current.key];
    const chapter = engine.loaded && !current.loading ? chapterAt(engine.state().char) : null;
    const key = `${current.key}|${chapter === null ? '' : chapter.title}`;
    if (key !== mediaKey && typeof MediaMetadata !== 'undefined') {
        mediaKey = key;
        navigator.mediaSession.metadata = new MediaMetadata({
            'title': chapter === null ? book.title : chapter.title, 'artist': book.author,
            'album': book.title, 'artwork': book.cover ? [{'src': book.cover, 'sizes': '512x768', 'type': 'image/jpeg'}] : []
        });
    }
    navigator.mediaSession.playbackState = isPlayingOrStarting() ? 'playing' : 'paused';
}


// Wire up the controls
element('toLibrary').addEventListener('click', () => {
    if (history.state?.view === 'player') {
        history.back();
    } else {
        showLibrary();
    }
});
window.addEventListener('popstate', () => {
    if (!playerView.hidden) {
        showLibrary();
    }
});
playPauseButton.addEventListener('click', togglePlay);
element('skipBack').addEventListener('click', () => skip(-skipSeconds));
element('skipForward').addEventListener('click', () => skip(skipSeconds));
element('contentsButton').addEventListener('click', () => openSheet('contents'));
element('bookmarksButton').addEventListener('click', () => openSheet('bookmarks'));
element('settingsButton').addEventListener('click', () => openSheet('settings'));
element('sheetClose').addEventListener('click', closeSheet);
sheet.addEventListener('close', () => {
    openSheetName = null;
});
sheet.addEventListener('click', (event) => {
    // A click on the backdrop closes the sheet
    if (event.target === sheet) {
        closeSheet();
    }
});
searchForm.addEventListener('submit', (event) => {
    event.preventDefault();
    searchInput.blur();
    browse(searchInput.value.trim(), 1);
});
let searchTimer = null;
searchInput.addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => browse(searchInput.value.trim(), 1), 150);
});
moreButton.addEventListener('click', () => browse(browseState.query, browseState.page + 1));
document.addEventListener('keydown', (event) => {
    if (sheet.open || event.target.closest('input, textarea, button[role="tab"]') || event.metaKey || event.ctrlKey || event.altKey) {
        return;
    }
    if (event.key === ' ' && current !== null && event.target === document.body) {
        event.preventDefault();
        togglePlay();
    } else if (event.key === 'ArrowLeft' && !playerView.hidden) {
        skip(-skipSeconds);
    } else if (event.key === 'ArrowRight' && !playerView.hidden) {
        skip(skipSeconds);
    }
});

// Keep the place when the page is hidden or closed
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') {
        savePosition(true);
    }
});
window.addEventListener('pagehide', () => savePosition(true));

// Refresh the player and mini player, and remember the place, as playback moves
setInterval(() => {
    renderPlayer();
    if (!miniPlayer.hidden) {
        renderMiniPlayer();
        renderContinuePlay();
    }
    if (current !== null && isPlayingOrStarting()) {
        savePosition(false);
    }
}, 250);


// Start - the library, or the player where the listener left off (paused - audio needs a tap to start)
browse('', 1);
catalogPromise.then((loaded) => {
    catalog = loaded;
    renderTabs();
    browse(searchInput.value.trim(), 1);
}, (error) => {
    browseState.error = error.message;
    renderResults();
});
const resumeBook = state.current === null ? undefined : state.books[state.current];
if (resumeBook !== undefined && state.view === 'player') {
    history.replaceState({'view': 'library'}, '');
    openBook(resumeBook, {'play': false, 'show': true});
} else {
    showLibrary();
    if (resumeBook !== undefined) {
        openBook(resumeBook, {'play': false, 'show': false});
    }
}
