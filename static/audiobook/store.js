// The listener's state in browser storage - the current book and view, the narrator settings, each started book's
// details and position, and each book's bookmarks. Storage may be unavailable (private browsing), so every access is
// guarded and the app works without it.


// The storage key and the default state
const storageKey = 'gutenberg-audiobooks-v1';
const defaultState = {
    'current': null, 'view': 'library', 'settings': {'voice': 'neural-male', 'rate': 1}, 'books': {}, 'bookmarks': {}
};


// Load the state
export function loadState() {
    try {
        const saved = JSON.parse(localStorage.getItem(storageKey));
        if (saved !== null && typeof saved === 'object') {
            return {
                ...defaultState, ...saved, 'settings': {...defaultState.settings, ...saved.settings},
                'books': saved.books || {}, 'bookmarks': saved.bookmarks || {}
            };
        }
    } catch {
        // Unavailable or corrupt storage - start fresh
    }
    return structuredClone(defaultState);
}


// Save the state - returns false if storage is unavailable
export function saveState(state) {
    try {
        localStorage.setItem(storageKey, JSON.stringify(state));
        return true;
    } catch {
        return false;
    }
}


// A short unique id
export function newId() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}
