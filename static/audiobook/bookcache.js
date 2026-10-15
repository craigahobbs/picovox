// The downloaded-book cache in IndexedDB - each prepared book (text and chapters) by its key, so reopening a book (or
// reloading the page) doesn't download it again, and books keep working offline. IndexedDB may be unavailable (private
// browsing), in which case the cache is simply empty.


// Open the database - resolves to null if IndexedDB is unavailable
let databasePromise = null;
function openDatabase() {
    if (databasePromise === null) {
        databasePromise = new Promise((resolve) => {
            try {
                const request = indexedDB.open('audiobooks', 1);
                request.onupgradeneeded = () => request.result.createObjectStore('books', {'keyPath': 'key'});
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => resolve(null);
                request.onblocked = () => resolve(null);
            } catch {
                resolve(null);
            }
        });
    }
    return databasePromise;
}


// Run a request on the books store - resolves to its result, or null on failure
async function withStore(mode, makeRequest) {
    const database = await openDatabase();
    if (database === null) {
        return null;
    }
    return new Promise((resolve) => {
        try {
            const request = makeRequest(database.transaction('books', mode).objectStore('books'));
            request.onsuccess = () => resolve(request.result === undefined ? null : request.result);
            request.onerror = () => resolve(null);
        } catch {
            resolve(null);
        }
    });
}


// Get a cached book - {key, text, chapters, title, author}, or null
export function getCachedBook(key) {
    return withStore('readonly', (store) => store.get(key));
}


// Cache a book
export function putCachedBook(record) {
    return withStore('readwrite', (store) => store.put(record));
}


// Remove a cached book
export function deleteCachedBook(key) {
    return withStore('readwrite', (store) => store.delete(key));
}
