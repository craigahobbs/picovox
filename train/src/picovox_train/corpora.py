# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
Download the pinned files (FILES), idempotently: the CMU Pronouncing Dictionary (the G2P's training data), into
downloads/cmudict/, and the listener panel's Whisper model (picovox_train/panel.py), into downloads/whisper/.

    build/venv-train/bin/python -I train/src/picovox_train/corpora.py [--list] [name ...]

The files (names): cmudict, whisper. With no names, all of them are fetched. A file present is skipped; an interrupted
download resumes; a file is checked against its pinned SHA-256 (a revised file fails the fetch, never trains).
Downloads are sequential (one connection at a time) with a descriptive User-Agent. (The human speech corpora it once
fetched - CMU ARCTIC, LibriTTS-R, LibriTTSLabel - are in git history; the training speech is tts.py's.)
"""

import argparse
import hashlib
import os
import re
import sys
import time
import urllib.error
import urllib.request


# (the repository root - train/src/picovox_train/ is three levels below it)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
USER_AGENT = 'picovox-corpora/1.0 (+https://github.com/craigahobbs/picovox)'

# The pinned files - name: {url (at a fixed revision), path, sha256, description}: the CMU Pronouncing Dictionary
# (cmusphinx/cmudict - its license, BSD-style, is the repository's LICENSE), and the listener panel's Whisper model
# (whisper.cpp's base.en - picovox_train/panel.py)
CMUDICT_PATH = os.path.join(ROOT, 'downloads', 'cmudict', 'cmudict.dict')
WHISPER_MODEL = os.path.join(ROOT, 'downloads', 'whisper', 'ggml-base.en.bin')
FILES = {
    'cmudict': {
        'url': 'https://raw.githubusercontent.com/cmusphinx/cmudict/0f8072f814306c5ee4fbf992ed853601b12c01f9/cmudict.dict',
        'path': CMUDICT_PATH,
        'sha256': '81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22',
        'description': 'The CMU Pronouncing Dictionary (cmusphinx/cmudict, 2024-12-17) - the G2P\'s training data'
    },
    'whisper': {
        'url': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-base.en.bin',
        'path': WHISPER_MODEL,
        'sha256': 'a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002',
        'description': 'Whisper base.en for whisper.cpp - the listener panel\'s intelligibility'
    }
}

# The CMU Pronouncing Dictionary's entries from its lines - {word: [pronunciation (a list of phones), ...]}: every listed
# pronunciation in the dictionary's order ("word(2)" is the word's second; repeats kept), comments ("# ...", ";;;" lines)
# skipped
def parse_dictionary(lines):
    entries = {}
    for line in lines:
        content = line.split('#')[0].strip()
        if content and not content.startswith(';;;'):
            word, *phones = content.split()
            entries.setdefault(re.sub(r'\(\d+\)$', '', word), []).append(phones)
    return entries


# The downloaded dictionary's entries (g2p.py, readings.py, and heteronyms.py read it - the tests give their own)
def read_dictionary():
    with open(CMUDICT_PATH, encoding='utf-8') as file:
        return parse_dictionary(file)


# Print a progress line to stderr
def log(message):
    print(message, file=sys.stderr, flush=True)


# Download a URL to a path, resuming a partial download (path + '.part') with an HTTP Range request
def download(url, path):
    if os.path.exists(path):
        return
    part = path + '.part'
    offset = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {'User-Agent': USER_AGENT}
    if offset:
        headers['Range'] = f'bytes={offset}-'
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        # The partial file is already complete
        if error.code == 416:
            os.replace(part, path)
            return
        raise
    with response:
        mode = 'ab' if offset and response.status == 206 else 'wb'
        total = response.headers.get('Content-Length')
        total = int(total) + (offset if mode == 'ab' else 0) if total else None
        done = offset if mode == 'ab' else 0
        start = time.time()
        last = 0
        with open(part, mode) as file:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                file.write(chunk)
                done += len(chunk)
                if time.time() - last > 10:
                    last = time.time()
                    rate = (done - offset) / max(1e-3, last - start) / 1e6
                    percent = f' ({100 * done / total:.0f}%)' if total else ''
                    log(f'  {os.path.basename(path)}: {done / 1e6:.0f} MB{percent}, {rate:.1f} MB/s')
    if total is not None and os.path.getsize(part) != total:
        raise IOError(f'Incomplete download of {url}: {os.path.getsize(part)} of {total} bytes')
    os.replace(part, path)


# Fetch a pinned file, unless it's present - its SHA-256 checked (a mismatch removes it and fails); returns its download
# size in bytes
def fetch_file(name):
    item = FILES[name]
    if os.path.exists(item['path']):
        log(f'{name}: present')
        return 0
    os.makedirs(os.path.dirname(item['path']), exist_ok=True)
    log(f'{name}: downloading {item["url"]}')
    download(item['url'], item['path'])
    digest = hashlib.sha256()
    with open(item['path'], 'rb') as file:
        for chunk in iter(lambda: file.read(1 << 20), b''):
            digest.update(chunk)
    if digest.hexdigest() != item['sha256']:
        os.remove(item['path'])
        raise IOError(f'{name}: SHA-256 {digest.hexdigest()} is not the pinned {item["sha256"]}')
    return os.path.getsize(item['path'])


# Command-line entry point
def main():
    parser = argparse.ArgumentParser(description='Download the pinned files into downloads/')
    parser.add_argument('names', nargs='*', help='files to fetch (default: all)')
    parser.add_argument('--list', action='store_true', help='list the files and exit')
    args = parser.parse_args()
    if args.list:
        for name, item in FILES.items():
            state = 'present' if os.path.exists(item['path']) else 'missing'
            print(f'{name:28s} {state:8s} {item["description"]}')
        return
    names = args.names or list(FILES)
    unknown = [name for name in names if name not in FILES]
    if unknown:
        parser.error(f'unknown files: {", ".join(unknown)}')
    start = time.time()
    total = 0
    for name in names:
        total += fetch_file(name)
    log(f'Downloaded {total / 1e6:.0f} MB in {time.time() - start:.0f} s')


if __name__ == '__main__':
    main()
