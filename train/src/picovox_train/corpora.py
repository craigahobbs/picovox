# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
Download the pinned files (FILES), idempotently: the CMU Pronouncing Dictionary (the G2P's training data), into
downloads/cmudict/, the wav2vec2 phoneme recognizer (ctc.py's, wav2vec2.py - its weights, config, and vocabulary at a
fixed revision), into downloads/wav2vec2/, and the listener panel's Whisper model (picovox_train/panel.py), into
downloads/whisper/.

    build/venv-train/bin/python -I train/src/picovox_train/corpora.py [name ...]

The files (names): cmudict, wav2vec2-model, wav2vec2-config, wav2vec2-vocab, whisper. With no names, all of them are
fetched. A file present is skipped; a file is downloaded whole (an interrupted download starts over) and checked against
its pinned SHA-256 (a revised file fails the fetch, never trains). Downloads are sequential (one connection at a time)
with a descriptive User-Agent. (The human speech corpora it once fetched - CMU ARCTIC, LibriTTS-R, LibriTTSLabel - are
in git history; the training speech is tts.py's.)
"""

import argparse
import hashlib
import os
import re
import shutil
import sys
import time
import urllib.request


# (the repository root - train/src/picovox_train/ is three levels below it)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
USER_AGENT = 'picovox-corpora/1.0 (+https://github.com/craigahobbs/picovox)'

# The pinned files - name: {url (at a fixed revision), path, sha256, description}: the CMU Pronouncing Dictionary
# (cmusphinx/cmudict - its license, BSD-style, is the repository's LICENSE), the wav2vec2 phoneme recognizer
# (facebook/wav2vec2-lv-60-espeak-cv-ft on the Hugging Face Hub, Apache 2.0 - its revision ae45363b; ctc.py), and the
# listener panel's Whisper model (whisper.cpp's base.en - picovox_train/panel.py)
CMUDICT_PATH = os.path.join(ROOT, 'downloads', 'cmudict', 'cmudict.dict')
WAV2VEC2_DIR = os.path.join(ROOT, 'downloads', 'wav2vec2')
WHISPER_MODEL = os.path.join(ROOT, 'downloads', 'whisper', 'ggml-base.en.bin')
WAV2VEC2_URL = 'https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/resolve/ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4/'
FILES = {
    'cmudict': {
        'url': 'https://raw.githubusercontent.com/cmusphinx/cmudict/0f8072f814306c5ee4fbf992ed853601b12c01f9/cmudict.dict',
        'path': CMUDICT_PATH,
        'sha256': '81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22',
        'description': 'The CMU Pronouncing Dictionary (cmusphinx/cmudict, 2024-12-17) - the G2P\'s training data'
    },
    'wav2vec2-model': {
        'url': WAV2VEC2_URL + 'pytorch_model.bin',
        'path': os.path.join(WAV2VEC2_DIR, 'pytorch_model.bin'),
        'sha256': '3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e',
        'description': 'The wav2vec2 phoneme recognizer\'s weights (1.26 GB) - the training speech\'s alignment and readings, the '
                       'perceptual loss, and the gap'
    },
    'wav2vec2-config': {
        'url': WAV2VEC2_URL + 'config.json',
        'path': os.path.join(WAV2VEC2_DIR, 'config.json'),
        'sha256': '4609fb49b7e1d28aecb2840da1926c40bd915bc6f1120a940afacf7159bbfb13',
        'description': 'The recognizer\'s network configuration'
    },
    'wav2vec2-vocab': {
        'url': WAV2VEC2_URL + 'vocab.json',
        'path': os.path.join(WAV2VEC2_DIR, 'vocab.json'),
        'sha256': 'd732ab2456c0c017930001dc9af0b41b3b93d25b2eb9740bf9d925508d7d87d0',
        'description': 'The recognizer\'s IPA token vocabulary'
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


# The SHA-256 of a file's bytes (hex)
def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


# Print a progress line to stderr
def log(message):
    print(message, file=sys.stderr, flush=True)


# Download a URL to a path (by way of path + '.part', so an interrupted download is never the file)
def download(url, path):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response, open(path + '.part', 'wb') as file:
        shutil.copyfileobj(response, file, 1 << 20)
    os.replace(path + '.part', path)


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
    digest = file_sha256(item['path'])
    if digest != item['sha256']:
        os.remove(item['path'])
        raise IOError(f'{name}: SHA-256 {digest} is not the pinned {item["sha256"]}')
    return os.path.getsize(item['path'])


# Command-line entry point
def main():
    parser = argparse.ArgumentParser(description='Download the pinned files into downloads/')
    parser.add_argument('names', nargs='*', help='files to fetch (default: all)')
    args = parser.parse_args()
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
