# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
Download and unpack the human speech corpora used as calibration targets, idempotently, into downloads/corpora/ - and
the listener panel's Whisper model (picovox_train/panel.py), into downloads/whisper/.

    build/venv-nn/bin/python -I train/src/picovox_train/corpora.py [--list] [--keep-archives] [name ...]

The corpora (names): the labeled CMU ARCTIC US English speakers (arctic-<speaker>), the LibriTTS-R dev-clean audiobook
and train-clean-100 subsets (libritts-r-dev-clean, libritts-r-train-clean-100), and the LibriTTSLabel forced alignments
of LibriTTS (libritts-label-dev-clean, libritts-label-train-clean-100).
With no names, all of them are fetched, and the Whisper model. A corpus already unpacked (or the model present) is
skipped; an interrupted download resumes.
Downloads are sequential (one connection at a time) with a descriptive User-Agent.
"""

import argparse
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request


# (the repository root - train/src/picovox_train/ is three levels below it)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CORPORA_DIR = os.path.join(ROOT, 'downloads', 'corpora')
ARCHIVES_DIR = os.path.join(CORPORA_DIR, 'archives')
USER_AGENT = 'picovox-corpora/1.0 (+https://github.com/craigahobbs/picovox; research calibration)'

ARCTIC_RELEASE_URL = 'http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_{speaker}_arctic-0.95-release.tar.bz2'
ARCTIC_LICENSE = (
    'CMU ARCTIC databases, Carnegie Mellon University, Language Technologies Institute (Kominek and Black, 2003). '
    'A free license - "free for use for any purpose (commercial or otherwise)": permission to use, copy, modify, and '
    'license without fee, provided the copyright notice is retained, modifications are marked, and the original '
    'authors\' names are kept (the full text is in each 0.95 release\'s COPYING file)'
)

# The listener panel's Whisper model - whisper.cpp's base.en (picovox_train/panel.py)
WHISPER_URL = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin'
WHISPER_MODEL = os.path.join(ROOT, 'downloads', 'whisper', 'ggml-base.en.bin')

# The US English ARCTIC speakers with a 0.95 release's phone labels (per festvox.org/cmu_arctic): sex and notes
ARCTIC_SPEAKERS = {
    'bdl': ('male', 'US English, experienced voice talent'),
    'rms': ('male', 'US English'),
    'slt': ('female', 'US English, experienced voice talent'),
    'clb': ('female', 'US English')
}

# The corpora - name: {url(s), destination, size note, license, and an optional member filter}
CORPORA = {}
for _speaker, (_sex, _note) in ARCTIC_SPEAKERS.items():
    CORPORA[f'arctic-{_speaker}'] = {
        'urls': [ARCTIC_RELEASE_URL.format(speaker=_speaker)],
        'dest': os.path.join(CORPORA_DIR, 'arctic'),
        'check': os.path.join(CORPORA_DIR, 'arctic', f'cmu_us_{_speaker}_arctic', 'lab'),
        'description': f'CMU ARCTIC {_speaker} ({_sex}, {_note}) - phonetically balanced sentences, 16 kHz WAV, EGG, and phone '
                       'labels (lab/*.lab, xwaves format, 10 ms resolution)',
        'license': ARCTIC_LICENSE
    }
CORPORA['libritts-r-dev-clean'] = {
    'urls': ['https://www.openslr.org/resources/141/dev_clean.tar.gz'],
    'dest': os.path.join(CORPORA_DIR, 'libritts_r'),
    'check': os.path.join(CORPORA_DIR, 'libritts_r', 'LibriTTS_R', 'dev-clean'),
    'sizePaths': [os.path.join(CORPORA_DIR, 'libritts_r', 'LibriTTS_R', 'dev-clean')],
    'description': 'LibriTTS-R dev-clean (OpenSLR 141) - 40 LibriVox audiobook readers (20 female, 20 male), 24 kHz, '
                   'sentence-segmented, with original and normalized text; speech-restored LibriTTS',
    'license': 'CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) - Koizumi et al., "LibriTTS-R", 2023; '
               'derived from LibriTTS (Zen et al., 2019, CC BY 4.0) and LibriSpeech/LibriVox (public domain recordings)'
}
CORPORA['libritts-r-train-clean-100'] = {
    'urls': ['https://openslr.elda.org/resources/141/train_clean_100.tar.gz'],    # the OpenSLR EU mirror (faster)
    'dest': os.path.join(CORPORA_DIR, 'libritts_r'),
    'check': os.path.join(CORPORA_DIR, 'libritts_r', 'LibriTTS_R', 'train-clean-100'),
    'sizePaths': [os.path.join(CORPORA_DIR, 'libritts_r', 'LibriTTS_R', 'train-clean-100')],
    'description': 'LibriTTS-R train-clean-100 (OpenSLR 141) - 247 LibriVox audiobook readers (123 female, 124 male; 33,232 '
                   'utterances, about 54 hours - the "100" is LibriSpeech\'s hours, before LibriTTS\'s filtering), 24 kHz, '
                   'sentence-segmented, with original and normalized text; speech-restored LibriTTS',
    'license': 'CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) - Koizumi et al., "LibriTTS-R", 2023; '
               'derived from LibriTTS (Zen et al., 2019, CC BY 4.0) and LibriSpeech/LibriVox (public domain recordings)'
}
CORPORA['libritts-r-doc'] = {
    'urls': ['https://www.openslr.org/resources/141/doc.tar.gz'],
    'dest': os.path.join(CORPORA_DIR, 'libritts_r'),
    'check': os.path.join(CORPORA_DIR, 'libritts_r', 'LibriTTS_R', 'speakers.tsv'),
    'description': 'LibriTTS-R documentation (OpenSLR 141) - speakers.tsv (reader sex and subset), SPEAKERS.txt, '
                   'BOOKS.txt, CHAPTERS.txt, and the license',
    'license': 'CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)'
}
CORPORA['libritts-label-dev-clean'] = {
    'urls': [f'https://raw.githubusercontent.com/kan-bayashi/LibriTTSLabel/master/lab.tar.gz-{ix}' for ix in range(5)],
    'dest': os.path.join(CORPORA_DIR, 'libritts_label'),
    'check': os.path.join(CORPORA_DIR, 'libritts_label', 'lab', 'phone', 'dev-clean'),
    'members': ('lab/phone/dev-clean/', 'lab/word/dev-clean/'),
    'sizePaths': [os.path.join(CORPORA_DIR, 'libritts_label', 'lab', kind, 'dev-clean') for kind in ('phone', 'word')],
    'description': 'LibriTTSLabel (github.com/kan-bayashi/LibriTTSLabel) - Montreal Forced Aligner phone and word '
                   'alignments of LibriTTS dev-clean (the same utterances as LibriTTS-R)',
    'license': 'No license stated by the repository - used here only locally, for measurement, and not redistributed'
}
CORPORA['libritts-label-train-clean-100'] = {
    'urls': CORPORA['libritts-label-dev-clean']['urls'],
    'dest': os.path.join(CORPORA_DIR, 'libritts_label'),
    'check': os.path.join(CORPORA_DIR, 'libritts_label', 'lab', 'phone', 'train-clean-100'),
    'members': ('lab/phone/train-clean-100/', 'lab/word/train-clean-100/'),
    'sizePaths': [os.path.join(CORPORA_DIR, 'libritts_label', 'lab', kind, 'train-clean-100') for kind in ('phone', 'word')],
    'description': 'LibriTTSLabel (github.com/kan-bayashi/LibriTTSLabel) - Montreal Forced Aligner phone and word '
                   'alignments of LibriTTS train-clean-100 (the same utterances as LibriTTS-R)',
    'license': 'No license stated by the repository - used here only locally, for measurement, and not redistributed'
}

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


# Concatenate split archive parts into one file (idempotent)
def concatenate(parts, path):
    if os.path.exists(path):
        return
    with open(path + '.part', 'wb') as output:
        for part in parts:
            with open(part, 'rb') as file:
                shutil.copyfileobj(file, output)
    os.replace(path + '.part', path)


# Unpack a tar archive safely (the 'data' filter - no absolute paths, links outside, or devices), optionally only the
# members under the given path prefixes
def unpack(archive, dest, members=None):
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(archive, 'r:*') as tar:
        if members is None:
            tar.extractall(dest, filter='data')
        else:
            selected = (info for info in tar if info.name.lstrip('./').startswith(members))
            tar.extractall(dest, members=selected, filter='data')


# Directory size in bytes
def directory_size(path):
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for filename in filenames:
            total += os.path.getsize(os.path.join(dirpath, filename))
    return total


# Fetch and unpack one corpus, unless it's already unpacked - returns its download size in bytes
def fetch(name, keep_archives):
    corpus = CORPORA[name]
    if os.path.exists(corpus['check']):
        log(f'{name}: present')
        return 0
    os.makedirs(ARCHIVES_DIR, exist_ok=True)
    paths = []
    for url in corpus['urls']:
        path = os.path.join(ARCHIVES_DIR, os.path.basename(url))
        log(f'{name}: downloading {url}')
        download(url, path)
        paths.append(path)
    archive = paths[0]
    if len(paths) > 1:
        archive = os.path.join(ARCHIVES_DIR, name + '.tar.gz')
        concatenate(paths, archive)
    size = sum(os.path.getsize(path) for path in paths)
    log(f'{name}: unpacking {size / 1e6:.0f} MB')
    unpack(archive, corpus['dest'], corpus.get('members'))
    if not os.path.exists(corpus['check']):
        raise IOError(f'{name}: unpacking did not produce {corpus["check"]}')
    if not keep_archives:
        for path in set(paths + [archive]):
            os.remove(path)
    return size


# Fetch the Whisper model, unless it's present - returns its download size in bytes
def fetch_whisper():
    if os.path.exists(WHISPER_MODEL):
        log('whisper: present')
        return 0
    os.makedirs(os.path.dirname(WHISPER_MODEL), exist_ok=True)
    log(f'whisper: downloading {WHISPER_URL}')
    download(WHISPER_URL, WHISPER_MODEL)
    return os.path.getsize(WHISPER_MODEL)


# Write downloads/corpora/README.md - each corpus's source, description, license, and unpacked size
def write_readme():
    lines = [
        '# Speech corpora',
        '',
        'Human speech for calibrating the synthesizer\'s voices, fetched by `train/src/picovox_train/corpora.py`. This directory is',
        'gitignored - the corpora are not redistributed with PicoVox. Only small derived statistics are committed',
        '(`data/nn/speakers.json`, `data/nn/cremad.json`).',
        ''
    ]
    for name, corpus in CORPORA.items():
        present = os.path.exists(corpus['check'])
        if not present:
            size = 'not downloaded'
        elif os.path.isdir(corpus['check']):
            paths = corpus.get('sizePaths', [os.path.dirname(corpus['check'])])
            size = f'{sum(directory_size(path) for path in paths) / 1e6:.0f} MB unpacked'
        else:
            size = 'small (documentation files)'
        lines.extend([
            f'## {name}',
            '',
            f'- Source: {", ".join(corpus["urls"]) if len(corpus["urls"]) == 1 else corpus["urls"][0] + " (and parts)"}',
            f'- Contents: {corpus["description"]}',
            f'- License: {corpus["license"]}',
            f'- Size: {size}',
            ''
        ])
    # Keep the sections other tools append (each between <!-- ... --> markers - cremad.py's, for example)
    readme_path = os.path.join(CORPORA_DIR, 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, encoding='utf-8') as file:
            existing = file.read()
        marker = existing.find('\n<!-- ')
        if marker >= 0:
            lines.append(existing[marker + 1:].rstrip('\n') + '\n')
    with open(readme_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines))


# Command-line entry point
def main():
    parser = argparse.ArgumentParser(description='Download the speech calibration corpora into downloads/corpora/')
    parser.add_argument('names', nargs='*', help='corpora to fetch (default: all, and the Whisper model)')
    parser.add_argument('--list', action='store_true', help='list the corpora and exit')
    parser.add_argument('--keep-archives', action='store_true', help='keep the downloaded archives')
    args = parser.parse_args()
    if args.list:
        for name, corpus in CORPORA.items():
            state = 'present' if os.path.exists(corpus['check']) else 'missing'
            print(f'{name:28s} {state:8s} {corpus["description"]}')
        return
    names = args.names or list(CORPORA)
    unknown = [name for name in names if name not in CORPORA]
    if unknown:
        parser.error(f'unknown corpora: {", ".join(unknown)}')
    os.makedirs(CORPORA_DIR, exist_ok=True)
    start = time.time()
    total = 0
    for name in names:
        total += fetch(name, args.keep_archives)
    if not args.names:
        total += fetch_whisper()
    write_readme()
    log(f'Downloaded {total / 1e6:.0f} MB in {time.time() - start:.0f} s')


if __name__ == '__main__':
    main()
