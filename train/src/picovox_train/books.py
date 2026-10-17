# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The pinned book texts - the Project Gutenberg books make train reads (the heteronym labels' books - contextlabels.py -
and the distillation's - distill.py), each pinned in data/nn/books.json to the exact text the shipped models were
trained on: its SHA-256, its size, its source URL, the date it was pinned, and an archived copy of the same bytes (a
Wayback Machine capture of the source URL, by its raw "id_" URL) where one exists.

Project Gutenberg revises its files in place (a new transcription, a fixed typo, a reflowed header) and keeps no old
revisions, so a book fetched by id today may not be the text a model was trained on - pg2554 changed on 2026-10-16
and moved the context labels by one example. A pinned text is fetched once into the books directory (its source URL,
else its archived copy - the first whose bytes have the pinned hash) and verified there on every use - the hash is of
the bytes as fetched and stored, which every reader decodes the same way. A text that isn't the pinned one is never
used: the fetch fails with the book, the expected and actual hashes, and how to proceed - restore the pinned copy, or
re-pin deliberately (pin, then retrain: the labels or the distillation set change with the text).

    PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.books fetch [NAME ...]   # fetch and verify (default: all)
    PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.books pin NAME           # re-pin to the books directory's copy

A text's name is its file's in the books directory (<name>.txt): pg<id> for a book's revision the labels read, and
pg<id>-<date> for another revision of the same book (the distillation's pg2554 is a later revision than the labels').
"""

import argparse
import base64
import datetime
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request

from . import ROOT


# The pins, the default books directory, and the fetch's User-Agent
PINS_PATH = os.path.join(ROOT, 'data', 'nn', 'books.json')
BOOKS_DIR = os.path.join(ROOT, 'build', 'nn', 'books')
USER_AGENT = 'voice-synth-train/1.0 (+https://github.com/craigahobbs/voice-synth)'

# The Wayback Machine's capture index (CDX) and a capture's raw bytes
CDX_URL = 'https://web.archive.org/cdx/search/cdx?url={url}&output=json&filter=digest:{digest}&filter=statuscode:200'
ARCHIVE_URL = 'https://web.archive.org/web/{timestamp}id_/{url}'


class PinError(Exception):
    """A book text that isn't its pinned text (or isn't pinned)"""


# A path relative to the repository root, for messages
def display_path(path):
    relative = os.path.relpath(path, ROOT)
    return path if relative.startswith('..') else relative


# The pins - data/nn/books.json's texts, {name: {id, url, sha256, bytes, pinned, updated, archive}}
def load_pins(path=PINS_PATH):
    with open(path, encoding='utf-8') as file:
        return json.load(file)['texts']


# Write the pins - data/nn/books.json, one text per line (in name order)
def write_pins(texts, path=PINS_PATH):
    with open(path, encoding='utf-8') as file:
        description = json.load(file)['description']
    lines = [f'        {json.dumps(name)}: {json.dumps(texts[name])}' for name in sorted(texts, key=name_key)]
    with open(path + '.tmp', 'w', encoding='utf-8') as file:
        file.write('{\n    "description": ' + json.dumps(description) + ',\n    "texts": {\n' + ',\n'.join(lines) + '\n    }\n}\n')
    os.replace(path + '.tmp', path)


# A text name's sort key - by book id, then revision
def name_key(name):
    match = re.match(r'pg(\d+)(.*)$', name)
    return (int(match.group(1)), match.group(2)) if match else (sys.maxsize, name)


# The SHA-256 of bytes (hex)
def sha256(data):
    return hashlib.sha256(data).hexdigest()


# A URL's bytes
def fetch_url(url):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


# How to proceed when a text isn't its pinned text
def remedy(name, pin, path):
    return (f'To proceed, either restore the pinned text - put a copy with sha256 {pin["sha256"]} (an earlier run\'s '
            f'{os.path.basename(path)}, kept in its books directory) at {display_path(path)} - or re-pin deliberately: put the new '
            f'text at {display_path(path)} and run "python -m picovox_train.books pin {name}", then retrain - the text is '
            'training data (the heteronym labels or the distillation set), so make train will no longer reproduce the '
            'committed models byte for byte.')


# The path of a pinned text in the books directory - fetched if missing (its source URL, else its archived copy - the
# first with the pinned bytes) and verified against its pin. PinError if it isn't pinned, or isn't the pinned text.
def pinned_path(books_dir, name, pins):
    path = os.path.join(books_dir, f'{name}.txt')
    pin = pins.get(name)
    if pin is None:
        raise PinError(f'book {name}: not pinned in {display_path(PINS_PATH)} - fetch it to {display_path(path)} and pin it '
                       f'(python -m picovox_train.books pin {name})')
    expected = f'sha256 {pin["sha256"]} ({pin["bytes"]} bytes, pinned {pin["pinned"]})'
    if os.path.exists(path):
        with open(path, 'rb') as file:
            actual = sha256(file.read())
        if actual != pin['sha256']:
            raise PinError(f'book {name} (Project Gutenberg #{pin["id"]}): the copy {display_path(path)} has sha256 {actual}, '
                           f'not the pinned {expected}. It is another revision of the book. Remove it to fetch the pinned '
                           f'text. ' + remedy(name, pin, path))
        return path
    os.makedirs(books_dir, exist_ok=True)
    attempts = []
    unpinned = None
    for url in [pin['url']] + ([pin['archive']] if pin.get('archive') else []):
        try:
            data = fetch_url(url)
        except (urllib.error.URLError, OSError) as error:
            attempts.append(f'{url}: {error}')
            continue
        actual = sha256(data)
        if actual == pin['sha256']:
            with open(path + '.tmp', 'wb') as file:
                file.write(data)
            os.replace(path + '.tmp', path)
            return path
        attempts.append(f'{url}: sha256 {actual} ({len(data)} bytes)')
        if unpinned is None:
            unpinned = data
    if unpinned is not None:
        with open(path + '.unpinned', 'wb') as file:
            file.write(unpinned)
        attempts.append(f'(the first of those texts is saved as {display_path(path)}.unpinned)')
    if not pin.get('archive'):
        attempts.append('(no archived copy of the pinned text is known)')
    raise PinError(f'book {name} (Project Gutenberg #{pin["id"]}): no source has the pinned text, {expected}:\n  ' +
                   '\n  '.join(attempts) + '\nProject Gutenberg revises its books in place. ' + remedy(name, pin, path))


# The paths of pinned texts in the books directory, fetched and verified (pinned_path)
def pinned_paths(books_dir, names, pins_path=PINS_PATH):
    pins = load_pins(pins_path)
    return [pinned_path(books_dir, name, pins) for name in names]


# The Wayback Machine capture of a URL with the bytes' SHA-1 (the CDX index's digest) - its raw URL, or None
def archived_copy(url, data):
    digest = base64.b32encode(hashlib.sha1(data).digest()).decode('ascii')
    for source in (url, url.replace('://www.', '://')):
        rows = json.loads(fetch_url(CDX_URL.format(url=source.split('://', 1)[1], digest=digest)) or b'[]')
        if len(rows) > 1:
            capture = dict(zip(rows[0], rows[-1]))
            return ARCHIVE_URL.format(timestamp=capture['timestamp'], url=capture['original'])
    return None


# A Gutenberg text's "Most recently updated" date (its revision), or None
def updated_date(data):
    match = re.search(rb'Most recently updated: ([^\r\n]+)', data[:8000])
    return match.group(1).decode('utf-8').strip() if match else None


def fetch_command(args):
    """Fetch and verify pinned texts (default: every one) in the books directory"""
    pins = load_pins(args.pins)
    names = args.names or sorted(pins, key=name_key)
    try:
        for name in names:
            pinned_path(args.books, name, pins)
    except PinError as error:
        print(f'books: {error}', file=sys.stderr)
        return 1
    print(f'books: {len(names)} pinned texts in {display_path(args.books)}')
    return 0


def pin_command(args):
    """Pin a text to the books directory's copy - its hash, size, source URL, the date, and an archived copy if the
    Wayback Machine has those bytes"""
    pins = load_pins(args.pins)
    path = os.path.join(args.books, f'{args.name}.txt')
    with open(path, 'rb') as file:
        data = file.read()
    old = pins.get(args.name)
    book_id = args.id or (old and old['id']) or int(re.match(r'pg(\d+)', args.name).group(1))
    url = f'https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt'
    try:
        archive = archived_copy(url, data)
    except (urllib.error.URLError, OSError, ValueError) as error:
        print(f'books: no archived copy of {args.name} ({error})', file=sys.stderr)
        archive = None
    pins[args.name] = {'id': book_id, 'url': url, 'sha256': sha256(data), 'bytes': len(data),
                       'pinned': args.date or datetime.date.today().isoformat(), 'updated': updated_date(data), 'archive': archive}
    write_pins(pins, args.pins)
    print(f'books: pinned {args.name} - sha256 {pins[args.name]["sha256"]} ({len(data)} bytes), archive {archive}' +
          (f' (was sha256 {old["sha256"]})' if old else ''))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picovox_train.books', description='The pinned book texts')
    parser.add_argument('--pins', default=PINS_PATH, help='the pins (default data/nn/books.json)')
    parser.add_argument('--books', default=BOOKS_DIR, help='the books directory (default build/nn/books)')
    commands = parser.add_subparsers(dest='command', required=True)
    fetch_parser = commands.add_parser('fetch', help='fetch and verify pinned texts')
    fetch_parser.add_argument('names', nargs='*', help='the texts (default: every pinned text)')
    fetch_parser.set_defaults(func=fetch_command)
    pin_parser = commands.add_parser('pin', help='pin a text to the books directory\'s copy')
    pin_parser.add_argument('name')
    pin_parser.add_argument('--id', type=int, help='its Project Gutenberg id (default: its pin\'s, else the name\'s)')
    pin_parser.add_argument('--date', help='the date pinned (default today)')
    pin_parser.set_defaults(func=pin_command)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
