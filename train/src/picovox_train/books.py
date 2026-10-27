# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The pinned book texts - the Project Gutenberg books make train reads (the heteronym labels' books - contextlabels.py -
and the distillation's - distill.py), each pinned in train/data/books.json to the exact text the shipped models were
trained on: its SHA-256, its size, its source URL, the date it was pinned, and an archived copy of the same bytes (a
Wayback Machine capture of the source URL, by its raw "id_" URL) where one exists.

Project Gutenberg revises its files in place (a new transcription, a fixed typo, a reflowed header) and keeps no old
revisions, so a book fetched by id today may not be the text a model was trained on - pg2554 changed on 2026-10-16
and moved the context labels by one example. A pinned text is fetched once into the books directory (its source URL,
else its archived copy - the first whose bytes have the pinned hash) and verified there on every use - the hash is of
the bytes as fetched and stored, which every reader decodes the same way. A text that isn't the pinned one is never
used: the fetch fails with the book, the expected and actual hashes, and how to proceed - restore the pinned copy, or
re-pin deliberately (pin, then retrain: the labels or the distillation set change with the text).

The pins' bundle is our own copy of every pinned text: one xz-compressed tar (deterministic - the texts in name order,
no times or owners) published with the picovox site (make gh-pages - https://craigahobbs.github.io/picovox/books/),
recorded in the pins with its SHA-256. It's the fetch's first source - downloaded once into the books directory and
verified - so make train reproduces its texts even after Gutenberg revises them; the book's source URL and its archived
copy are the fallbacks. A re-pinned text needs a new bundle (bundle with a new tag, then make gh-pages).

    PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.books fetch [NAME ...]   # fetch and verify (default: all)
    PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.books pin NAME           # re-pin to the books directory's copy
    PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.books bundle TAG         # bundle every pinned text as TAG
    PYTHONPATH=train/src python3 -B -m picovox_train.books publish DIR                          # the bundle into the site (DIR/books/)

A text's name is its file's in the books directory (<name>.txt): pg<id> for a book's revision the labels read, and
pg<id>-<date> for another revision of the same book (the distillation's pg2554 is a later revision than the labels').
"""

import argparse
import base64
import datetime
import hashlib
import io
import json
import lzma
import os
import re
import sys
import tarfile
import urllib.error
import urllib.request

from . import ROOT


# The pins, the default books directory, and the fetch's User-Agent
PINS_PATH = os.path.join(ROOT, 'train', 'data', 'books.json')
BOOKS_DIR = os.path.join(ROOT, 'build', 'nn', 'books')
USER_AGENT = 'picovox-train/1.0 (+https://github.com/craigahobbs/picovox)'

# The Wayback Machine's capture index (CDX) and a capture's raw bytes
CDX_URL = 'https://web.archive.org/cdx/search/cdx?url={url}&output=json&filter=digest:{digest}&filter=statuscode:200'
ARCHIVE_URL = 'https://web.archive.org/web/{timestamp}id_/{url}'

# The pins' bundle - published with the picovox site (make gh-pages)
BUNDLE_URL = 'https://craigahobbs.github.io/picovox/books/{tag}.tar.xz'


class PinError(Exception):
    """A book text that isn't its pinned text (or isn't pinned)"""


# A path relative to the repository root, for messages
def display_path(path):
    relative = os.path.relpath(path, ROOT)
    return path if relative.startswith('..') else relative


# The pins - train/data/books.json's texts, {name: {id, url, sha256, bytes, pinned, updated, archive}}
def load_pins(path=PINS_PATH):
    with open(path, encoding='utf-8') as file:
        return json.load(file)['texts']


# The pins' bundle - train/data/books.json's bundle, {tag, url, sha256, bytes}, or None
def load_bundle(path=PINS_PATH):
    with open(path, encoding='utf-8') as file:
        return json.load(file).get('bundle')


# Write the pins - train/data/books.json, one text per line (in name order), and its bundle (default: the file's)
def write_pins(texts, path=PINS_PATH, bundle=None):
    with open(path, encoding='utf-8') as file:
        pins = json.load(file)
    bundle = bundle or pins.get('bundle')
    lines = [f'        {json.dumps(name)}: {json.dumps(texts[name])}' for name in sorted(texts, key=name_key)]
    with open(path + '.tmp', 'w', encoding='utf-8') as file:
        file.write('{\n    "description": ' + json.dumps(pins['description']) + ',\n' +
                   (f'    "bundle": {json.dumps(bundle)},\n' if bundle else '') +
                   '    "texts": {\n' + ',\n'.join(lines) + '\n    }\n}\n')
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


# The bundle's archive in the books directory - downloaded once (its URL) and verified. PinError if the download isn't
# the pinned bundle.
def bundle_path(books_dir, bundle):
    path = os.path.join(books_dir, os.path.basename(bundle['url']))
    if os.path.exists(path):
        with open(path, 'rb') as file:
            if sha256(file.read()) == bundle['sha256']:
                return path
    data = fetch_url(bundle['url'])
    actual = sha256(data)
    if actual != bundle['sha256']:
        raise PinError(f'sha256 {actual} ({len(data)} bytes), not the pinned bundle\'s {bundle["sha256"]}')
    os.makedirs(books_dir, exist_ok=True)
    with open(path + '.tmp', 'wb') as file:
        file.write(data)
    os.replace(path + '.tmp', path)
    return path


# Unpack the bundle's pinned texts into the books directory - in one pass, each verified against its pin (a text the
# directory has, or that isn't pinned or isn't the pinned text, is skipped)
def unpack_bundle(books_dir, path, pins):
    with tarfile.open(path, 'r:xz') as archive:
        for member in archive:
            pin = pins.get(member.name[:-len('.txt')]) if member.name.endswith('.txt') else None
            target = os.path.join(books_dir, os.path.basename(member.name))
            if pin is None or os.path.exists(target):
                continue
            data = archive.extractfile(member).read()
            if sha256(data) == pin['sha256']:
                with open(target + '.tmp', 'wb') as file:
                    file.write(data)
                os.replace(target + '.tmp', target)


# The path of a pinned text in the books directory - fetched if missing (the pins' bundle, else its source URL, else
# its archived copy - the first with the pinned bytes) and verified against its pin. PinError if it isn't pinned, or
# isn't the pinned text.
def pinned_path(books_dir, name, pins, bundle=None):
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
    sources = ([bundle['url']] if bundle else []) + [pin['url']] + ([pin['archive']] if pin.get('archive') else [])
    for url in sources:
        try:
            if bundle and url == bundle['url']:
                unpack_bundle(books_dir, bundle_path(books_dir, bundle), pins)
                if os.path.exists(path):
                    return path
                attempts.append(f'{url}: no pinned copy in the bundle')
                continue
            data = fetch_url(url)
        except (OSError, PinError, tarfile.TarError, lzma.LZMAError) as error:
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
    pins, bundle = load_pins(pins_path), load_bundle(pins_path)
    return [pinned_path(books_dir, name, pins, bundle) for name in names]


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
    pins, bundle = load_pins(args.pins), load_bundle(args.pins)
    names = args.names or sorted(pins, key=name_key)
    try:
        for name in names:
            pinned_path(args.books, name, pins, bundle)
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


def bundle_command(args):
    """Bundle every pinned text (verified in the books directory) as TAG.tar.xz - a deterministic tar (the texts in
    name order, no times or owners), xz-compressed - and record it as the pins' bundle"""
    pins = load_pins(args.pins)
    names = sorted(pins, key=name_key)
    try:
        paths = [pinned_path(args.books, name, pins) for name in names]
    except PinError as error:
        print(f'books: {error}', file=sys.stderr)
        return 1
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode='w', format=tarfile.USTAR_FORMAT) as archive:
        for name, path in zip(names, paths):
            with open(path, 'rb') as file:
                text = file.read()
            info = tarfile.TarInfo(f'{name}.txt')
            info.size = len(text)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(text))
    data = lzma.compress(tar_bytes.getvalue(), preset=9 | lzma.PRESET_EXTREME)
    url = BUNDLE_URL.format(tag=args.tag)
    path = os.path.join(args.books, os.path.basename(url))
    with open(path, 'wb') as file:
        file.write(data)
    bundle = {'tag': args.tag, 'url': url, 'sha256': sha256(data), 'bytes': len(data)}
    write_pins(pins, args.pins, bundle)
    print(f'books: bundled {len(names)} pinned texts as {display_path(path)} - sha256 {bundle["sha256"]} ({len(data)} bytes); '
          f'publish it with the site (make gh-pages - {url})')
    return 0


def publish_command(args):
    """Copy the pins' bundle (verified in the books directory - downloaded if missing) into the site's books/"""
    bundle = load_bundle(args.pins)
    if bundle is None:
        print(f'books: no bundle in {display_path(args.pins)} (python -m picovox_train.books bundle TAG)', file=sys.stderr)
        return 1
    try:
        source = bundle_path(args.books, bundle)
    except (OSError, PinError) as error:
        print(f'books: the bundle {bundle["url"]}: {error}', file=sys.stderr)
        return 1
    directory = os.path.join(args.site, 'books')
    os.makedirs(directory, exist_ok=True)
    with open(source, 'rb') as file:
        data = file.read()
    with open(os.path.join(directory, os.path.basename(source)), 'wb') as file:
        file.write(data)
    print(f'books: published {os.path.basename(source)} in {display_path(directory)}')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picovox_train.books', description='The pinned book texts')
    parser.add_argument('--pins', default=PINS_PATH, help='the pins (default train/data/books.json)')
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
    bundle_parser = commands.add_parser('bundle', help='bundle every pinned text (published with the site)')
    bundle_parser.add_argument('tag', help='the bundle\'s name (e.g. training-books-v1)')
    bundle_parser.set_defaults(func=bundle_command)
    publish_parser = commands.add_parser('publish', help='copy the bundle into the site')
    publish_parser.add_argument('site', help='the site directory (the bundle goes in its books/)')
    publish_parser.set_defaults(func=publish_command)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
