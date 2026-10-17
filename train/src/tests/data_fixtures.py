# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
Tiny fake corpora for the data tests (test_data, test_multi, test_cremad) - ARCTIC and LibriTTS-R layouts with
synthetic recordings and phone labels that match the front end's tokens
"""

import contextlib
import os
import sys

import numpy
import soundfile

from picovox_train import frontend


class FakePool:
    """multiprocessing.Pool, in this process (so the workers' code is covered) - results, if given, replaces the work's
    results"""

    results = None

    def __init__(self, processes=None, initializer=None, initargs=()):
        self.processes = processes
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def imap(self, function, iterable, chunksize=1):
        assert chunksize >= 1
        return iter(FakePool.results) if FakePool.results is not None else map(function, iterable)

    def map(self, function, iterable, chunksize=1):
        return list(self.imap(function, iterable, chunksize))


# Fake modules in sys.modules (by name) for the context - only those names restored after (unittest.mock.patch.dict would
# drop every module imported meanwhile, and a compiled extension can't be imported twice)
@contextlib.contextmanager
def fake_modules(**modules):
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name in saved:
            del sys.modules[name]
        sys.modules.update({name: module for name, module in saved.items() if module is not None})


# Seconds of each phone, and of the edge silences
PHONE_SECONDS = 0.08
EDGE_SECONDS = 0.25


# A voiced, harmonic signal (a buzz at f0 with a vowel-like spectral tilt) of a duration at a sample rate
def buzz(seconds, rate, f0=150.0, seed=0):
    times = numpy.arange(int(round(seconds * rate))) / rate
    signal = sum(numpy.sin(2 * numpy.pi * f0 * harmonic * times) / harmonic for harmonic in range(1, 12))
    return 0.2 * signal + 0.002 * numpy.random.default_rng(seed).standard_normal(len(times))


# The phone labels [(label, start, end)] of a text's tokens - edge pauses, each phone PHONE_SECONDS; fmt 'xwaves' (ARCTIC
# lower-case, pau, ax) or 'mfa' (ARPAbet with stress, the edges as unlabeled gaps); drop removes phone indexes (0-based
# among the phones), replace maps phone indexes to other labels, pause_after inserts a pause after phone indexes
def text_labels(text, fmt='xwaves', drop=(), replace=None, pause_after=()):
    tokens = frontend.tokenize(text)['tokens']
    phones = [token for token in tokens if token not in frontend.BOUNDARIES]
    labels = []
    time = EDGE_SECONDS
    if fmt == 'xwaves':
        labels.append(('pau', 0.0, time))
    for index, phone in enumerate(phones):
        if index in drop:
            continue
        label = (replace or {}).get(index)
        if label is None:
            if fmt == 'xwaves':
                base = phone.rstrip('012').lower()
                label = 'ax' if phone == 'AH0' else base
            else:
                label = phone
        labels.append((label, time, time + PHONE_SECONDS))
        time += PHONE_SECONDS
        if index in pause_after:
            labels.append(('pau' if fmt == 'xwaves' else 'sil', time, time + 0.2))
            time += 0.2
    if fmt == 'xwaves':
        labels.append(('pau', time, time + EDGE_SECONDS))
    return labels, time + EDGE_SECONDS


# The recording of labels - silence in the pauses, a buzz in the phones - at a sample rate (float64 samples)
def labels_audio(labels, seconds, rate, seed=0):
    samples = numpy.zeros(int(round(seconds * rate)))
    for label, start, end in labels:
        if label.lower() not in ('pau', 'sil', 'sp', ''):
            begin = int(round(start * rate))
            segment = buzz(end - start, rate, seed=seed)
            samples[begin:begin + len(segment)] = segment[:len(samples) - begin]
    return samples


# Write an xwaves (ARCTIC) label file
def write_xwaves(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        file.write('signal arctic\ntype 0\ncolor 121\nfile_type 0\n#\n')
        for label, _, end in labels:
            file.write(f'{end:.5f} 125 {label}\n')


# Write a LibriTTSLabel (MFA) label file - tab-separated start, end, label
def write_mfa(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        for label, start, end in labels:
            file.write(f'{start:.3f}\t{end:.3f}\t{label}\n')


# Write the MFA word labels of a text (the word label file beside lab/phone - lab/word) from its phone labels
def write_mfa_words(path, text, labels):
    tokenized = frontend.tokenize(text)
    phone_labels = [label for label in labels if label[0] not in ('sil', '')]
    rows = []
    phone_index = 0
    for word in tokenized['words']:
        count = word['end'] - word['start']
        start, end = phone_labels[phone_index][1], phone_labels[phone_index + count - 1][2]
        rows.append((word['text'], start, end))
        phone_index += count
    write_mfa(path, rows)


# Make an ARCTIC speaker - cmu_us_<speaker>_arctic/{etc/txt.done.data, wav/<id>.wav, lab/<id>.lab} under arctic_dir
# from {sentence id: text} (a text of None: a recording with no prompt); returns the speaker directory
def make_arctic(arctic_dir, speaker, texts, rate=16000):
    directory = os.path.join(arctic_dir, f'cmu_us_{speaker}_arctic')
    os.makedirs(os.path.join(directory, 'etc'), exist_ok=True)
    os.makedirs(os.path.join(directory, 'wav'), exist_ok=True)
    with open(os.path.join(directory, 'etc', 'txt.done.data'), 'w', encoding='utf-8') as file:
        for sentence, text in texts.items():
            if text is not None:
                file.write(f'( {sentence} "{text}" )\n')
        file.write('not a prompt line\n')
    for number, (sentence, text) in enumerate(texts.items()):
        labels, seconds = text_labels(text or 'Call me Ishmael.')
        soundfile.write(os.path.join(directory, 'wav', sentence + '.wav'), labels_audio(labels, seconds, rate, number), rate)
        write_xwaves(os.path.join(directory, 'lab', sentence + '.lab'), labels)
    return directory


# Make a LibriTTS-R speaker - <libritts_dir>/<subset>/<speaker>/<chapter>/<id>.wav and .normalized.txt, and its MFA phone
# and word labels under label_dir (lab/phone) - from {utterance id: text}; without_labels: utterance ids with no labels
def make_libritts(libritts_dir, label_dir, subset, speaker, chapter, texts, without_labels=(), rate=24000):
    directory = os.path.join(libritts_dir, subset, speaker, chapter)
    os.makedirs(directory, exist_ok=True)
    for number, (utterance_id, text) in enumerate(texts.items()):
        labels, seconds = text_labels(text, 'mfa')
        soundfile.write(os.path.join(directory, utterance_id + '.wav'), labels_audio(labels, seconds, rate, number), rate)
        with open(os.path.join(directory, utterance_id + '.normalized.txt'), 'w', encoding='utf-8') as file:
            file.write(text + '\n')
        if utterance_id not in without_labels:
            phone_path = os.path.join(label_dir, subset, speaker, chapter, utterance_id + '.lab')
            write_mfa(phone_path, labels)
            write_mfa_words(phone_path.replace(os.sep + 'phone' + os.sep, os.sep + 'word' + os.sep), text, labels)
    return directory
