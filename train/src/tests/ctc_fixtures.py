# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The measurement tests' fixtures - a fake wav2vec2 phoneme recognizer (ctc.py's), and a tiny training speech corpus
(tts.py's: the recipe's tts section, a selection, WAVs, and alignments) in a temporary downloads/tts/
"""

import contextlib
import json
import os
import tempfile
import types
import unittest.mock

import torch

from picovox_train import ctc, frontend, tts
from tests import data_fixtures
from tests.helpers import write_wav


class FakeExecutor:
    """concurrent.futures.ProcessPoolExecutor, in this process (so the workers' code is covered)"""

    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    @staticmethod
    def map(function, iterable):
        return map(function, iterable)


# A recognizer vocabulary - the blank and specials, then every IPA token
VOCAB = {'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3}
for _tokens in ctc.IPA_TOKENS.values():
    for _token in _tokens:
        VOCAB.setdefault(_token, len(VOCAB))


# The recognizer's greedy string of a text's phones (each phone's usual IPA token), less the last tail phones
def greedy_of(text, tail=0):
    keys = [key for _, key in ctc.phone_positions(frontend.tokenize(text)['tokens'])]
    return ' '.join(ctc.IPA_TOKENS[key][0] for key in keys[:len(keys) - tail])


# A clip's alignment labels [[label, start, end]] (our token names, 'pau' edges) - and its seconds
def clip_labels(text):
    labels, seconds = data_fixtures.text_labels(text)
    rows = [['pau', 0.0, data_fixtures.EDGE_SECONDS]] + [[label, start, end] for label, start, end in labels]
    rows.append(['pau', labels[-1][2], seconds])
    return rows, seconds


class FakeCtc:
    """A fake wav2vec2 CTC model - random logits, one frame per 20 ms"""

    def __init__(self):
        self.device = None
        self.calls = []

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, inputs, attention_mask):
        self.calls.append(tuple(inputs.shape))
        generator = torch.Generator().manual_seed(inputs.shape[1])
        logits = torch.randn(inputs.shape[0], inputs.shape[1] // 320, len(VOCAB), generator=generator) * 2
        logits[..., 0] += 3 * (attention_mask[:, ::320][:, :logits.shape[1]] == 0)
        return types.SimpleNamespace(logits=logits)

    @staticmethod
    def _get_feat_extract_output_lengths(lengths):
        return lengths // 320


# The fake recognizer's modules (transformers, huggingface_hub) and a snapshot directory with its vocabulary
@contextlib.contextmanager
def fake_recognizer():
    with tempfile.TemporaryDirectory() as directory:
        with open(os.path.join(directory, 'vocab.json'), 'w', encoding='utf-8') as file:
            json.dump(VOCAB, file)
        model = FakeCtc()
        transformers = types.ModuleType('transformers')
        transformers.Wav2Vec2ForCTC = types.SimpleNamespace(from_pretrained=unittest.mock.Mock(return_value=model))
        hub = types.ModuleType('huggingface_hub')
        hub.snapshot_download = unittest.mock.Mock(return_value=directory)
        with data_fixtures.fake_modules(transformers=transformers, huggingface_hub=hub), \
                unittest.mock.patch.dict(os.environ), \
                unittest.mock.patch.object(torch.backends.mps, 'is_available', return_value=False):
            yield model, hub


# A tiny recipe's tts section - two voices (male, female)
def tts_settings(**overrides):
    return {
        'engine': {'url': 'http://localhost:11434', 'model': 'phonon', 'digest': '0123456789ab', 'license': 'Phonon (Apache 2.0)'},
        'voices': {'male': {'voice': 'Trevor', 'sex': 'male'}, 'female': {'voice': 'Harper', 'sex': 'female'}},
        'hoursPerVoice': 0.001, 'wordsPerSecond': 3.0, 'workers': 2, 'seed': 1,
        'texts': {'books': 'build/nn/books', 'pins': 'train/data/books.json', 'minWords': 3, 'maxWords': 40},
        'coverage': {'gate': 2, 'context': 1, 'other': 1},
        'split': {'test': 5, 'val': 5},
        'checks': {'maxPhoneError': 0.35, 'truncatedTail': 3, 'truncatedAlways': 5, 'truncatedSilence': 0.05, 'maxClipped': 0.005},
        'analysis': {'formantCeilings': {'female': 5500, 'male': 5000}, 'bandwidth': 12000},
        **overrides
    }


# A tiny training speech corpus in a temporary downloads/tts/ - the texts' clips ([(key, text, split)]: a selection,
# each clip's synthetic 24 kHz WAV, and, when aligned, its alignment record from its phones' synthetic labels); yields
# (the corpus directory, the settings, the clips) with tts.TTS_DIR and tts.load_settings patched
@contextlib.contextmanager
def tts_corpus(texts, settings=None, aligned=True):
    settings = settings or tts_settings()
    with tempfile.TemporaryDirectory() as root, \
            unittest.mock.patch.object(tts, 'TTS_DIR', os.path.join(root, 'tts')), \
            unittest.mock.patch.object(tts, 'load_settings', return_value=settings):
        directory = tts.corpus_dir(settings)
        clips = []
        records = []
        for key, text, split in texts:
            voice = settings['voices'][key]
            clip = {'id': tts.clip_id(voice['voice'], text), 'key': key, 'voice': voice['voice'], 'sex': voice['sex'],
                    'book': 'pg1.txt', 'text': text, 'split': split, 'coverage': False}
            clips.append(clip)
            labels, seconds = clip_labels(text)
            samples = data_fixtures.labels_audio(labels, seconds, 24000)
            os.makedirs(os.path.dirname(tts.wav_path(directory, clip)), exist_ok=True)
            write_wav(tts.wav_path(directory, clip), samples, 24000)
            tokens = frontend.tokenize(text)['tokens']
            records.append({'id': clip['id'], 'aligner': tts.aligner_key(), 'keys': tts.phone_keys(tokens), 'tokens': tokens,
                            'seconds': round(seconds, 4), 'labels': labels, 'confidences': [0.9] * (len(labels) - 2), 'score': -1.0,
                            'greedy': greedy_of(text)})
        os.makedirs(directory, exist_ok=True)
        recipe = {'context': {'labels': {'texts': []}, 'gate': {'words': []}}, 'distill': {'targets': {'texts': []}}}
        with open(os.path.join(directory, f'selection-{selection_key(settings, recipe)}.json'), 'w', encoding='utf-8') as file:
            json.dump({'engine': settings['engine'], 'clips': clips}, file)
        if aligned:
            with open(os.path.join(directory, 'alignments.jsonl'), 'w', encoding='utf-8') as file:
                file.write(''.join(json.dumps(record) + '\n' for record in records))
        with unittest.mock.patch.object(tts, 'selection_key', return_value=selection_key(settings, recipe)):
            yield directory, settings, clips


# The fixture's selection key - stands for tts.selection_key (the corpus's selection is the fixture's)
def selection_key(settings, recipe):
    return f'{len(settings["voices"])}{len(recipe)}'
