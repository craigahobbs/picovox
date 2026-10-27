# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The step-0 check - the multi-speaker trainer's losses on fixed batches, from a fixed checkpoint, written exactly (as
float hex strings), so two versions of the code can be compared bit for bit. It imports the package from the current
directory's train/src, so the same file checks another tree (run it with that tree as the working directory):

    build/venv-nn/bin/python -B train/src/picovox_train/stepcheck.py --init build/nn/runs/multi-a/step-15000.pt --out a.json
    (cd ../baseline && ../worktree/build/venv-nn/bin/python -B ../worktree/train/src/picovox_train/stepcheck.py ... --out b.json)

Each batch is a few validation utterances (readers, and the emotional corpus when it is built) with fixed crops and
noise seeds; the model runs in eval mode (no dropout) and in train mode (dropout, seeded). The losses are trainmulti.py's
batch_losses with its default settings, and the gradient norm of the train-mode total.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'train', 'src'))

# pylint: disable=wrong-import-position
import numpy  # noqa: E402
import torch  # noqa: E402

from picovox_train import dsynth, trainmulti  # noqa: E402
from picovox_train.trainmulti import MULTI_PRIORS, Corpus, batch_losses, collate, default_settings, sonorant_mask  # noqa: E402
from picovox_train.model import MultiModel  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(prog='stepcheck.py', description='Step-0 losses on fixed batches, exactly')
    parser.add_argument('--init', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batches', type=int, default=3)
    parser.add_argument('--batch', type=int, default=3)
    parser.add_argument('--config', default='{}', help='model config overrides (JSON) - for checking an option on')
    args = parser.parse_args(argv)
    settings = default_settings()
    device = torch.device(args.device)
    state = torch.load(args.init, map_location='cpu', weights_only=False)
    config = {**state['config'], **json.loads(args.config)}
    corpora = [Corpus('multi', 'val', max_items=64)]
    if os.path.exists(os.path.join(trainmulti.DATA_DIR, 'cremad', 'index.json')):
        corpora.append(Corpus('cremad', 'val', speaker_offset=state['readerCount'], max_items=32))
    sonorants = torch.tensor(sonorant_mask(corpora[0].data.index['vocab']), device=device)
    synth = dsynth.DSynth(settings.rate).to(device)
    out = {'init': args.init, 'device': args.device, 'config': json.loads(args.config), 'batches': []}
    for corpus in corpora:
        order = sorted(range(len(corpus)), key=lambda index, items=corpus.items: items[index]['frameCount'])
        for number in range(args.batches):
            indices = order[number * 7:number * 7 + args.batch]
            row = {'corpus': corpus.name, 'items': [corpus.items[index]['id'] for index in indices]}
            for mode in ('eval', 'train'):
                torch.manual_seed(1)
                model = MultiModel(config, priors=MULTI_PRIORS).to(device)
                model.load_state_dict(state['model'], strict=False)
                model.train(mode == 'train')
                batch = collate(corpus, indices, device)
                rng = numpy.random.default_rng(1234 + number)
                losses, _, _, stats = batch_losses(model, synth, batch, settings, rng, 99 + number, 0.1, sonorants)
                row[mode] = {key: float(value.detach()).hex() for key, value in {**losses, **stats}.items()}
                if mode == 'train':
                    model.zero_grad()
                    losses['total'].backward()
                    norm = torch.sqrt(sum((parameter.grad.cpu().double() ** 2).sum() for parameter in model.parameters()
                                          if parameter.grad is not None))
                    row['gradNorm'] = float(norm).hex()
            out['batches'].append(row)
            print(json.dumps({'corpus': corpus.name, 'batch': number, 'spectral': float.fromhex(row['eval']['spectral']),
                              'total': float.fromhex(row['eval']['total'])}), file=sys.stderr, flush=True)
    with open(args.out, 'w', encoding='utf-8') as file:
        json.dump(out, file, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
