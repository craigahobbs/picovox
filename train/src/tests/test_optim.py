# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import copy
import math
import unittest

import torch

from picovox_train import optim
from picovox_train.model import MultiModel

from . import helpers


# A small model of Linear layers (a hidden matrix taller than wide, one wider than tall) and a bias-only parameter
class Small(torch.nn.Module):

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.tall = torch.nn.Linear(3, 5)
        self.wide = torch.nn.Linear(5, 2)
        self.scale = torch.nn.Parameter(torch.ones(2))

    def forward(self, x):
        return self.wide(torch.tanh(self.tall(x))) * self.scale


# The gradients of a fixed loss of a model (scaled so the clip binds, or not)
def backward(model, scale=1.0, seed=1):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(7, 3, generator=generator)
    model.zero_grad()
    (scale * model(inputs).pow(2).sum()).backward()


class TestSchedule(unittest.TestCase):

    def test_cosine(self):
        # The default's arithmetic - the warmup, the cosine to the floor, the floor past the end
        self.assertEqual(optim.lr_factor(0, 100, 10), 0.1)
        self.assertEqual(optim.lr_factor(9, 100, 10), 1.0)
        self.assertEqual(optim.lr_factor(10, 100, 10), 1.0)
        self.assertAlmostEqual(optim.lr_factor(55, 100, 10), 0.525)
        self.assertAlmostEqual(optim.lr_factor(100, 100, 10), optim.LR_FLOOR)
        self.assertAlmostEqual(optim.lr_factor(200, 100, 10), optim.LR_FLOOR)
        self.assertEqual(optim.lr_factor(5, 5, 5), 0.05 + 0.95 * 0.5 * (1 + math.cos(0.0)))

    def test_wsd(self):
        # Constant after the warmup, then the last quarter cooled down linearly or by 1 - sqrt
        self.assertEqual(optim.lr_factor(0, 100, 10, 'wsd', 0.25), 0.1)
        self.assertEqual(optim.lr_factor(74, 100, 10, 'wsd', 0.25), 1.0)
        self.assertEqual(optim.lr_factor(75, 100, 10, 'wsd', 0.25), 1.0)
        self.assertAlmostEqual(optim.lr_factor(87.5, 100, 10, 'wsd', 0.25), 0.525)
        self.assertAlmostEqual(optim.lr_factor(100, 100, 10, 'wsd', 0.25), optim.LR_FLOOR)
        self.assertAlmostEqual(optim.lr_factor(150, 100, 10, 'wsd', 0.25), optim.LR_FLOOR)
        self.assertAlmostEqual(optim.lr_factor(81.25, 100, 10, 'wsd', 0.25, '1-sqrt'), 0.05 + 0.95 * 0.5)
        self.assertAlmostEqual(optim.lr_factor(100, 100, 10, 'wsd', 0.25, '1-sqrt'), optim.LR_FLOOR)
        # (no cooldown - constant to the end)
        self.assertEqual(optim.lr_factor(100, 100, 10, 'wsd'), 1.0)

    def test_unknown(self):
        with self.assertRaises(ValueError):
            optim.lr_factor(50, 100, 10, 'step')


class TestModelParameters(unittest.TestCase):

    def test_zero_init_projections(self):
        model = MultiModel(helpers.tiny_config())
        count = optim.zero_init_projections(model)
        # (each encoder and decoder layer's conv-FFN, and each encoder layer's attention)
        self.assertEqual(count, 3)
        for module in model.modules():
            for name in ('outer', 'out'):
                projection = getattr(module, name, None)
                if isinstance(projection, torch.nn.Linear):
                    self.assertEqual(projection.weight.abs().sum().item(), 0.0)
                    self.assertEqual(projection.bias.abs().sum().item(), 0.0)
        # (every block starts as the identity - the model still runs)
        self.assertGreater(model.head.weight.abs().sum().item(), 0.0)

    def test_hidden_matrices(self):
        model = MultiModel(helpers.tiny_config())
        model.decoder_in.weight.requires_grad_(False)
        matrices = optim.hidden_matrices(model)
        identities = {id(matrix) for matrix in matrices}
        for excluded in (model.head.weight, model.duration_out.weight, model.position_in.weight, model.decoder_in.weight):
            self.assertNotIn(id(excluded), identities)
        self.assertTrue(all(matrix.dim() == 2 for matrix in matrices))
        self.assertIn(id(model.condition_decoder.weight), identities)

    def test_param_groups(self):
        model = MultiModel(helpers.tiny_config())
        extra = torch.nn.Parameter(torch.zeros(3))
        groups = optim.param_groups(model, [extra])
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]['params']), len(list(model.parameters())) + 1)
        groups = optim.param_groups(model, [extra], muon=True)
        self.assertEqual(len(groups), 2)
        self.assertTrue(groups[1]['muon'])
        self.assertEqual({id(parameter) for parameter in groups[1]['params']},
                         {id(matrix) for matrix in optim.hidden_matrices(model)})
        self.assertIn(id(extra), {id(parameter) for parameter in groups[0]['params']})
        self.assertEqual(len(groups[0]['params']) + len(groups[1]['params']), len(list(model.parameters())) + 1)


class TestNewtonSchulz(unittest.TestCase):

    def test_orthogonalized(self):
        # The singular values pushed toward 1 (the quintic's band, about 0.7-1.2)
        generator = torch.Generator().manual_seed(0)
        matrix = torch.randn(1, 4, 6, generator=generator)
        singular = torch.linalg.svdvals(optim.newton_schulz(matrix)[0])
        self.assertTrue(torch.all(singular > 0.5) and torch.all(singular < 1.3))

    def test_zero_padding(self):
        # A matrix zero-padded into a larger batch gets the same update in its block, and zeros outside it
        generator = torch.Generator().manual_seed(1)
        matrix = torch.randn(3, 5, generator=generator)
        padded = torch.zeros(2, 4, 7)
        padded[0, :3, :5] = matrix
        padded[1] = torch.randn(4, 7, generator=generator)
        result = optim.newton_schulz(padded)
        self.assertTrue(torch.allclose(result[0, :3, :5], optim.newton_schulz(matrix[None])[0], atol=1e-6))
        self.assertEqual(result[0, 3:].abs().sum().item(), 0.0)
        self.assertEqual(result[0, :, 5:].abs().sum().item(), 0.0)


class TestLeanOptimizer(unittest.TestCase):

    def test_matches_adamw(self):
        # The lean step and the default (clip_grad_norm_, then AdamW) agree up to rounding - the clip binding or not
        for scale in (100.0, 0.001):
            lean_model, default_model = Small(), Small()
            lean = optim.LeanOptimizer(optim.param_groups(lean_model), lr=0.01, weight_decay=0.1)
            default = torch.optim.AdamW(default_model.parameters(), lr=0.01, weight_decay=0.1, betas=(0.9, 0.98))
            for seed in range(3):
                backward(lean_model, scale, seed)
                backward(default_model, scale, seed)
                norm = lean.step()
                default_norm = torch.nn.utils.clip_grad_norm_(default_model.parameters(), 1.0)
                default.step()
                self.assertTrue(torch.allclose(norm, default_norm, rtol=1e-6))
            for lean_parameter, default_parameter in zip(lean_model.parameters(), default_model.parameters()):
                self.assertTrue(torch.allclose(lean_parameter, default_parameter, rtol=1e-5, atol=1e-7))
            # (one step counter shared by the group's parameters)
            steps = {id(lean.state[parameter]['step']) for parameter in lean_model.parameters()}
            self.assertEqual(len(steps), 1)
            self.assertEqual(lean.state[lean_model.scale]['step'].item(), 3.0)

    def test_grad_norm_rows(self):
        # The norm of exactly whole rows (no padding), and of gradients with a parameter without one
        weight = torch.nn.Parameter(torch.zeros(2, optim.NORM_ROW))
        unused = torch.nn.Parameter(torch.zeros(3))
        lean = optim.LeanOptimizer([{'params': [weight, unused]}], lr=0.1)
        weight.grad = torch.full_like(weight, 0.5)
        self.assertAlmostEqual(lean.grad_norm().item(), math.sqrt(2 * optim.NORM_ROW * 0.25), places=4)
        lean.step()
        self.assertNotIn('exp_avg', lean.state[unused])
        self.assertIn('exp_avg', lean.state[weight])
        # (a group without gradients takes no step)
        empty = optim.LeanOptimizer([{'params': [weight]}, {'params': [unused]}], lr=0.1)
        empty.step()
        self.assertEqual(dict(empty.state[unused]), {})

    def test_nonfinite_skipped(self):
        # A non-finite gradient norm skips the step on the device - parameters, moments, and the step count unchanged
        for muon in (False, True):
            model = Small()
            lean = optim.LeanOptimizer([{'params': [model.scale, model.tall.bias, model.wide.bias]},
                                        {'params': [model.tall.weight, model.wide.weight], 'muon': muon}], lr=0.01)
            backward(model)
            lean.step()
            before = [parameter.detach().clone() for parameter in model.parameters()]
            backward(model)
            model.wide.weight.grad[0, 0] = math.inf
            lean.step()
            for parameter, previous in zip(model.parameters(), before):
                self.assertTrue(torch.equal(parameter, previous))
            self.assertEqual(lean.state[model.scale]['step'].item(), 1.0)

    def test_muon(self):
        # Muon's update - orthogonalized, scaled to AdamW's RMS - of the matrices in either orientation, one without a
        # gradient; the momentum warmed up
        model = Small()
        groups = [{'params': [model.scale, model.tall.bias, model.wide.bias]},
                  {'params': [model.tall.weight, model.wide.weight], 'muon': True}]
        lean = optim.LeanOptimizer(groups, lr=0.01, weight_decay=0.0)
        backward(model)
        before = model.tall.weight.detach().clone()
        wide_before = model.wide.weight.detach().clone()
        model.wide.weight.grad = None
        lean.step()
        update = (model.tall.weight - before) / (-0.01 * 0.2 * math.sqrt(5))
        singular = torch.linalg.svdvals(update)
        self.assertTrue(torch.all(singular > 0.5) and torch.all(singular < 1.3))
        self.assertTrue(torch.equal(model.wide.weight, wide_before))
        layout = lean.layouts[1]
        self.assertEqual(layout['transposed'], [True, False])
        self.assertEqual(layout['shape'], (2, 3, 5))
        self.assertEqual(lean.state[model.tall.weight]['muonStep'], 1)
        # (the second step reuses the layout and the momentum buffer)
        backward(model)
        lean.step()
        self.assertEqual(lean.state[model.tall.weight]['muonStep'], 2)
        self.assertEqual(lean.state[model.tall.weight]['momentum_buffer'].shape, (2, 3, 5))

    def test_weight_decay(self):
        # Muon's decoupled weight decay - a zero gradient's update is the decay alone
        weight = torch.nn.Parameter(torch.ones(2, 3))
        lean = optim.LeanOptimizer([{'params': [weight], 'muon': True}], lr=0.1, weight_decay=0.5)
        weight.grad = torch.zeros_like(weight)
        lean.step()
        self.assertTrue(torch.allclose(weight, torch.full_like(weight, 0.95)))

    def test_load_state_dict(self):
        # A saved and loaded optimizer shares its step counters again, and continues identically
        model, twin = Small(), Small()
        groups = [{'params': [model.scale, model.tall.bias, model.wide.bias]},
                  {'params': [model.tall.weight, model.wide.weight], 'muon': True}]
        lean = optim.LeanOptimizer(groups, lr=0.01)
        backward(model)
        lean.step()
        twin.load_state_dict(model.state_dict())
        loaded = optim.LeanOptimizer([{'params': [twin.scale, twin.tall.bias, twin.wide.bias]},
                                      {'params': [twin.tall.weight, twin.wide.weight], 'muon': True}], lr=0.01)
        # (a copy, as a checkpoint's - loading keeps the tensors it's given)
        loaded.load_state_dict(copy.deepcopy(lean.state_dict()))
        self.assertIs(loaded.state[twin.wide.bias]['step'], loaded.state[twin.scale]['step'])
        for optimizer, net in ((lean, model), (loaded, twin)):
            backward(net, seed=2)
            optimizer.step()
        for parameter, other in zip(model.parameters(), twin.parameters()):
            self.assertTrue(torch.equal(parameter, other))

    def test_load_state_dict_partial(self):
        # A loaded state without step counts (a parameter never stepped), and an empty group
        model = Small()
        lean = optim.LeanOptimizer([{'params': [model.scale, model.tall.bias]}, {'params': []}], lr=0.01)
        model.scale.grad = torch.ones_like(model.scale)
        lean.step()
        state = lean.state_dict()
        loaded = optim.LeanOptimizer([{'params': [model.tall.bias, model.scale]}, {'params': []}], lr=0.01)
        loaded.load_state_dict(state)
        # (the state is by position - the stepped one's is the first parameter's now)
        self.assertIn('step', loaded.state[model.tall.bias])
        self.assertNotIn('step', loaded.state[model.scale])


class TestFlatEMA(unittest.TestCase):

    def test_ema(self):
        model, shadow = Small(), Small()
        ema = optim.FlatEMA(model.parameters(), 0.5)
        initial = [parameter.detach().clone() for parameter in model.parameters()]
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(2.0)
        ema.update()
        ema.copy_to(shadow.parameters())
        for parameter, start in zip(shadow.parameters(), initial):
            self.assertTrue(torch.allclose(parameter, start + 1.0))
        # (its state saved and loaded)
        other = optim.FlatEMA(Small().parameters(), 0.5)
        other.load_state_dict(ema.state_dict())
        self.assertTrue(torch.equal(other.shadow, ema.shadow))
        self.assertEqual(ema.state_dict()['decay'], 0.5)
