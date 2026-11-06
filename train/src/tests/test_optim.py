# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import copy
import math
import unittest

import torch

from picovox_train import optim


# A small model of two Linear layers and a bias-only parameter
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
        # Constant after the warmup, then the last quarter cooled down by 1 - sqrt
        self.assertEqual(optim.lr_factor(0, 100, 10, 'wsd', 0.25), 0.1)
        self.assertEqual(optim.lr_factor(74, 100, 10, 'wsd', 0.25), 1.0)
        self.assertEqual(optim.lr_factor(75, 100, 10, 'wsd', 0.25), 1.0)
        self.assertAlmostEqual(optim.lr_factor(81.25, 100, 10, 'wsd', 0.25), 0.05 + 0.95 * 0.5)
        self.assertAlmostEqual(optim.lr_factor(100, 100, 10, 'wsd', 0.25), optim.LR_FLOOR)
        self.assertAlmostEqual(optim.lr_factor(150, 100, 10, 'wsd', 0.25), optim.LR_FLOOR)
        # (no cooldown - constant to the end)
        self.assertEqual(optim.lr_factor(100, 100, 10, 'wsd'), 1.0)


class TestLeanOptimizer(unittest.TestCase):

    def test_matches_adamw(self):
        # The lean step and the default (clip_grad_norm_, then AdamW) agree up to rounding - the clip binding or not
        for scale in (100.0, 0.001):
            lean_model, default_model = Small(), Small()
            lean = optim.LeanOptimizer([{'params': list(lean_model.parameters())}], lr=0.01, weight_decay=0.1)
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
        model = Small()
        lean = optim.LeanOptimizer([{'params': [model.scale, model.tall.bias, model.wide.bias]},
                                    {'params': [model.tall.weight, model.wide.weight]}], lr=0.01)
        backward(model)
        lean.step()
        before = [parameter.detach().clone() for parameter in model.parameters()]
        backward(model)
        model.wide.weight.grad[0, 0] = math.inf
        lean.step()
        for parameter, previous in zip(model.parameters(), before):
            self.assertTrue(torch.equal(parameter, previous))
        self.assertEqual(lean.state[model.scale]['step'].item(), 1.0)

    def test_load_state_dict(self):
        # A saved and loaded optimizer shares its step counters again, and continues identically
        model, twin = Small(), Small()
        groups = [{'params': [model.scale, model.tall.bias, model.wide.bias]},
                  {'params': [model.tall.weight, model.wide.weight]}]
        lean = optim.LeanOptimizer(groups, lr=0.01)
        backward(model)
        lean.step()
        twin.load_state_dict(model.state_dict())
        loaded = optim.LeanOptimizer([{'params': [twin.scale, twin.tall.bias, twin.wide.bias]},
                                      {'params': [twin.tall.weight, twin.wide.weight]}], lr=0.01)
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
