# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import os
import re
import tomllib
import unittest

from picovox_train import ROOT, SOURCE_DIR


class TestPackage(unittest.TestCase):

    def test_root(self):
        self.assertTrue(os.path.isfile(os.path.join(ROOT, 'data', 'nn', 'train.json')))
        self.assertEqual(SOURCE_DIR, os.path.join(ROOT, 'train', 'src'))

    def test_dependencies(self):
        # The package's dependencies are pinned as the training environment (requirements/nn.txt) pins them
        with open(os.path.join(ROOT, 'train', 'pyproject.toml'), 'rb') as file:
            dependencies = tomllib.load(file)['project']['dependencies']
        with open(os.path.join(ROOT, 'requirements', 'nn.txt'), encoding='utf-8') as file:
            pins = dict(re.findall(r'^([A-Za-z0-9_.-]+)==(\S+)$', file.read(), re.MULTILINE))
        self.assertTrue(dependencies)
        for dependency in dependencies:
            name, version = re.fullmatch(r'(\S+) == (\S+)', dependency).groups()
            self.assertEqual(pins.get(name), version, name)
