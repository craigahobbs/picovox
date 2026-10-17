# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import unittest
import unittest.mock

from . import helpers


class TestWaitUntil(unittest.TestCase):

    def test_wait_until(self):
        # True at once, true after polling, and false at the timeout
        self.assertTrue(helpers.wait_until(lambda: True))
        answers = iter([False, False, True])
        with unittest.mock.patch('time.sleep') as sleep:
            self.assertTrue(helpers.wait_until(lambda: next(answers)))
        self.assertEqual(sleep.call_count, 2)
        self.assertFalse(helpers.wait_until(lambda: False, timeout=0))
