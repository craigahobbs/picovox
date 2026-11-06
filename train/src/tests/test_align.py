# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import unittest

from picovox_train import align


class TestAlign(unittest.TestCase):

    def test_token_key(self):
        self.assertEqual(align.token_key('AH0'), 'AX')
        self.assertEqual(align.token_key('AH1'), 'AH')
        self.assertEqual(align.token_key('IY2'), 'IY')
        self.assertEqual(align.token_key('B'), 'B')

    def test_mean_durations(self):
        labels = [('pau', 0.0, 0.2), ('K', 0.2, 0.3), ('AH0', 0.3, 0.35), ('K', 0.35, 0.4), ('AH1', 0.4, 0.5), ('pau', 0.5, 0.9)]
        means = align.mean_durations(labels)
        self.assertEqual(sorted(means), ['AH', 'AX', 'K'])
        self.assertAlmostEqual(means['K'], 0.075)
        self.assertAlmostEqual(means['AX'], 0.05)
        self.assertAlmostEqual(means['AH'], 0.1)
        self.assertEqual(align.mean_durations([]), {})

    def test_label_durations(self):
        # The leading pause to <bos>, each phone its label's, the boundary between none, the trailing pause to the end
        tokens = ['<bos>', 'K', 'AE1', 'T', '_', 'S', 'AH0', '<period>']
        labels = [('pau', 0.0, 0.3), ('K', 0.3, 0.35), ('AE1', 0.35, 0.47), ('T', 0.47, 0.53), ('S', 0.53, 0.63),
                  ('AH0', 0.63, 0.7), ('pau', 0.7, 1.1)]
        durations = align.label_durations(tokens, labels, align.mean_durations(labels))
        self.assertEqual(durations[4], 0.0)
        for duration, expected in zip(durations, [0.3, 0.05, 0.12, 0.06, 0.0, 0.1, 0.07, 0.4]):
            self.assertAlmostEqual(duration, expected)

        # No edge pauses
        tokens = ['<bos>', 'S', 'AH1', 'P', '<period>']
        labels = [('S', 0.0, 0.1), ('AH1', 0.1, 0.2), ('P', 0.2, 0.3)]
        durations = align.label_durations(tokens, labels, align.mean_durations(labels))
        self.assertEqual((durations[0], durations[-1]), (0.0, 0.0))
        self.assertAlmostEqual(sum(durations), 0.3)

        # A phone's time multiplied and divided by its mean - its float rounding (0.13 x 0.11 / 0.11 isn't 0.13)
        durations = align.label_durations(['<bos>', 'K', '<period>'], [('K', 0.0, 0.13)], {'K': 0.11})
        self.assertEqual(durations[1], 0.13 * 0.11 / 0.11)

    def test_durations_to_frames(self):
        # The boundaries round, the total is kept
        self.assertEqual(align.durations_to_frames([0.014, 0.014, 0.014], 0.01, ['<bos>', 'K', '<period>']), [1, 2, 1])

        # A phone with no frames takes one from its longest neighbor phone
        self.assertEqual(align.durations_to_frames([0.0, 0.05, 0.001, 0.03, 0.0], 0.01, ['<bos>', 'K', 'AE1', 'T', '<period>']),
                         [0, 4, 1, 3, 0])

        # ... else from the longest phone anywhere
        self.assertEqual(align.durations_to_frames([0.0, 0.01, 0.0, 0.0, 0.0, 0.05, 0.0], 0.01,
                                                   ['<bos>', 'K', 'AE1', '_', '_', 'T', '<period>']), [0, 1, 1, 0, 0, 4, 0])

        # ... else it stays empty
        self.assertEqual(align.durations_to_frames([0.0, 0.01, 0.0, 0.0], 0.01, ['<bos>', 'K', 'AE1', '<period>']), [0, 1, 0, 0])

    def test_refine_pauses(self):
        tokens = ['<bos>', 'K', 'AE1', 'T', '_', 'S', 'AE1', 'T', '<comma>', 'G', 'OW1', '<period>']
        seconds = [0.2, 0.1, 0.1, 0.3, 0.0, 0.1, 0.1, 0.2, 0.0, 0.1, 0.1, 0.2]
        # A long silence inside the T before the word boundary, a short one at the comma (inside T and G), and one too
        # short to count
        runs = [(0.0, 0.2), (0.45, 0.7), (1.05, 1.16), (1.4, 1.42)]
        refined = align.refine_pauses(tokens, seconds, runs)
        self.assertAlmostEqual(sum(refined), sum(seconds))
        self.assertGreater(refined[4], 0.2)
        self.assertGreater(refined[8], 0.05)
        self.assertGreaterEqual(refined[3], align.KEEP_STOP - 1e-9)
        self.assertEqual(refined[0], 0.2)
        self.assertEqual([index for index, (old, new) in enumerate(zip(seconds, refined)) if abs(old - new) > 1e-9], [3, 4, 7, 8, 9])

        # No silences - unchanged
        unchanged = [right - left for left, right in zip([0.0] + list(_cumulative(seconds))[:-1], _cumulative(seconds))]
        self.assertEqual(align.refine_pauses(tokens, seconds, []), unchanged)

        # A silence that only covers the boundary itself - nothing moves
        self.assertEqual(align.refine_pauses(['<bos>', 'K', '<period>'], [0.2, 0.1, 0.3], [(0.0, 0.2), (0.3, 0.6)]),
                         align.refine_pauses(['<bos>', 'K', '<period>'], [0.2, 0.1, 0.3], []))


# Running sums
def _cumulative(values):
    total = 0.0
    for value in values:
        total += value
        yield total
