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

    def test_label_key(self):
        for label in ('', ' ', 'pau', 'sil', 'sp', 'ssil', 'h#'):
            self.assertEqual(align.label_key(label), align.PAUSE)
        self.assertEqual(align.label_key('spn'), 'SPN')
        self.assertEqual(align.label_key('ax'), 'AX')
        self.assertEqual(align.label_key('AH0'), 'AX')
        self.assertEqual(align.label_key('AH1'), 'AH')
        self.assertEqual(align.label_key('iy'), 'IY')
        self.assertEqual(align.label_key('ER2'), 'ER')

    def test_is_vowel_key(self):
        self.assertTrue(align.is_vowel_key('AX'))
        self.assertTrue(align.is_vowel_key('IY'))
        self.assertFalse(align.is_vowel_key('B'))

    def test_substitution_cost(self):
        self.assertEqual(align.substitution_cost('B', 'B'), 0.0)
        self.assertEqual(align.substitution_cost('B', 'SPN'), align.COST_CONSONANT)
        self.assertEqual(align.substitution_cost('AH', 'AX'), align.COST_NEAR)
        self.assertEqual(align.substitution_cost('AX', 'AH'), align.COST_NEAR)
        self.assertEqual(align.substitution_cost('IY', 'IH'), align.COST_VOWEL)
        self.assertEqual(align.substitution_cost('P', 'B'), align.COST_CLOSE)
        self.assertEqual(align.substitution_cost('P', 'K'), align.COST_CONSONANT)
        self.assertEqual(align.substitution_cost('IY', 'K'), align.COST_CROSS)

    def test_align_sequences(self):
        # An exact pairing, the boundary paired with a pause
        path, cost = align.align_sequences(['K', 'AE1', 'T', '_', 'S', 'AE1', 'T'], ['K', 'AE', 'T', 'PAU', 'S', 'AE', 'T'])
        self.assertEqual(cost, 0.0)
        self.assertEqual(path, [(index, index) for index in range(7)])

        # A boundary without a pause, a deletion, and an insertion
        path, cost = align.align_sequences(['K', 'AE1', '_', 'S'], ['K', 'AE', 'S', 'IY'])
        self.assertEqual(cost, align.COST_INDEL)
        self.assertEqual(path, [(0, 0), (1, 1), (2, None), (3, 2), (None, 3)])

        # A pause inside a word
        path, cost = align.align_sequences(['K', 'AE1'], ['K', 'PAU', 'AE'])
        self.assertEqual(cost, align.COST_PAUSE_INSIDE)
        self.assertEqual(path, [(0, 0), (None, 1), (1, 2)])

        # A phone missing from the labels
        path, cost = align.align_sequences(['K', 'AE1', 'T'], ['K', 'AE'])
        self.assertEqual(cost, align.COST_INDEL)
        self.assertEqual(path, [(0, 0), (1, 1), (2, None)])

    def test_label_segments(self):
        self.assertEqual(align.label_segments([('pau', 0.1), ('sil', 0.2), ('k', 0.05), ('', 0.1)]),
                         [(align.PAUSE, 0.1 + 0.2), ('K', 0.05), (align.PAUSE, 0.1)])

    def test_assign_durations(self):
        tokens = ['<bos>', 'K', 'AE1', 'T', '_', 'S', 'AH0', '<period>']
        labels = [('pau', 0.3), ('k', 0.05), ('ae', 0.12), ('t', 0.06), ('pau', 0.2), ('s', 0.1), ('ax', 0.07), ('pau', 0.4)]
        durations, report = align.assign_durations(tokens, labels, {})
        for duration, expected in zip(durations, [0.3, 0.05, 0.12, 0.06, 0.2, 0.1, 0.07, 0.4]):
            self.assertAlmostEqual(duration, expected)
        self.assertEqual(report['phones'], 5)
        self.assertEqual(report['exact'], 5)
        self.assertEqual(report['pauseAtBoundary'], 1)
        self.assertEqual(report['status'], [None, 'exact', 'exact', 'exact', None, 'exact', 'exact', None])

    def test_assign_durations_no_edges(self):
        # No leading or trailing silence; a near match (AH for AX) and a substitution
        tokens = ['<bos>', 'S', 'AH1', 'P', '<period>']
        durations, report = align.assign_durations(tokens, [('s', 0.1), ('ax', 0.1), ('b', 0.1)], {})
        self.assertEqual(durations[0], 0.0)
        self.assertEqual(durations[-1], 0.0)
        self.assertEqual(report['near'], 1)
        self.assertEqual(report['substituted'], 1)
        self.assertEqual(report['edits'], ['P>B'])
        self.assertEqual(report['status'][1:4], ['exact', 'near', 'substituted'])

    def test_assign_durations_blocks(self):
        # A deleted phone joins the previous anchor (split by the typical durations); an inserted label joins it too; a
        # pause inside the word counts
        tokens = ['<bos>', 'K', 'AE1', 'T', '<period>']
        durations, report = align.assign_durations(tokens, [('k', 0.1), ('ae', 0.3), ('pau', 0.05), ('z', 0.04)],
                                                   {'AE': 0.1, 'T': 0.05})
        self.assertEqual(report['pauseInsideWord'], 1)
        self.assertEqual(report['substituted'], 1)
        self.assertEqual(len(durations), 5)
        self.assertAlmostEqual(sum(durations), 0.49)

        # Inserted label phones - before any token (the first phone's), and after the last
        durations, report = align.assign_durations(['<bos>', 'K', '<period>'], [('s', 0.05), ('k', 0.1), ('s', 0.05)], {})
        self.assertEqual(report['inserted'], 2)
        self.assertEqual(report['status'][1], 'exact+inserted')
        self.assertAlmostEqual(durations[1], 0.2)

        # A deleted phone with no anchor before it in its span joins the following anchor
        durations, report = align.assign_durations(['<bos>', 'T', 'K', '<period>'], [('k', 0.2)], {'T': 0.1, 'K': 0.1})
        self.assertEqual(report['deleted'], 1)
        self.assertEqual(report['edits'], ['T>-'])
        self.assertEqual(report['status'][1:3], ['deleted', 'exact'])
        self.assertAlmostEqual(durations[1], 0.1)
        self.assertAlmostEqual(durations[2], 0.1)

        # A deleted phone alone in its span joins the anchor before (or, at the start, after) the span
        durations, report = align.assign_durations(['<bos>', 'K', '_', 'T', '<period>'], [('k', 0.2)], {})
        self.assertEqual(report['deleted'], 1)
        self.assertAlmostEqual(durations[1] + durations[3], 0.2)
        durations, report = align.assign_durations(['<bos>', 'T', '_', 'K', '<period>'], [('k', 0.2)], {})
        self.assertEqual(report['deleted'], 1)
        self.assertAlmostEqual(durations[1] + durations[3], 0.2)

    def test_assign_durations_errors(self):
        with self.assertRaises(ValueError):
            align.assign_durations(['K', '<period>'], [('k', 0.1)], {})
        with self.assertRaises(ValueError):
            align.assign_durations(['<bos>', 'K'], [('k', 0.1)], {})
        with self.assertRaises(ValueError):
            align.assign_durations(['<bos>', 'K', '<bos>'], [('k', 0.1)], {})
        with self.assertRaisesRegex(ValueError, 'no paired phones'):
            align.assign_durations(['<bos>', '_', '<period>'], [('k', 0.1)], {})

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
        refined, changed, moved = align.refine_pauses(tokens, seconds, runs)
        self.assertEqual(changed, 2)
        self.assertAlmostEqual(sum(refined), sum(seconds))
        self.assertGreater(refined[4], 0.2)
        self.assertGreater(refined[8], 0.05)
        self.assertGreaterEqual(refined[3], align.KEEP_STOP - 1e-9)
        self.assertAlmostEqual(moved, refined[4] + refined[8])
        self.assertEqual(refined[0], 0.2)

        # No silences - unchanged
        refined, changed, moved = align.refine_pauses(tokens, seconds, [])
        self.assertEqual((changed, moved), (0, 0.0))
        self.assertEqual(refined, [right - left for left, right in zip([0.0] + list(_cumulative(seconds))[:-1], _cumulative(seconds))])

        # A silence that only covers the boundary itself - nothing moves
        refined, changed, _ = align.refine_pauses(['<bos>', 'K', '<period>'], [0.2, 0.1, 0.3], [(0.0, 0.2), (0.3, 0.6)])
        self.assertEqual(changed, 0)


# Running sums
def _cumulative(values):
    total = 0.0
    for value in values:
        total += value
        yield total
