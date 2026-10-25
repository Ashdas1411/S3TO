import unittest

import numpy as np

from s3to.grouping import (
    _unique_partitions,
    assignment_from_groups,
    choose_complementary_groups,
)


class TestS3TOGrouping(unittest.TestCase):
    def setUp(self):
        self.histograms = {
            0: [50, 50, 0, 0, 0, 0],
            1: [0, 0, 50, 50, 0, 0],
            2: [0, 0, 0, 0, 50, 50],
            3: [25, 25, 25, 25, 0, 0],
            4: [0, 0, 25, 25, 25, 25],
            5: [25, 0, 0, 25, 25, 25],
        }

    def test_six_clients_generate_ten_unique_partitions(self):
        partitions = list(_unique_partitions(range(6), 3))
        self.assertEqual(len(partitions), 10)
        self.assertEqual(len({str(p) for p in partitions}), 10)

    def test_nine_clients_generate_all_valid_partitions(self):
        partitions = list(_unique_partitions(range(9), 3))
        self.assertEqual(len(partitions), 280)

        for groups in partitions:
            self.assertEqual(len(groups), 3)
            self.assertTrue(all(len(group) == 3 for group in groups))
            flattened = [client for group in groups for client in group]
            self.assertEqual(sorted(flattened), list(range(9)))

    def test_assignment_covers_every_client_once(self):
        groups, scores, pairwise = choose_complementary_groups(
            self.histograms, group_size=3
        )
        assignment = assignment_from_groups(groups)

        self.assertEqual(set(assignment), set(self.histograms))
        self.assertEqual(
            sorted(assignment.values()).count("edge1"), 3
        )
        self.assertEqual(
            sorted(assignment.values()).count("edge2"), 3
        )
        self.assertEqual(len(scores), 2)
        self.assertEqual(set(pairwise), set(self.histograms))

    def test_result_is_deterministic(self):
        first = choose_complementary_groups(self.histograms, 3)
        second = choose_complementary_groups(self.histograms, 3)
        self.assertEqual(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])

    def test_rejects_empty_histogram(self):
        data = dict(self.histograms)
        data[2] = [0, 0, 0, 0, 0, 0]
        with self.assertRaises(ValueError):
            choose_complementary_groups(data, 3)

    def test_rejects_negative_counts(self):
        data = dict(self.histograms)
        data[2] = [-1, 2, 3, 4, 5, 6]
        with self.assertRaises(ValueError):
            choose_complementary_groups(data, 3)

    def test_rejects_nonfinite_counts(self):
        data = dict(self.histograms)
        data[2] = [1, 2, 3, 4, float("nan"), 6]
        with self.assertRaises(ValueError):
            choose_complementary_groups(data, 3)

    def test_rejects_mismatched_histogram_lengths(self):
        data = dict(self.histograms)
        data[2] = [1, 2]
        with self.assertRaises(ValueError):
            choose_complementary_groups(data, 3)

    def test_rejects_invalid_group_size(self):
        with self.assertRaises(ValueError):
            choose_complementary_groups(self.histograms, 0)

    def test_rejects_undividable_client_count(self):
        with self.assertRaises(ValueError):
            choose_complementary_groups(
                {0: [1, 0], 1: [0, 1], 2: [1, 1], 3: [2, 1]},
                group_size=3,
            )


if __name__ == "__main__":
    unittest.main()
