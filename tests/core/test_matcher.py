import numpy as np
import tensorflow as tf
from kerod.core.matcher import Matcher, hungarian_matching


def test_matcher():
    similarity = np.array([
        [[1., 1, 1, 3, 1], [2, -1, 2, 0, 4]],
        [[1., 0.1, 1, 3, 0], [8, 0.4, 2, 0, 0.2]],
    ])
    num_valid_boxes = np.array([[2], [2]], np.int32)
    matcher = Matcher([0.3, 0.5], [0, -1, 1])

    matches, match_labels = matcher(similarity, num_valid_boxes)

    expected_matched = np.array([[1, 0, 1, 0, 1], [1, 1, 1, 0, 1]])
    expected_matched_labels = np.array([[1, 1, 1, 1, 1], [1, -1, 1, 1, 0]])

    np.testing.assert_array_equal(matches, expected_matched)
    np.testing.assert_array_equal(match_labels, expected_matched_labels)


def test_matcher_low_quality_matcher():

    num_valid_boxes = np.array([[3], [3]], np.int32)
    similarity = np.array([[
        [0,  0.2, 0.49, 0.1,  0.3],
        [2,   -1,  0.2,   4, 0.38],
        [1, 0.25,  0.3,   5, 0.37],
        # This line is not valid and should be discarded (num_valid_boxes = 3)
        [0,    0,    0,   0, 0.50],
    ], [
        [1, 0.3,   1,   3,    0],
        [8, 0.4,   2,   0,  0.2],
        [0,  -1, 0.2, 0.1, 0.39],
        # This line is not valid and should be discarded (num_valid_boxes = 3)
        [0,   0,   0,   0,    0],
    ]])  # yapf: disable

    matcher = Matcher([0.3, 0.5], [0, -1, 1], allow_low_quality_matches=True)

    matches, match_labels = matcher(similarity, num_valid_boxes)

    expected_matched = np.array([[1, 2, 0, 2, 3], [1, 1, 1, 0, 2]])
    # if allow_low_quality_matches was False
    # [[1, 0, -1, 1, -1], [1, -1, 1, 1, 0]]
    expected_matched_labels = np.array([[1, 0, 1, 1, -1], [1, -1, 1, 1, 1]])

    np.testing.assert_array_equal(matches, expected_matched)
    np.testing.assert_array_equal(match_labels, expected_matched_labels)


def test_matcher_low_quality_matcher_when_the_best_box_is_undefined():
    num_valid_boxes = np.array([[4], [3]], np.int32)
    similarity = np.array([[
        [0,  0.31,    0, 0,     0],
        [0.1,   0,    0, 0,     0],
        [0,     0, 0.32, 0,     0],
        [0,     0,    0, 0,  0.48],
    ], [
        [1, 0.3,   1,   3,    0],
        [8, 0.4,   2,   0,  0.2],
        [0,  -1, 0.2, 0.1, 0.39],
        [0,   0,   0,   0, 0.31],
    ]])  # yapf: disable

    expected_matched = np.array([[1, 0, 2, 0, 3], [1, 1, 1, 0, 2]])

    matcher = Matcher([0.3, 0.5], [0, -1, 1], allow_low_quality_matches=False)
    matches, match_labels = matcher(similarity, num_valid_boxes)
    expected_matched_labels = np.array([[0, -1, -1, 0, -1], [1, -1, 1, 1, -1]])
    np.testing.assert_array_equal(matches, expected_matched)
    np.testing.assert_array_equal(match_labels, expected_matched_labels)

    matcher = Matcher([0.3, 0.5], [0, -1, 1], allow_low_quality_matches=True)
    matches, match_labels = matcher(similarity, num_valid_boxes)
    # Explanation expactation for batch[0]
    #  0 -> 1 because anchor 0 has the highest IoU with gt 1 so it becomes a low quality match
    #  -1 -> 1 because anchor 1 has the highest IoU with gt 0 => low quality match
    #  -1 -> 1 because anchor 2 has the highest IoU with gt 2 => low quality match
    #  0 = 0 because anchor 3 isn't close enough to any groundtruths
    #  -1 -> 1 because anchor 4 has the highest IoU with gt 3 => low quality match
    expected_matched_labels = np.array([[1, 1, 1, 0, 1], [1, -1, 1, 1, 1]])
    np.testing.assert_array_equal(matches, expected_matched)
    np.testing.assert_array_equal(match_labels, expected_matched_labels)


def test_matcher_low_quality_matcher_with_one_ground_truth():
    num_valid_boxes = np.array([[1]], np.int32)
    similarity = np.array([[
        [0, 0.31, 0, 0, 0],
    ]])

    expected_matched = np.array([[0, 0, 0, 0, 0]])

    matcher = Matcher([0.3, 0.5], [0, -1, 1], allow_low_quality_matches=False)
    matches, match_labels = matcher(similarity, num_valid_boxes)
    expected_matched_labels = np.array([[0, -1, 0, 0, 0]])
    assert match_labels.shape == (1, 5)
    np.testing.assert_array_equal(matches, expected_matched)
    np.testing.assert_array_equal(match_labels, expected_matched_labels)

    matcher = Matcher([0.3, 0.5], [0, -1, 1], allow_low_quality_matches=True)
    matches, match_labels = matcher(similarity, num_valid_boxes)
    expected_matched_labels = np.array([[0, 1, 0, 0, 0]])
    assert match_labels.shape == (1, 5)
    np.testing.assert_array_equal(matches, expected_matched)
    np.testing.assert_array_equal(match_labels, expected_matched_labels)


def test_matcher_with_one_threshold():
    similarity = np.array([
        [[1., 1, 1, 3, 1], [2, -1, 2, 0, 4]],
        [[1., 0.1, 1, 3, 0], [8, 0.4, 2, 0, 0.2]],
    ])
    num_valid_boxes = np.array([[2], [2]], np.int32)
    matcher = Matcher([0.5], [0, 1])

    matches, match_labels = matcher(similarity, num_valid_boxes)

    expected_matched = np.array([[1, 0, 1, 0, 1], [1, 1, 1, 0, 1]])
    expected_matched_labels = np.array([[1, 1, 1, 1, 1], [1, 0, 1, 1, 0]])

    np.testing.assert_array_equal(matches, expected_matched)
    np.testing.assert_array_equal(match_labels, expected_matched_labels)


def test_hungarian_matcher_compute_cost_matrix():
    similarity = np.array([
        [
            [0., 1, 1, 3, -1],
            [2, -1, 2, 0, 4],
            [0.5, 2, 2.3, 0.25, 4],
        ],
        [
            [1., 0.15, 1, 0.3, 0.16],
            [0.2, 0.1, 2, 0.1, 0.2],
            [0.3, 0.4, 2, 0, 0.2],
        ],
    ])
    num_valid_boxes = np.array([[3], [1]], np.int32)
    matches, match_labels = hungarian_matching(similarity, num_valid_boxes)
    expected_matched = np.array([[0, 1, 0, 2, 0], [1, 0, 2, 0, 0]])
    # For the second batch notice that the number of valid boxes is 1 instead of 2
    # It means that the second row (groundtruth = 1) is a padding. Hence we flag
    # it has -1.
    expected_matched_labels = np.array([[0, 1, 0, 1, 1], [-1, 1, -1, 0, 0]])

    np.testing.assert_array_equal(matches, expected_matched)
    np.testing.assert_array_equal(match_labels, expected_matched_labels)

    # In this part we show that the linear assignment is not the same
    # when we do not have padded ground_truths
    num_valid_boxes = np.array([[3], [3]], np.int32)
    matches, match_labels = hungarian_matching(similarity, num_valid_boxes)
    expected_matched = np.array([[0, 1, 0, 2, 0], [0, 1, 0, 2, 0]])
    # For the second batch notice that the number of valid boxes is 1 instead of 2
    # It means that the second row (groundtruth = 1) is a padding. Hence we flag
    # it has -1.
    expected_matched_labels = np.array([[0, 1, 0, 1, 1], [0, 1, 0, 1, 1]])

    np.testing.assert_array_equal(matches, expected_matched)
    np.testing.assert_array_equal(match_labels, expected_matched_labels)
