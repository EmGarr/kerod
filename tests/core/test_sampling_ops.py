import pytest
import numpy as np
import tensorflow as tf

from kerod.core import sampling_ops


@pytest.mark.parametrize(
    "np_indicator,num_samples,expected_num_samples",
    [
        [[True, False, True, False, True, True, False], 3, 3],
        # indicator when less true elements than num_samples
        [[True, False, True, False, True, True, False], 5, 4]
    ])
def test_subsample_indicator(np_indicator, num_samples, expected_num_samples):
    indicator = tf.constant(np_indicator)
    samples = sampling_ops.subsample_indicator(indicator, num_samples)
    assert np.sum(samples) == expected_num_samples
    np.testing.assert_array_equal(samples, np.logical_and(samples, np_indicator))


def test_batch_subsample_indicator():
    numpy_labels = np.stack([np.arange(300) for _ in range(2)], axis=0)
    indicator = tf.constant(np.ones((2, 300)) == 1)
    numpy_labels = (numpy_labels - 200) > 0

    labels = tf.constant(numpy_labels)

    samples = sampling_ops.batch_sample_balanced_positive_negative(indicator, 64, labels)
    assert np.sum(samples) == 128
    assert samples.shape == (2, 300)


def test_subsample_indicator_when_num_samples_is_zero():
    np_indicator = [True, False, True, False, True, True, False]
    indicator = tf.constant(np_indicator)
    samples_none = sampling_ops.subsample_indicator(indicator, 0)
    np.testing.assert_array_equal(np.zeros_like(samples_none, dtype=bool), samples_none)


def test_subsample_indicator_when_indicator_all_false():
    indicator_empty = tf.zeros([0], dtype=tf.bool)
    samples_empty = sampling_ops.subsample_indicator(indicator_empty, 4)
    assert samples_empty.numpy().size == 0


def test_subsample_all_examples():
    numpy_labels = np.random.permutation(300)
    indicator = tf.constant(np.ones(300) == 1)
    numpy_labels = (numpy_labels - 200) > 0

    labels = tf.constant(numpy_labels)

    is_sampled = sampling_ops.sample_balanced_positive_negative(indicator, 64, labels)
    assert np.sum(is_sampled) == 64
    assert np.sum(np.logical_and(numpy_labels, is_sampled)) == 32
    assert np.sum(np.logical_and(np.logical_not(numpy_labels), is_sampled)) == 32


def test_sample_balanced_positive_negative():
    # Test random sampling when only some examples can be sampled:
    # 100 samples, 20 positives, 10 positives cannot be sampled
    numpy_labels = np.arange(100)
    numpy_indicator = numpy_labels < 90
    indicator = tf.constant(numpy_indicator)
    numpy_labels = (numpy_labels - 80) >= 0

    labels = tf.constant(numpy_labels)

    is_sampled = sampling_ops.sample_balanced_positive_negative(indicator, 64, labels)
    assert np.sum(is_sampled) == 64
    assert np.sum(np.logical_and(numpy_labels, is_sampled)) == 10
    assert np.sum(np.logical_and(np.logical_not(numpy_labels), is_sampled)) == 54
    np.testing.assert_array_equal(is_sampled, np.logical_and(is_sampled, numpy_indicator))


def test_sample_balance_positive_negative_selection_larger_sample_size():
    # Test random sampling when total number of examples that can be sampled are
    # less than batch size:
    # 100 samples, 50 positives, 40 positives cannot be sampled, batch size 64.
    numpy_labels = np.arange(100)
    numpy_indicator = numpy_labels < 60
    indicator = tf.constant(numpy_indicator)
    numpy_labels = (numpy_labels - 50) >= 0

    labels = tf.constant(numpy_labels)

    is_sampled = sampling_ops.sample_balanced_positive_negative(indicator, 64, labels)
    assert np.sum(is_sampled) == 60
    assert np.sum(np.logical_and(numpy_labels, is_sampled)) == 10
    assert np.sum(np.logical_and(np.logical_not(numpy_labels), is_sampled)) == 50
    np.testing.assert_array_equal(is_sampled, np.logical_and(is_sampled, numpy_indicator))


def test_sample_balance_positive_negative_selection_no_sample_size():
    # Test random sampling when only some examples can be sampled:
    # 1000 samples, 6 positives (5 can be sampled).
    numpy_labels = np.arange(1000)
    numpy_indicator = numpy_labels < 999
    indicator = tf.constant(numpy_indicator)
    numpy_labels = (numpy_labels - 994) >= 0

    labels = tf.constant(numpy_labels)

    is_sampled = sampling_ops.sample_balanced_positive_negative(indicator,
                                                                None,
                                                                labels,
                                                                positive_fraction=0.01)
    assert np.sum(is_sampled) == 500
    assert np.sum(np.logical_and(numpy_labels, is_sampled)) == 5
    assert np.sum(np.logical_and(np.logical_not(numpy_labels), is_sampled)) == 495
    np.testing.assert_array_equal(is_sampled, np.logical_and(is_sampled, numpy_indicator))


def test_batch_sample_balanced_positive_negative():
    numpy_labels = np.stack([np.arange(300) for _ in range(2)], axis=0)
    indicator = tf.constant(np.ones((2, 300)) == 1)
    numpy_labels = (numpy_labels - 200) > 0

    labels = tf.constant(numpy_labels)

    samples = sampling_ops.batch_sample_balanced_positive_negative(indicator, 64, labels)
    assert np.sum(samples) == 128
    assert samples.shape == (2, 300)
