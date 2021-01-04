# Copyright 2017 The TensorFlow Authors and modified by Emilien Garreau. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Method to subsample minibatches by balancing positives and negatives.

Subsamples minibatches based on a pre-specified positive fraction in range
[0,1]. The class presumes there are many more negatives than positive examples:
if the desired sample_size cannot be achieved with the pre-specified positive
fraction, it fills the rest with negative examples. If this is not sufficient
for obtaining the desired sample_size, it returns fewer examples.

The main function to call is Subsample(self, indicator, labels). For convenience
one can also call SubsampleWeights(self, weights, labels) which is defined in
the minibatch_sampler base class.

When is_static is True, it implements a method that guarantees static shapes.
It also ensures the length of output of the subsample is always sample_size, even
when number of examples set to True in indicator is less than sample_size.
"""

import tensorflow as tf

from kerod.utils import ops


def subsample_indicator(indicator, num_samples):
    """Subsample indicator vector.

    Given a boolean indicator vector with M elements set to `True`, the function
    assigns all but `num_samples` of these previously `True` elements to
    `False`. If `num_samples` is greater than M, the original indicator vector
    is returned.

    Arguments:
    - *indicator*: a 1-dimensional boolean tensor indicating which elements
        are allowed to be sampled and which are not.

    - *num_samples*: int32 scalar tensor

    Returns:

    A boolean tensor with the same shape as input (indicator) tensor
    """
    indices = tf.where(indicator)
    indices = tf.random.shuffle(indices)
    indices = tf.reshape(indices, [-1])

    num_samples = tf.minimum(tf.size(indices), num_samples)
    selected_indices = tf.slice(indices, [0], tf.reshape(num_samples, [1]))

    selected_indicator = ops.indices_to_dense_vector(selected_indices, tf.shape(indicator)[0])

    return tf.equal(selected_indicator, 1)


def sample_balanced_positive_negative(indicator, sample_size, labels, positive_fraction=0.5):
    """Subsamples minibatches to a desired balance of positives and negatives.

    Arguments:

    - *indicator*: boolean tensor of shape [N] whose True entries can be sampled.
    - *sample_size*: desired batch size. If None, keeps all positive samples and
        randomly selects negative samples so that the positive sample fraction
        matches positive_fraction.
    - *labels*: boolean tensor of shape [N] denoting positive(=True) and negative
        (=False) examples.
    - *positive_fraction*: desired fraction of positive examples (scalar in [0,1])
        in the batch.

    Returns:

    *sampled_idx_indicator*: boolean tensor of shape [N], True for entries which are sampled.
    """

    negative_idx = tf.logical_not(labels)
    positive_idx = tf.logical_and(labels, indicator)
    negative_idx = tf.logical_and(negative_idx, indicator)

    # Sample positive and negative samples separately
    if sample_size is None:
        max_num_pos = tf.reduce_sum(tf.cast(positive_idx, dtype=tf.int32))
    else:
        max_num_pos = int(positive_fraction * sample_size)
    sampled_pos_idx = subsample_indicator(positive_idx, max_num_pos)
    num_sampled_pos = tf.reduce_sum(tf.cast(sampled_pos_idx, tf.int32))
    if sample_size is None:
        negative_positive_ratio = (1 - positive_fraction) / positive_fraction
        max_num_neg = tf.cast(negative_positive_ratio * tf.cast(num_sampled_pos, dtype=tf.float32),
                              dtype=tf.int32)
    else:
        max_num_neg = sample_size - num_sampled_pos
    sampled_neg_idx = subsample_indicator(negative_idx, max_num_neg)

    return tf.logical_or(sampled_pos_idx, sampled_neg_idx)


def batch_sample_balanced_positive_negative(indicators,
                                            sample_size,
                                            labels,
                                            positive_fraction=0.5,
                                            dtype=tf.float32):
    """Subsamples minibatches to a desired balance of positives and negatives.

    Arguments:

    - *indicator*: boolean tensor of shape [batch_size, N] whose True entries can be sampled.
    - *sample_size*: desired batch size. If None, keeps all positive samples and
        randomly selects negative samples so that the positive sample fraction
        matches positive_fraction.
    - *labels*: boolean tensor of shape [batch_size, N] denoting positive(=True) and negative
        (=False) examples.
    - *positive_fraction*: desired fraction of positive examples (scalar in [0,1])
        in the batch.

    Returns:

    A boolean tensor of shape [M, N], True for entries which are sampled.
    """

    def _minibatch_subsample_fn(inputs):
        indicators, targets = inputs
        return sample_balanced_positive_negative(tf.cast(indicators, tf.bool),
                                                 sample_size,
                                                 tf.cast(targets, tf.bool),
                                                 positive_fraction=positive_fraction)

    return tf.cast(tf.map_fn(_minibatch_subsample_fn, [indicators, labels],
                             dtype=tf.bool,
                             parallel_iterations=16,
                             back_prop=True),
                   dtype=dtype)
