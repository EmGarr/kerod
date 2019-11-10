import tensorflow as tf

import tensorflow.keras.losses as losses


class SmoothL1Localization(losses.Huber):
    """Smooth L1 localization loss function aka Huber Loss..
    The smooth L1_loss is defined elementwise as .5 x^2 if |x| <= delta and
    delta * (|x|- 0.5*delta) otherwise, where x is the difference between
    predictions and target.

    See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
    """

    def __init__(self, delta=1.0, name='huber_loss'):
        """Constructor.

        Arguments:

        - *delta*: delta for smooth L1 loss.

        """
        super().__init__(delta=delta, reduction=tf.keras.losses.Reduction.NONE, name=name)

    def call(self, y_true, y_pred):
        """Compute losses

        Arguments:

        - *y_true*: A float tensor of shape [batch_size, num_anchors, code_size]
                representing the (encoded) predicted locations of objects.
        - *y_pred*: A float tensor of shape [batch_size, num_anchors,
                code_size] representing the regression targets

        Returns:
            loss: A tensor of shape [batch_size, num_anchors] tensor
            representing the value of the loss function.
        """
        return tf.reduce_sum(super().call(y_true, y_pred), axis=2)


class BinaryCrossentropy(losses.BinaryCrossentropy):

    def __init__(self, **kwargs):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, from_logits=True, **kwargs)

    def call(self, y_true, y_pred):
        return super().call(y_true, y_pred)


class CategoricalCrossentropy(losses.CategoricalCrossentropy):

    def __init__(self, **kwargs):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, from_logits=True, **kwargs)

    def call(self, y_true, y_pred):
        return super().call(y_true, y_pred)


# class FocalCategoricalCrossentropy(losses.Loss):
#     """Focal categorical cross entropy loss.

#     Focal loss down-weights well classified examples and focusses on the hard
#     examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
#     """

#     def __init__(self, gamma=2.0, alpha=0.25, name='focal_loss'):
#         """Constructor.

#         Arguments:

#         - *gamma*: exponent of the modulating factor (1 - p_t) ^ gamma.
#         - *alpha*: optional alpha weighting factor to balance positives vs negatives.
#         """
#         super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)
#         self._alpha = alpha
#         self._gamma = gamma

#     def call(self, y_true, y_pred):

#         per_entry_cross_ent = (losses.cateforical_cross_entropy(y_true, y_pred, from_logits=True))
#         prediction_probabilities = tfkl.actications.softmax(y_pred)

#         p_t = ((y_true * prediction_probabilities) + ((1 - y_true) *
#                                                       (1 - prediction_probabilities)))
#         modulating_factor = 1.0
#         if self._gamma:
#             modulating_factor = tf.pow(1.0 - p_t, self._gamma)
#         alpha_weight_factor = 1.0
#         if self._alpha is not None:
#             alpha_weight_factor = (y_true * self._alpha + (1 - y_true) * (1 - self._alpha))
#         focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * per_entry_cross_ent)
#         return focal_cross_entropy_loss
