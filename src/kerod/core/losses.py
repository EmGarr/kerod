import tensorflow as tf

import tensorflow.keras.losses as losses


class SmoothL1Localization(losses.Huber):
    """Tweeked Smooth L1 localization loss function aka Huber Loss..
    The smooth L1_loss is defined elementwise as .5 x^2 if |x| <= delta and
    delta * (|x|- 0.5*delta) otherwise, where x is the difference between
    predictions and target.

    See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
    """

    def __init__(self, delta=1.0 / 9, name='huber_loss'):
        """Constructor.

        Arguments:

        - *delta*: delta for smooth L1 loss. The default value is the one used in [tensorpack](https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/modeling/model_rpn.py#L94)
        for the rpn.

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
        y_true = tf.expand_dims(y_true, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=-1)
        return tf.reduce_sum(super().call(y_true, y_pred), axis=-1)
