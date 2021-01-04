import tensorflow as tf
from tensorflow.keras.losses import Loss


class L1Loss(Loss):

    def call(self, y_true, y_pred):
        return tf.norm(y_true - y_pred, ord=1, axis=-1)
