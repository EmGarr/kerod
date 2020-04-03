import tensorflow as tf


def freeze_layers_before(model: tf.keras.Model, layer_name: str):
    """Freezes layers of a Keras `model` before a given `layer_name` (excluded)."""

    freeze_before = model.get_layer(layer_name)
    index_freeze_before = model.layers.index(freeze_before)
    for layer in model.layers[:index_freeze_before]:
        layer.trainable = False


def freeze_batch_normalization(model: tf.keras.Model):
    """In Object detection we usually do not train on big batch. The BatchNormalization is
    not useful and need to be frozen.
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
