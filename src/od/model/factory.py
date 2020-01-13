import tensorflow as tf
from od.core.standard_fields import BoxField, DatasetField


def build_input_layers(training=False, batch_size=None):
    """Factory that build input layers for training and inference mode

    Arguments:

    - *training*: Boolean indicating if the network is either in training or inference mode.
    - *batch_size*: In training this value should be specified. The current value used is 1 but the
    the limit only depends on the limit of your GPU capacity.

    Returns:

    - *images*: A keras input for the images.
    - *images_information*: A keras input for the images information (image shapes without the padding).
    - *ground_truths*: If the training is set to true return the ground_truths needed by the model.

    Raises:

    - *ValueError*: is the batch_size has not been specified in training mode.
    """
    if training and batch_size is None:
        raise ValueError('In training you should specify a batch_size: 1 is a current value '
                         'in object detection of course it depends on your GPU capacity')

    images = tf.keras.layers.Input(shape=(None, None, 3),
                                   batch_size=batch_size,
                                   name=DatasetField.IMAGES)
    images_information = tf.keras.layers.Input(shape=(2),
                                               batch_size=batch_size,
                                               name=DatasetField.IMAGES_INFO)

    if training:
        ground_truths = {
            BoxField.BOXES:
                tf.keras.layers.Input(shape=(None, 4), batch_size=batch_size, name=BoxField.BOXES),
            BoxField.LABELS:
                tf.keras.layers.Input(shape=(None,),
                                      batch_size=batch_size,
                                      dtype=tf.int32,
                                      name=BoxField.LABELS),
            BoxField.WEIGHTS:
                tf.keras.layers.Input(shape=(None,),
                                      batch_size=batch_size,
                                      dtype=tf.float32,
                                      name=BoxField.WEIGHTS),
            BoxField.NUM_BOXES:
                tf.keras.layers.Input(shape=(batch_size),
                                      batch_size=batch_size,
                                      dtype=tf.int32,
                                      name=BoxField.NUM_BOXES)
        }
        return images, images_information, ground_truths

    return images, images_information
