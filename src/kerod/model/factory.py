import os
import tensorflow as tf

from kerod.model.faster_rcnn import FasterRcnnFPNResnet50
from kerod.core.standard_fields import DatasetField
from kerod.utils.training import freeze_layers_before

WEIGHTS_PATH = ('https://files.heuritech.com/raw_files/')


def build_model(num_classes, weights: str = 'imagenet'):
    model = FasterRcnnFPNResnet50(num_classes)
    images_information = tf.constant([[1300, 650]], tf.float32)
    features = tf.zeros((1, 1300, 650, 3), tf.float32)

    # Blank shot to init weights
    model([{DatasetField.IMAGES: features, DatasetField.IMAGES_INFO: images_information}])

    # The weights need to be loaded here before freezing any layers
    if weights == 'imagenet':
        weights_path = tf.keras.utils.get_file('resnet50_tensorpack_converted.h5',
                                               os.path.join(WEIGHTS_PATH,
                                                            'resnet50_tensorpack_converted.h5'),
                                               cache_subdir='models',
                                               md5_hash='f67cc7a3179ec847f2e2073d9533789b')
        model.resnet.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    model.resnet.freeze_normalization()
    # Will freeze all the layers before group number one
    # Should be done here because the layers won't be registered before the previous inference
    freeze_layers_before(model.resnet, model.resnet.groups[1].name)

    return model
