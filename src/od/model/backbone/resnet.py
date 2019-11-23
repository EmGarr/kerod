"""ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
Adapted from code contributed by BigMoyan.
"""
import os

import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras import backend as K
from tensorflow.keras import utils

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    Arguments:

    - *input_tensor*: input tensor
    - *kernel_size*: default 3, the kernel size of
            middle conv layer at main path
    - *filters*: list of integers, the filters of 3 conv layer at main path
    - *stage*: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Return:

    Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(filters1, (1, 1), kernel_initializer='he_normal',
                  name=conv_name_base + '2a')(input_tensor)
    x = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', trainable=False)(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(filters2,
                  kernel_size,
                  padding='same',
                  kernel_initializer='he_normal',
                  name=conv_name_base + '2b')(x)
    x = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', trainable=False)(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(filters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(x)
    x = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', trainable=False)(x)

    x = KL.add([x, input_tensor])
    x = KL.Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    Arguments:

    - *input_tensor*: input tensor
    - *kernel_size*: default 3, the kernel size of
             middle conv layer at main path
    - *filters*: list of integers, the filters of 3 conv layer at main path
    - *stage*: integer, current stage label, used for generating layer names
    - *block*: 'a','b'..., current block label, used for generating layer names
    - *strides*: Strides for the first conv layer in the block.

    Returns:

    Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(filters1, (1, 1),
                  strides=strides,
                  kernel_initializer='he_normal',
                  name=conv_name_base + '2a')(input_tensor)
    x = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', trainable=False)(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(filters2,
                  kernel_size,
                  padding='same',
                  kernel_initializer='he_normal',
                  name=conv_name_base + '2b')(x)
    x = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', trainable=False)(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(filters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(x)
    x = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', trainable=False)(x)

    shortcut = KL.Conv2D(filters3, (1, 1),
                         strides=strides,
                         kernel_initializer='he_normal',
                         name=conv_name_base + '1')(input_tensor)
    shortcut = KL.BatchNormalization(axis=bn_axis, name=bn_name_base + '1',
                                     trainable=False)(shortcut)

    x = KL.add([x, shortcut])
    x = KL.Activation('relu')(x)
    return x


def ResNet50(weights='imagenet', input_tensor=None, input_shape=None):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Arguments:

    - *include_top*: whether to include the fully-connected
        layer at the top of the network.
    - *weights*: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
    - *input_tensor*: optional Keras tensor (i.e. output of `KL.Input()`)
        to use as image input for the model.
    - *input_shape*: optional shape tuple, only to be specified
        if `include_top` is False (otherwise the input shape
        has to be `(224, 224, 3)` (with `channels_last` data format)
        or `(3, 224, 224)` (with `channels_first` data format).
        It should have exactly 3 inputs channels,
        and width and height should be no smaller than 32.
        E.g. `(200, 200, 3)` would be one valid value.
    - *classes*: optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified.

    Return:

    A Keras model instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if input_tensor is None:
        img_input = KL.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = KL.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = KL.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = KL.Conv2D(64, (7, 7),
                  strides=(2, 2),
                  padding='valid',
                  kernel_initializer='he_normal',
                  name='conv1')(x)
    x = KL.BatchNormalization(axis=bn_axis, name='bn_conv1', trainable=False)(x)
    x = KL.Activation('relu')(x)
    x = KL.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = KL.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    b1 = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(b1, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    b2 = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(b2, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    b3 = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(b3, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    b4 = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = tf.keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = tf.keras.Model(inputs, [b2, b2, b3, b4], name='resnet50')

    # Load weights.
    if weights == 'imagenet':
        weights_path = tf.keras.utils.get_file(
            'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model
