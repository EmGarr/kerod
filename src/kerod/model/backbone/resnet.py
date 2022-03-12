# Copyright 2015 The TensorFlow Authors and Modified by Emilien Garreau. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); # you may not use this file except in compliance with the License.
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
"""ResNet models for Keras.

Reference paper:

  - [Deep Residual Learning for Image Recognition] (https://arxiv.org/abs/1512.03385) (CVPR 2015)
"""

import os

from typing import Callable

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import resnet
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.utils import data_utils, layer_utils

OFFICIAL_WEIGHTS_PATH = ('https://storage.googleapis.com/tensorflow/keras-applications/resnet/')

WEIGHTS_HASHES = {
    'resnet50': ('4d473c1dd8becc155b73f8504c6f6626', OFFICIAL_WEIGHTS_PATH),
    'resnet50_pytorch': ('3ffd584081cc56435a3689d12afd7cf9', "https://drive.google.com/uc?export=download&id=11QKe1WD4s0IdAPh6hJXJozeqSO7QD0ZU"),
    'resnet101': ('88cf7a10940856eca736dc7b7e228a21', OFFICIAL_WEIGHTS_PATH),
    'resnet152': ('ee4c566cf9a93f14d82f913c2dc6dd0c', OFFICIAL_WEIGHTS_PATH),
    'resnet50v2': ('fac2f116257151a9d068a22e544a4917', OFFICIAL_WEIGHTS_PATH),
    'resnet101v2': ('c0ed64b8031c3730f411d2eb4eea35b5', OFFICIAL_WEIGHTS_PATH),
    'resnet152v2': ('ed17cf2e0169df9d443503ef94b23b33', OFFICIAL_WEIGHTS_PATH),
    'resnext50': ('62527c363bdd9ec598bed41947b379fc', OFFICIAL_WEIGHTS_PATH),
    'resnext101': ('0f678c91647380debd923963594981b3', OFFICIAL_WEIGHTS_PATH)
}


def padd_for_aligning_pixels(inputs: tf.Tensor):
    """This padding operation is here to make the pixels of the output perfectly aligned.
    It will make the output perfectly aligned at stride 32.
    """

    chan = inputs.shape[3]
    b4_stride = 32.0
    shape2d = tf.shape(inputs)[1:3]
    new_shape2d = tf.cast(
        tf.math.ceil(tf.cast(shape2d, tf.float32) / b4_stride) * b4_stride, tf.int32)
    pad_shape2d = new_shape2d - shape2d
    inputs = tf.pad(inputs,
                    tf.stack([[0, 0],
                              [3, 2 + pad_shape2d[0]],
                              [3, 2 + pad_shape2d[1]],
                              [0, 0]]),
                    name='conv1_pad') # yapf: disable
    inputs.set_shape([None, None, None, chan])
    return inputs


def ResNet(stack_fn: Callable,
           preprocessing_func: Callable,
           preact: bool,
           use_bias: bool,
           model_name='resnet',
           weights='imagenet',
           input_tensor=None,
           input_shape=None,
           **kwargs) -> tf.keras.Model:
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

    Reference paper:

    [Deep Residual Learning for Image Recognition]
        (https://arxiv.org/abs/1512.03385) (CVPR 2015)
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    Caution: Be sure to properly pre-process your inputs to the application.
    Please see `applications.resnet.preprocess_input` for an example.

    Arguments:
        stack_fn: a function that returns output tensor for the
            stacked residual blocks.
        preprocessing_func: a function that returns the corresponding preprocessing of the network.
        preact: whether to use pre-activation or not
            (True for ResNetV2, False for ResNet and ResNeXt).
        use_bias: whether to use biases for convolutional layers or not
            (True for ResNet and ResNetV2, False for ResNeXt).
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        kwargs: For backwards compatibility only.

    Returns:
        A `keras.Model` instance.

    Raises:
        *ValueError*: in case of invalid argument for `weights`, or invalid input shape.
    """
    if kwargs:
        raise ValueError('Unknown argument(s): %s' % (kwargs,))
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.Lambda(preprocessing_func, name="preprocess_input")(img_input)
    x = layers.Lambda(padd_for_aligning_pixels, name="padd_for_aligning_pixels")(x)
    x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    if not preact:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    outputs = stack_fn(x)

    if preact:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='post_bn')(outputs[-1])
        outputs[-1] = layers.Activation('relu', name='post_relu')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = training.Model(inputs, outputs, name=model_name)

    # Load weights.
    if (weights == 'imagenet') and (model_name in WEIGHTS_HASHES):
        file_hash, base_weight_path = WEIGHTS_HASHES[model_name]

        if model_name == "resnet50_pytorch":
            file_name = 'resnet50_tensorpack_conversion.h5'
            url = base_weight_path
        else:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
            url = base_weight_path + file_name

        weights_path = data_utils.get_file(file_name, url, cache_subdir='models', file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def ResNet50(weights='imagenet', input_tensor=None, input_shape=None, **kwargs):
    """Instantiates the ResNet50 architecture."""
    # Or set to None
    resnet.layers = tf.keras.layers
    def stack_fn(x):
        c2 = resnet.stack1(x, 64, 3, stride1=1, name='conv2')
        c3 = resnet.stack1(c2, 128, 4, name='conv3')
        c4 = resnet.stack1(c3, 256, 6, name='conv4')
        c5 = resnet.stack1(c4, 512, 3, name='conv5')
        return [c2, c3, c4, c5]

    return ResNet(stack_fn,
                  resnet.preprocess_input,
                  False,
                  True,
                  'resnet50',
                  weights,
                  input_tensor,
                  input_shape,
                  **kwargs)


def ResNet50PytorchStyle(weights='imagenet', input_tensor=None, input_shape=None, **kwargs):
    """Instantiates the ResNet50 with the pytorch style architecture.
    In the bottleneck residual block, pytorch-style ResNet uses a 1x1 stride-1 convolutional layer
    followed by a 3x3 stride-2 convolutional layer.

    **Warning**: Do not forget to use `bgr` instead of `rgb`.

    ```python
    import functools
    import tensorflow_datasets as tfds
    from kerod.preprocessing import preprocess

    ds_train, ds_info = tfds.load(name="coco/2017", split="train", shuffle_files=True, with_info=True)
    ds_train = ds_train.map(functools.partial(preprocess, bgr=True),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ```
    """

    def stack_fn(x):
        c2 = Group(x, 64, 3, strides=1, name='resnet50/group0')
        c3 = Group(c2, 128, 4, strides=2, name='resnet50/group1')
        c4 = Group(c3, 256, 6, strides=2, name='resnet50/group2')
        c5 = Group(c4, 512, 3, strides=2, name='resnet50/group3')
        return [c2, c3, c4, c5]

    return ResNet(stack_fn, preprocess_input_pytorch, False, False, 'resnet50_pytorch', weights,
                  input_tensor, input_shape, **kwargs)


def preprocess_input_pytorch(images: tf.Tensor):
    """Will preprocess the images for the resnet trained on Tensorpack.
    The network has been trained using BGR.
    """
    mean = tf.constant([103.53, 116.28, 123.675], dtype=images.dtype)
    std = tf.constant([57.375, 57.12, 58.395], dtype=images.dtype)
    images = (images - mean) / std
    return images


def Group(inputs: tf.Tensor, filters: int, blocks: int, strides: int, name=None):
    """A set of stacked residual blocks with the pytorch style

    Arguments:
        filters: integer, filters of the bottleneck layer in a block.
        blocks: number of blocks in the stacked blocks.
        strides: Stride of the second conv layer in the first block.
    """

    x = Block(inputs, filters, strides, use_conv_shortcut=True, name=f'{name}/block0')
    for i in range(1, blocks):
        x = Block(x, filters, 1, use_conv_shortcut=False, name=f'{name}/block{i}')
    return x


def Block(inputs: tf.Tensor, filters: int, strides: int = 1, use_conv_shortcut=True, name=None):
    """A residual block with the pytorch_style

    Arguments:
        inputs: The inputs tensor
        filters: integer, filters of the bottleneck layer.
        strides: default 1, stride of the second convolution layer. In the official Keras
            implementation the stride is performed on the first convolution. This is different in
            the pytorch implementation.
        use_conv_shortcut: Use convolution shortcut if True, otherwise identity shortcut.
    """

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if use_conv_shortcut:
        shortcut = layers.Conv2D(4 * filters,
                                 1,
                                 strides=strides,
                                 use_bias=False,
                                 padding='same',
                                 name=f'{name}/convshortcut')(inputs)
        shortcut = layers.BatchNormalization(axis=bn_axis, name=f'{name}/convshortcut/bn')(shortcut)
    else:
        shortcut = inputs

    x = layers.Conv2D(filters, 1, strides=1, use_bias=False, padding='same',
                      name=f'{name}/conv1')(inputs)
    x = layers.BatchNormalization(axis=bn_axis, name=f'{name}/conv1/bn')(x)
    x = layers.Activation('relu', name=f'{name}/conv1/relu')(x)

    if strides == 2:
        x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)), name=f'{name}/pad2')(x)
        x = layers.Conv2D(filters,
                          3,
                          padding='valid',
                          use_bias=False,
                          strides=strides,
                          name=f'{name}/conv2')(x)
    else:
        x = layers.Conv2D(filters,
                          3,
                          use_bias=False,
                          padding='same',
                          strides=strides,
                          name=f'{name}/conv2')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=f'{name}/conv2/bn')(x)
    x = layers.Activation('relu', name=f'{name}/conv2/relu')(x)

    x = layers.Conv2D(filters * 4, 1, use_bias=False, padding='same', name=f'{name}/conv3')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=f'{name}/conv3/bn')(x)
    x = layers.Add()([shortcut, x])
    return layers.Activation('relu', name=f'{name}/last_relu')(x)


