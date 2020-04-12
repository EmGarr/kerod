from typing import List

import tensorflow as tf
from tensorflow.keras import backend, layers


def padd_for_aligning_pixels(inputs: tf.Tensor):
    """This padding operation is here to make the pixels of the output perfectly aligned,
    It padd with 0 the bottom and the right of the images
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


def preprocess_input(images: tf.Tensor):
    """Will preprocess the images for the resnet trained on Tensorpack.
    The network has been trained using BGR.
    """
    mean = tf.constant([-103.53, -116.28, -123.675], dtype=images.dtype)
    std = tf.constant([57.375, 57.12, 58.395], dtype=images.dtype)
    images = (images - mean) / std
    return images


class Group(tf.keras.layers.Layer):
    """A set of stacked residual blocks.

    Arguments:

    - *filters*: integer, filters of the bottleneck layer in a block.
    - *blocks*: number of blocks in the stacked blocks.
    - *strides*: Stride of the second conv layer in the first block.
    name: string, stack label.
    - *kernel_regularizer*: Apply a regularizer to all the Conv2D of the group.
    """

    def __init__(self, filters: int, blocks: int, strides: int, kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [
            Block(filters,
                  strides,
                  use_conv_shortcut=True,
                  kernel_regularizer=kernel_regularizer,
                  name='block0')
        ]
        self.blocks += [
            Block(filters,
                  1,
                  use_conv_shortcut=False,
                  kernel_regularizer=kernel_regularizer,
                  name=f'block{i}') for i in range(1, blocks)
        ]

    def call(self, inputs):
        x = inputs
        for block in self.blocks:
            x = block(x)
        return x

    def freeze_normalization(self):
        for block in self.blocks:
            block.freeze_normalization()

    @property
    def trainable(self):
        """We are forced to overide this method because keras do not succeed to set the state off
        self.blocks recursively.
        """
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        """We are forced to overide this method because keras do not succeed to set the state off
        self.blocks recursively.
        """
        self._trainable = value
        for block in self.blocks:
            block.trainable = value


class Block(tf.keras.layers.Layer):
    """A residual block.

    Arguments:

    - *filters*: integer, filters of the bottleneck layer.
    - *strides*: default 1, stride of the second convolution layer. In the official Keras
    implementation the stride is performed on the first convolution. This is different in
    the tensorpack implementation.
    - *use_conv_shortcut*: Use convolution shortcut if True, otherwise identity shortcut.
    - *kernel_regularizer*: Apply a kernel regularizer to all the Conv2D of the Block.
    """

    def __init__(self,
                 filters: int,
                 strides: int = 1,
                 use_conv_shortcut=True,
                 kernel_regularizer=None,
                 **kwargs):
        super().__init__(**kwargs)
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

        self.filters = filters
        self.strides = strides
        self._use_conv_shortcut = use_conv_shortcut

        if self._use_conv_shortcut:
            self.conv_shortcut = layers.Conv2D(4 * self.filters,
                                               1,
                                               strides=self.strides,
                                               use_bias=False,
                                               kernel_regularizer=kernel_regularizer,
                                               padding='same',
                                               name='convshortcut')
            self.bn_shortcut = layers.BatchNormalization(
                axis=bn_axis,
                name='convshortcut/bn',
            )
            self.relu_shortcut = layers.Activation('relu', name='convshortcut/relu')

        # Compared to keras resnet the stride 2 isn't done here (see conv2)
        self.conv1 = layers.Conv2D(self.filters,
                                   1,
                                   strides=1,
                                   use_bias=False,
                                   kernel_regularizer=kernel_regularizer,
                                   padding='same',
                                   name='conv1')
        self.bn1 = layers.BatchNormalization(axis=bn_axis, name='conv1/bn')
        self.relu1 = layers.Activation('relu', name='conv1/relu')

        if self.strides == 2:
            # Here we have a big difference the strides is done here instead of at the conv1
            self.pad = layers.ZeroPadding2D(padding=((1, 0), (1, 0)), name='pad2')
            self.conv2 = layers.Conv2D(self.filters,
                                       3,
                                       padding='VALID',
                                       use_bias=False,
                                       kernel_regularizer=kernel_regularizer,
                                       strides=self.strides,
                                       name='conv2')
        else:
            self.conv2 = layers.Conv2D(self.filters,
                                       3,
                                       use_bias=False,
                                       padding='same',
                                       kernel_regularizer=kernel_regularizer,
                                       strides=self.strides,
                                       name='conv2')

        self.bn2 = layers.BatchNormalization(axis=bn_axis, name='conv2/bn')
        self.relu2 = layers.Activation('relu', name='conv2/relu')

        self.conv3 = layers.Conv2D(self.filters * 4,
                                   1,
                                   use_bias=False,
                                   kernel_regularizer=kernel_regularizer,
                                   padding='same',
                                   name='conv3')
        self.bn3 = layers.BatchNormalization(axis=bn_axis, name='conv3/bn')
        self.relu3 = layers.Activation('relu', name='conv3/relu')

        self.last_relu = layers.Activation('relu', name='last_relu')

    def freeze_normalization(self):
        if self._use_conv_shortcut:
            self.bn_shortcut.trainable = False
        self.bn1.trainable = False
        self.bn2.trainable = False
        self.bn3.trainable = False

    def call(self, inputs):
        if self._use_conv_shortcut:
            shortcut = self.conv_shortcut(inputs)
            shortcut = self.bn_shortcut(shortcut)
            shortcut = self.relu_shortcut(shortcut)
        else:
            shortcut = inputs

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        if self.strides == 2:
            x = self.pad(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = x + shortcut
        return self.last_relu(x)


class Resnet(tf.keras.Model):
    """It instantiates the same ResNet v1 architecture than
    [tensorpack](https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/modeling/backbone.py).

    Arguments:

    - *groups*: A set of stacked residual blocks.
    - *kernel_regularizer*: Apply a kernel regularizer to all the Conv2D of the Resnet.
    """

    def __init__(self, groups: List[Group], kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        self.conv0 = layers.Conv2D(64,
                                   7,
                                   strides=2,
                                   use_bias=False,
                                   kernel_regularizer=kernel_regularizer,
                                   name='conv0')

        self.bn0 = layers.BatchNormalization(axis=bn_axis, name='conv0/bn')
        self.relu0 = layers.Activation('relu', name='conv0_relu')

        self.pad0 = layers.ZeroPadding2D(padding=((1, 0), (1, 0)), name='pool0_pad')
        self.pool0 = layers.MaxPooling2D(3, strides=2, name='pool0_pool')
        self.groups = groups

    def call(self, inputs: tf.Tensor) -> List[tf.Tensor]:
        x = preprocess_input(inputs)
        x = layers.Lambda(padd_for_aligning_pixels, name="padd_for_aligning_pixels")(x)

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.pad0(x)
        x = self.pool0(x)

        outputs = []
        for group in self.groups:
            x = group(x)
            outputs.append(x)

        # outputs[-1] = 32x downsampling
        # ceil(input/32)
        return outputs

    def freeze_normalization(self):
        self.bn0.trainable = False
        for group in self.groups:
            group.freeze_normalization()


class Resnet50(Resnet):

    def __init__(self, kernel_regularizer=None, **kwargs):

        groups = [
            Group(64, 3, strides=1, kernel_regularizer=kernel_regularizer, name='group0'),
            Group(128, 4, strides=2, kernel_regularizer=kernel_regularizer, name='group1'),
            Group(256, 6, strides=2, kernel_regularizer=kernel_regularizer, name='group2'),
            Group(512, 3, strides=2, kernel_regularizer=kernel_regularizer, name='group3')
        ]

        super().__init__(groups, kernel_regularizer=kernel_regularizer, **kwargs)


class Resnet101(Resnet):

    def __init__(self, kernel_regularizer=None, **kwargs):

        groups = [
            Group(64, 3, strides=1, kernel_regularizer=kernel_regularizer, name='group0'),
            Group(128, 4, strides=2, kernel_regularizer=kernel_regularizer, name='group1'),
            Group(256, 23, strides=2, kernel_regularizer=kernel_regularizer, name='group2'),
            Group(512, 3, strides=2, kernel_regularizer=kernel_regularizer, name='group3')
        ]
        super().__init__(groups, kernel_regularizer=kernel_regularizer, **kwargs)
