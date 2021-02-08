import tensorflow as tf


class Patches(tf.keras.layers.Layer):
    """Extract `patches` from `images`.

    This op collects patches from the input image, as if applying a
    convolution. All extracted patches are stacked in the depth (last) dimension
    of the output.

    Argument:

        patch_size:

    Inputs:

        images: A 3-D tensor of shape [batch_size, height, width, nb_channel]

    Output:
        patches: A 3-D tensor of shape [batch_size, ]
    """




    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
