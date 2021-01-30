import tensorflow as tf


class DynamicalWeightMaps(tf.keras.layers.Layer):
    """Dynamical spatial weight maps described in
    [Fast Convergence of DETR with Spatially Modulated Co-Attention](https://arxiv.org/pdf/2101.07448.pdf).

    Each object query first dynamically predicts the center and scale
    of its responsible object, which are then used to generate a 2-D Gaussian
    like spatial weight map.

    Example:

    This is a spatial weight map on a 600 x 1200 image with a reference point
    of y=0.62, x=0.73, height=0.33 and width=0.31.

    ![Spatial weight map](https://raw.githubusercontent.com/EmGarr/cv/master/ressources/spatial_weight_map.png)
    """

    def __init__(self, beta=1.):
        self._beta = tf.convert_to_tensor(beta)

    def call(self, height, width, ref_points):
        """

        Args:
            height: The targeted height
            width: The targeted width
            ref_points: A tensor of shape [batch_size, N, heads, (y, x, h, w)] in [0, 1]

        Returns:
            weight_map: A 4D tensor of float 32 and shape
                [batch_size, N, heads, height, width] representing
                a weight map per reference points..
        """
        x = tf.cast(tf.linspace(0, 1, width), self.dtype)
        y = tf.cast(tf.linspace(0, 1, height), self.dtype)
        x, y = tf.meshgrid(x, y)  # x and y have shape [height, width]

        return self.gaussian_weight_map(x, y, ref_points)

    def gaussian_weight_map(self, x, y, ref_points):
        y_cent, x_cent, height, width = tf.split(ref_points, 4, axis=-1)

        # [batch_size, N, heads, 1] => [batch_size, N, heads, 1, 1]
        y_cent, x_cent = y_cent[..., tf.newaxis], x_cent[..., tf.newaxis]
        height, width = height[..., tf.newaxis], width[..., tf.newaxis]

        # [height, width] => [1, 1, 1, height, width]
        y, x = y[tf.newaxis, tf.newaxis, tf.newaxis], x[tf.newaxis, tf.newaxis, tf.newaxis]

        # [batch_size, N, heads, height, width]
        x_term = -(x - x_cent)**2 / (self._beta * width**2)
        y_term = -(y - y_cent)**2 / (self._beta * height**2)
        return tf.math.exp(x_term + y_term)

    def get_config(self):
        config = super().get_config()
        config['beta'] = self._beta
        return config
