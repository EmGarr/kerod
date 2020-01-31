import tensorflow as tf
import numpy as np
from od.model.layers.relation_network import RelationFeature

from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized


@keras_parameterized.run_all_keras_modes
class ActivationsTest(keras_parameterized.TestCase):

    def test_relation_feature(self):

        # Verify the layer can be instanciated in a keras model without issue
        last_dim = 1
        inputs = [keras.Input(shape=(None, None, last_dim)), keras.Input(shape=(None, 4))]
        num_units = 32
        outputs = RelationFeature(num_units)(inputs)
        model = keras.Model(inputs, outputs)

        boxes = tf.constant([[[3.0, 4.0, 6.0, 8.0], [0.0, 0.0, 20.0, 20.0]],
                             [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]]])

        feature = np.random.rand(2, 2, last_dim)

        out = keras.backend.eval(model([feature, boxes]))

        self.assertAllClose(out.shape, (2, 2, num_units))
        # No nan in the output
        self.assertEqual(tf.math.is_nan(tf.reduce_sum(out)), False)
