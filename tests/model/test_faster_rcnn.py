import os

import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.python.keras import keras_parameterized

from od.model.faster_rcnn import build_fpn_resnet50_faster_rcnn

TESTCASES = ({
    'testcase_name': 'mixed_precision',
    'use_mixed_precision': True
}, {
    'testcase_name': 'base',
    'use_mixed_precision': False,
})


@keras_parameterized.run_all_keras_modes
class BuildFasterRCNNTest(keras_parameterized.TestCase):

    # Pytest parameterized doesn't work in a class with an __init__
    @parameterized.named_parameters(*TESTCASES)
    def test_build_fpn_resnet50_faster_rcnn(self, use_mixed_precision):
        if use_mixed_precision:
            mixed_precision.set_policy('mixed_float16')
        tmp_dir = self.get_temp_dir()
        path_save_model = os.path.join(tmp_dir, 'save_model')
        num_classes = 20
        batch_size = 3
        model = build_fpn_resnet50_faster_rcnn(num_classes, batch_size)
        is_trainable = False
        for layer in model.layers:
            if layer.name == 'conv2_block3_out':
                is_trainable = True
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                assert not layer.trainable
            else:
                assert layer.trainable == is_trainable
        # Ensure layer have been set to false
        assert not is_trainable

        model_inference = build_fpn_resnet50_faster_rcnn(num_classes, None, training=False)

        model_inference.save(path_save_model)
        if use_mixed_precision:
            with pytest.raises(Exception):
                my_model = tf.keras.models.load_model(path_save_model)
            mixed_precision.set_policy('float32')
