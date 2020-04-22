import os

import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.python.keras import keras_parameterized

from od.core.standard_fields import BoxField
from od.dataset.preprocessing import expand_dims_for_single_batch, preprocess
from od.model.faster_rcnn import FasterRcnnFPNResnet50

TESTCASES = [{
    'testcase_name': 'base',
    'use_mixed_precision': False,
}]


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
        model = FasterRcnnFPNResnet50(num_classes)

        inputs = {
            'image': np.zeros((100, 50, 3)),
            'objects': {
                BoxField.BOXES: np.array([[0, 0, 1, 1]], dtype=np.float32),
                BoxField.LABELS: np.array([1])
            }
        }

        data = expand_dims_for_single_batch(*preprocess(inputs))

        model(data, training=True)
        model([data[0]])

        with pytest.raises(Exception):
            model.save(path_save_model)
            model = tf.keras.models.load_model(path_save_model)

        if use_mixed_precision:
            # with pytest.raises(Exception):
            #     my_model = tf.keras.models.load_model(path_save_model)
            mixed_precision.set_policy('float32')
