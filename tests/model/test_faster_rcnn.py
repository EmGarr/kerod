import numpy as np
from absl.testing import parameterized
from tensorflow.python.keras import keras_parameterized

from kerod.core.standard_fields import BoxField
from kerod.dataset.preprocessing import expand_dims_for_single_batch, preprocess
from kerod.model.faster_rcnn import FasterRcnnFPNResnet50Caffe, FasterRcnnFPNResnet50Pytorch

TESTCASES = [{
    'testcase_name': 'caffe style',
    'model': FasterRcnnFPNResnet50Caffe,
}, {
    'testcase_name': 'pytorch style',
    'model': FasterRcnnFPNResnet50Pytorch,
}]


@keras_parameterized.run_all_keras_modes
class BuildFasterRCNNTest(keras_parameterized.TestCase):

    # Pytest parameterized doesn't work in a class with an __init__
    @parameterized.named_parameters(*TESTCASES)
    def test_build_fpn_resnet50_faster_rcnn(self, model):
        tmp_dir = self.get_temp_dir()
        num_classes = 2
        model = model(num_classes)

        inputs = {
            'image': np.zeros((100, 50, 3)),
            'objects': {
                BoxField.BOXES: np.array([[0, 0, 1, 1]], dtype=np.float32),
                BoxField.LABELS: np.array([1])
            }
        }

        x, y = expand_dims_for_single_batch(*preprocess(inputs))

        model(x)
        x['ground_truths'] = y
        model(x, training=True)
