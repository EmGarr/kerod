import numpy as np
import pytest

from kerod.model.backbone.resnet import ResNet50, ResNet50PytorchStyle, padd_for_aligning_pixels


@pytest.mark.parametrize(["input_shape", "output_shape"], [
    [(320, 320), (325, 325)],
    [(321, 321), (357, 357)],
    [(800, 900), (805, 933)],
    [(797, 1333), (805, 1349)],
])
def test_padd_for_aligning_pixels(input_shape, output_shape):
    inputs = np.zeros((1, input_shape[0], input_shape[1], 3))
    padded_inputs = padd_for_aligning_pixels(inputs)
    assert padded_inputs.shape == (1, output_shape[0], output_shape[1], 3)


@pytest.mark.parametrize("model", [ResNet50PytorchStyle, ResNet50])
@pytest.mark.parametrize(["input_shape", "output_shape"], [
    [(320, 320), (320, 320)],
    [(321, 321), (352, 352)],
    [(800, 900), (800, 928)],
    [(797, 1333), (800, 1344)],
])
def test_resnet_shape(input_shape, output_shape, model):
    inputs = np.zeros((1, input_shape[0], input_shape[1], 3))

    model = model(input_shape=[None, None, 3])
    outputs = model(inputs)

    for output, stride in zip(outputs, [4, 8, 16, 32]):
        assert output.numpy().shape[:-1] == (1, output_shape[0] / stride, output_shape[1] / stride)
