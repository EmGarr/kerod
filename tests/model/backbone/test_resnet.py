import pytest
import numpy as np
from od.model.backbone.resnet import padd_for_aligning_pixels, ResNet50V2


@pytest.mark.parametrize(["input_shape", "output_shape"], [
    [(320, 320), (320, 320)],
    [(321, 321), (352, 352)],
    [(800, 900), (800, 928)],
])
def test_padd_for_aligning_pixels(input_shape, output_shape):
    inputs = np.zeros((1, input_shape[0], input_shape[1], 3))
    padded_inputs = padd_for_aligning_pixels(inputs)
    assert padded_inputs.shape == (1, output_shape[0], output_shape[1], 3)

@pytest.mark.parametrize(["input_shape", "output_shape"], [
    [(320, 320), (320, 320)],
    [(321, 321), (352, 352)],
    [(800, 900), (800, 928)],
])
def test_resnet_shape(input_shape, output_shape):
    inputs = np.zeros((1, input_shape, output_shape, 3))

    model = ResNet50V2(input_shape=(None, None, 3))
    outputs = model(inputs)

    for output in zip(outputs, 8, 16, 32, 64):
        break


