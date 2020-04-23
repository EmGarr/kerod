import numpy as np
import pytest

from kerod.model.backbone.resnet import (Group, Resnet50, padd_for_aligning_pixels)


@pytest.mark.parametrize(["input_shape", "output_shape"], [
    [(320, 320), (325, 325)],
    [(321, 321), (357, 357)],
    [(800, 900), (805, 933)],
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
    inputs = np.zeros((1, input_shape[0], input_shape[1], 3))

    model = Resnet50()
    outputs = model(inputs)

    for output, stride in zip(outputs, [4, 8, 16, 32]):
        assert output.numpy().shape[:-1] == (1, output_shape[0] / stride, output_shape[1] / stride)


def test_freeze_normalization_group():
    group = Group(64, 3, strides=1, name='group0')
    group.freeze_normalization()

    for block in group.blocks:
        if block._use_conv_shortcut:
            assert not block.bn_shortcut.trainable
            assert not block.bn1.trainable
            assert not block.bn2.trainable
            assert not block.bn3.trainable


def test_freeze_normalization_resnet():
    model = Resnet50()
    model.freeze_normalization()
    assert not model.bn0.trainable
    for group in model.groups:
        for block in group.blocks:
            if block._use_conv_shortcut:
                assert not block.bn_shortcut.trainable
            assert not block.bn1.trainable
            assert not block.bn2.trainable
            assert not block.bn3.trainable
