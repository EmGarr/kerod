import os
from functools import partial

import numpy as np
import pytest
import tensorflow as tf
from kerod.core.standard_fields import BoxField, DatasetField
from kerod.dataset.preprocessing import (expand_dims_for_single_batch, preprocess)
from kerod.model import factory
from kerod.model.factory import KerodModel
from tensorflow.keras.callbacks import ModelCheckpoint


def test_build_fpn_resnet50_faster_rcnn_from_factory(tmpdir):
    num_classes = 20
    model = factory.build_model(num_classes)

    # Look at the trainable structure after the factory

    is_trainable = False
    for layer in model.backbone.layers:
        # All the layers before this one should be frozen
        if layer.name == 'resnet50/group0/block2/last_relu':
            is_trainable = True
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            assert not layer.trainable
        else:
            assert layer.trainable == is_trainable

    ## test the training
    inputs = {
        'image': np.zeros((2, 100, 50, 3)),
        'objects': {
            BoxField.BOXES: np.array([[[0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=np.float32),
            BoxField.LABELS: np.array([[1], [1]], np.int32)
        }
    }

    data = tf.data.Dataset.from_tensor_slices(inputs)
    data = data.map(preprocess)
    data = data.map(expand_dims_for_single_batch)

    base_lr = 0.02
    optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr)
    model.compile(optimizer=optimizer, loss=None)
    model.fit(data,
              validation_data=data,
              epochs=2,
              callbacks=[ModelCheckpoint(os.path.join(tmpdir, 'checkpoints'))])

    # Ensure kernel regularization has been applied
    assert len(model.backbone.losses) == 42
    assert len(model.rpn.losses) == 5
    assert len(model.fast_rcnn.losses) == 6
    assert len(model.fpn.losses) == 8
    assert len(model.losses) == 61

    model.predict(data, batch_size=2)

    serving_path = os.path.join(tmpdir, 'serving')
    model.export_for_serving(serving_path)
    reload_model = tf.keras.models.load_model(serving_path)
    for x, _ in data:
        reload_model.serving_step(x[DatasetField.IMAGES], x[DatasetField.IMAGES_INFO])


@pytest.mark.parametrize("model_name", [KerodModel.smca_r50, KerodModel.detr_resnet50])
def test_detr_like_architecture(model_name, tmpdir):
    num_classes = 20
    model = factory.build_model(num_classes, name=model_name)

    # Look at the trainable structure after the factory
    is_trainable = False
    for layer in model.backbone.layers:
        # All the layers before this one should be frozen
        if layer.name == 'resnet50/group0/block2/last_relu':
            is_trainable = True
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            assert not layer.trainable
        else:
            assert layer.trainable == is_trainable

    ## test the training
    inputs = {
        'image': np.zeros((2, 100, 50, 3)),
        'objects': {
            BoxField.BOXES: np.array([[[0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=np.float32),
            BoxField.LABELS: np.array([[1], [1]], np.int32)
        }
    }
    padded_shape = ({
        DatasetField.IMAGES: [None, None, 3],
        DatasetField.IMAGES_INFO: [2],
        DatasetField.IMAGES_PMASK: [None, None]
    }, {
        BoxField.BOXES: [None, 4],
        BoxField.LABELS: [None],
        BoxField.NUM_BOXES: [1],
        BoxField.WEIGHTS: [None]
    })
    batch_size = 2
    data = tf.data.Dataset.from_tensor_slices(inputs)
    data = data.map(partial(preprocess, padded_mask=True))
    data = data.padded_batch(batch_size, padded_shape)

    base_lr = 0.00002
    optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr)
    model.compile(optimizer=optimizer, loss=None)
    model.fit(data,
              validation_data=data,
              epochs=2,
              callbacks=[ModelCheckpoint(os.path.join(tmpdir, 'checkpoints'))])

    model.predict(data, batch_size=2)

    serving_path = os.path.join(tmpdir, 'serving')
    # Should fix that bug
    model.save(serving_path)
    model = tf.keras.models.load_model(serving_path)
    model.predict(data, batch_size=2)
