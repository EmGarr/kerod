import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from kerod.core.standard_fields import BoxField
from kerod.dataset.preprocessing import (expand_dims_for_single_batch,
                                         preprocess)
from kerod.model import factory


def test_build_fpn_resnet50_faster_rcnn_from_factory(tmpdir):
    num_classes = 20
    model = factory.build_model(num_classes)

    # Look at the trainable structure after the factory

    is_trainable = False
    for layer in model.resnet.layers:
        # All the layers before this one should be frozen
        if layer.name == 'conv2_block3_out':
            is_trainable = True
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            assert not layer.trainable
        else:
            assert layer.trainable == is_trainable

    assert len(model.losses) == 42
    ## test the training
    inputs = {
        'image': np.zeros((2, 100, 50, 3)),
        'objects': {
            BoxField.BOXES: np.array([[[0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=np.float32),
            BoxField.LABELS: np.array([[1], [1]])
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
    model.predict(data, batch_size=2)

    # model.save(tmpdir)
