<h3 align="center">
<p>KEROD - Object Detection for TensorFlow 2.X (FasterRCNN, DeTr)
</h3>

![Build](https://github.com/EmGarr/kerod/workflows/Test/badge.svg)
[![codecov.io](https://codecov.io/gh/Emgarr/kerod/coverage.svg?branch=master)](https://codecov.io/gh/Emgarr/kerod/?branch=master)
![Documentation](https://github.com/EmGarr/kerod/workflows/Documentation/badge.svg)
_________________

[Read Latest Documentation](https://emgarr.github.io/kerod/) - [Browse GitHub Code Repository](https://github.com/Emgarr/kerod)
_________________


**Kerod** is pure `tensorflow 2` implementation of object detection algorithms (Faster R-CNN, DeTr) aiming production. It stands for Keras Object Detection.

It aims to build a clear, reusable, tested, simple and documented codebase for tensorflow 2.X.

Many ideas have been based on [google object detection](https://github.com/tensorflow/models/tree/master/research/object_detection), [tensorpack](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) and [mmdetection](https://github.com/open-mmlab/mmdetection).


## Features

- As powerful and concise as Keras
- Low barrier to entry for educators and practitioners
- Handle batch in training and inference
- Rich [Documentation](https://emgarr.github.io/kerod/)
- Multi-GPU
- Mixed_precision. You can try it with this [notebook](https://colab.research.google.com/github/Emgarr/kerod/blob/master/notebooks/mixed_precision_pascal_voc_training_fpn50.ipynb)
- No need to struggle to download COCO or PascalVoc. We only used `tensorflow_datasets`
- Simple (again)

### Algorithms

- [x] [Feature Pyramidal Network](https://arxiv.org/abs/1612.03144) 
- [x] DeTr from: [End to end object detection with transformers](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers)
- [x] Single scale SMCA (WIP not 100% the exact implementation o the paper): [Fast Convergence of DETR with Spatially Modulated Co-Attention](https://arxiv.org/pdf/2101.07448.pdf)
- [ ] [Mask-RCNN](https://arxiv.org/abs/1703.06870) (Much of the pieces are already here just need to put everything together. It will arrive soon.)

## Try Kerod 

### Notebooks

Training an algorithm on COCO or Pascal VOC has never been so easy. You just need to run the cells and everything will be done for you. 

You can find examples in the [notebooks folder](./notebooks). There are no runners shipped with the library.

| Algorithm | Dataset | Performance | MultiGPU | Mixed Precision | Notebook |
| -------- | -------- | --- |----|---- | --|
| [FasterRcnnFPNResnet50Pytorch](https://github.com/EmGarr/kerod/blob/master/src/kerod/model/faster_rcnn.py) | PascalVoc |             |          |                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Emgarr/kerod/blob/master/notebooks/pascal_voc_training_fpn50.ipynb) |
| [FasterRcnnFPNResnet50Pytorch](https://github.com/EmGarr/kerod/blob/master/src/kerod/model/faster_rcnn.py) | PascalVoc |             |                    | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Emgarr/kerod/blob/master/notebooks/mixed_precision_pascal_voc_training_fpn50.ipynb) |   |
| [FasterRcnnFPNResnet50Pytorch](https://github.com/EmGarr/kerod/blob/master/src/kerod/model/faster_rcnn.py) | COCO      |             |                    |                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Emgarr/kerod/blob/master/notebooks/coco_training.ipynb)                             |   |
| [FasterRcnnFPNResnet50Pytorch](https://github.com/EmGarr/kerod/blob/master/src/kerod/model/faster_rcnn.py) | COCO      | 30 mAP on old code base. A bug has been removed since. I need 4 GPUs to rerun the training. | :heavy_check_mark: |                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Emgarr/kerod/blob/master/notebooks/coco_training_multi_gpu.ipynb)                   |   |
| [DetrResnet50Pytorch](https://github.com/EmGarr/kerod/blob/master/src/kerod/model/detr.py) | COCO      | NEED 16 GPUS for 3 days | :heavy_check_mark: |                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Emgarr/kerod/blob/master/notebooks/detr_coco_training_multi_gpu.ipynb)              |   |
| [SMCAR50Pytorch](https://github.com/EmGarr/kerod/blob/master/src/kerod/model/smca_detr.py) | COCO      | NEED 8 GPUS for 1 days  | :heavy_check_mark: |                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Emgarr/kerod/blob/master/notebooks/smca_coco_training_multi_gpu.ipynb)          |   |

Pytorch: means resnet implementation Pytorch style. In the residual block we have: conv (1x1) stride 1 -> conv (3x3) stride 2 instead of conv (1x1) stride 2 -> conv (3x3) stride 1 (Caffe, Keras implementations)

If you want to perform an overfit you have an example with the detr architecture:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Emgarr/kerod/blob/master/notebooks/overfit-detr.ipynb)

### Requirements

If you don't run the examples on Colab please install `tensorflow_datasets`:

```bash
pip install tensorflow_datasets
```

### No configuration file

The code is (I hope) as simple as possible. You won't find any configuration file. All the parameters have already been chosen for you. If you need to change something simply code it and create a new layer.

Why: In deep learning each parameter is important. You must think thoroughly before a change on how it will impact your model. Here, the code base is super simple just rewrite the blocks that you need and create new layers using the power of Keras. Also, it makes the code easier to read.

## Installation

This repo is tested on Python 3.6, 3.7, 3.8 and TensorFlow 2.4.0

You may want to install 'kerod' in a [virtual environment](https://docs.python.org/3/library/venv.html) or with [pyenv](https://github.com/pyenv/pyenv). Create a virtual environment with the version of Python you wanna use and activate it.

### With pip

```bash
pip install git+https://github.com/EmGarr/kerod.git
```

### From source

```bash
git clone https://github.com/EmGarr/kerod.git
cd kerod
pip install .
```

When you update the repository, you should upgrade installation and its dependencies as follows:

```bash
git pull
pip install --upgrade .
```

You can install the package in dev mode as follow and everytime you refresh the package it will be automatically updated:

```bash
pip install -e .
```

## Tutorials

### Simple example

To run a training you just need to write the following. 

```python
import numpy as np
from kerod.dataset.preprocessing import expand_dims_for_single_batch, preprocess
from kerod.model import factory, KerodModel

num_classes = 20
model = factory.build_model(num_classes, name=KerodModel.faster_rcnn_resnet50_pytorch)

# Same format than COCO and Pascal VOC in tensorflow datasets
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
model.fit(data, epochs=2, callbacks=[ModelCheckpoint('checkpoints')])

results = model.predict(data, batch_size=1)

use_faster_rcnn = True
if use_faster_rcnn:
    model.export_for_serving('saved_model')
else:
    model.save('saved_model')
reload_model = tf.keras.models.load_model('saved_model')
for x, _ in data:
    if use_faster_rcnn:
        reload_model.serving_step(x[DatasetField.IMAGES], x[DatasetField.IMAGES_INFO])
    else:
        reload_model.predict_step(x)
```

### Mixed Precision

```python
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from kerod.dataset.preprocessing import expand_dims_for_single_batch, preprocess
from kerod.model import factory

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

num_classes = 20
model = factory.build_model(num_classes)
```

### Multi-GPU training

```python

import numpy as np
from kerod.dataset.preprocessing import expand_dims_for_single_batch, preprocess
from kerod.model import factory

batch_size_per_gpu = 2
num_gpus = 8
batch_size = batch_size_per_gpu * num_gpus

padded_shape = ({
      DatasetField.IMAGES: [None, None, 3],
      DatasetField.IMAGES_INFO: [2]
    },
    {
      BoxField.BOXES: [None, 4],
      BoxField.LABELS: [None],
      BoxField.NUM_BOXES: [1],
      BoxField.WEIGHTS: [None]
    })    

data = tf.data.Dataset.from_tensor_slices(inputs)
data =  data.padded_batch(batch_size, padded_shape)
data = data.prefetch(tf.data.experimental.AUTOTUNE)

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope(): 
    model = factory.build_model(num_classes)
    base_lr = 0.02
    optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr)
    model.compile(optimizer=optimizer, loss=None)

model.fit(data, epochs=2, callbacks=[ModelCheckpoint('checkpoints')])
```

### Serving

You can then use it in production with [tensorflow model server](https://www.tensorflow.org/tfx/serving/docker).

```python
import requests

from kerod.core.standard_fields import DatasetField

url = 'https://my_server:XXX/v1/models/serving:predict'

image = resize_to_min_dim(inputs['image'], 800.0, 1300.0)
image_information = tf.cast(tf.shape(image)[:2], dtype=tf.float32)

# Will perform a query for a single batch but you can perform query on batch
inputs = [
    tf.expand_dims(images, axis=0).numpy().tolist(),
  tf.expand_dims(image_information, axis=0).numpy().tolist()
]

headers = {"content-type": "application/json"}
response = requests.post(url, data=json.dumps(inputs), headers=headers)
outputs = json.loads(response.text)['outputs']
```

See the outputs of the predict_step of your.
For FasterRCNN they will have a similar format:

- *bbox*: A Tensor of shape [batch_size, max_detections, 4]
containing the non-max suppressed boxes. The coordinates returned are
between 0 and 1.
- *scores*: A Tensor of shape [batch_size, max_detections] containing
the scores for the boxes.
- *label*: A Tensor of shape [batch_size, max_detections] 
containing the class for boxes.
- *num_boxes*: A [batch_size] int32 tensor indicating the number of
valid detections per batch item. Only the top valid_detections[i] entries
in nms_boxes[i], nms_scores[i] and nms_class[i] are valid. The rest of the
entries are zero paddings.


## Tests

In order to run the tests you should install pytest.

```bash
pip install pytest
```

Here's the easiest way to run tests for the library:

```bash
make test
```

or

```bash
pytest tests/
```
