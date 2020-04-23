<h3 align="center">
<p>KEROD - Faster R-CNN for TensorFlow 2.2
</h3>

[![Build Status](https://img.shields.io/travis/TheAlgorithms/Python.svg?label=Travis%20CI&logo=travis&style=flat-square)](https://travis-ci.com/Emgarr/kerod)
[![codecov.io](https://codecov.io/gh/Emgarr/kerod/coverage.svg?branch=master)](https://codecov.io/gh/Emgarr/kerod/?branch=master)
_________________

[Read Latest Documentation](https://emgarr.github.io/od/) - [Browse GitHub Code Repository](https://github.com/Emgarr/kerod)
_________________


**Kerod** is pure `tensorflow 2` implementation of object detection algorithms (Faster R-CNN) aiming production. It stands for Keras Object Detection.

It aims to build a clear, reusable, tested, simple and documented codebase for tensorflow 2.2.

Many ideas have been based on [google object detection](https://github.com/tensorflow/models/tree/master/research/object_detection) and [tensorpack](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN).
 `Warning`: It is still a work in progress and some breaking changes could arrive soon. If you need to have good performances I'll advise to choose [tensorpack](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) (This is actually the same developer than Detectron2) for now but my aim is too match its benchmarks soon. The current AP@[.5:.95] on the [coco notebook](https://colab.research.google.com/github/Emgarr/kerod/blob/master/notebooks/coco_training.ipynb) is `27.4` (at commit e311bdf9a7c7f977bc7a82180d6877fb9f287372) which is quite low (but it was the first run so let's found those bugs). 


## Features

- As powerful and concise as Keras
- Low barrier to entry for educators and practitioners
- Handle batch in training and inference
- [Documentation](https://emgarr.github.io/kerod/)
- Simple (again)

### WIP features

- The mixed_precision is almost supported (should investigate). You can try it with this [notebook](https://colab.research.google.com/github/Emgarr/kerod/blob/master/notebooks/mixed_precision_pascal_voc_training_fpn50.ipynb)

### Algorithms

- [x] [Feature Pyramidal Network](https://arxiv.org/abs/1612.03144) 
- [ ] [Mask-RCNN](https://arxiv.org/abs/1703.06870) (Much of the pieces are already here just need to put everything together. It will arrive soon.)
- [ ] [Object Relation Network for object detection](https://arxiv.org/abs/1711.11575): aims to replace the fast-rcnn head multiclass nms. Will allow to make a better usage of the GPU (The NMS is used on CPU and block the serving efficiency).
- [ ] [Cascade R-CNN](https://arxiv.org/abs/1906.09756)
- [ ] [Max pool nms](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cai_MaxpoolNMS_Getting_Rid_of_NMS_Bottlenecks_in_Two-Stage_Object_Detectors_CVPR_2019_paper.pdf) will make the network more efficient on GPU.

### No configuration file

The code is (I hope) as simple as possible. You won't find any configuration file. All the parameters have already been chosen for you. If you need to change something simply code it and create a new layer.

Why: In deep learning each parameter is important. You must think thoroughly before a change on how it will impact your model. Here, the code base is super simple just rewrite the blocks that you need and create new layers using the power of Keras. Also, it makes the code easier to read.

## Installation

This repo is tested on Python 3.6, 3.7, 3.8 and TensorFlow 2.2.

You may want to install 'kerod' in a [virtual environment](https://docs.python.org/3/library/venv.html) or with [pyenv](https://github.com/pyenv/pyenv). Create a virtual environment with the version of Python you wanna use and activate it.

### With pip

```bash
pip install git+https://github.com/EmGarr/kerod.git
```

### From source

```bash
git clone https://github.com/EmGarr/kerod.git
cd od 
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
from kerod.model import factory

num_classes = 20
model = factory.build_model(num_classes, weights='imagenet')

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
```

### Notebooks

#### Requirements

If you don't run the examples on Colab please install `tensorflow_datasets`:

```bash
pip install tensorflow_datasets
```

#### Examples

Training an algorithm on COCO or Pascal VOC has never been so easy. You just need to run the cells and everything will be done for you. 

You can find examples in the [notebooks folder](./notebooks). There are no runners shipped with the library.

- Pascal VOC training example (single GPU) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Emgarr/kerod/blob/master/notebooks/pascal_voc_training_fpn50.ipynb).
- Mixed precision Pascal VOC training (single GPU)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Emgarr/kerod/blob/master/notebooks/mixed_precision_pascal_voc_training_fpn50.ipynb).
- Coco example (single GPU) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Emgarr/kerod/blob/master/notebooks/coco_training.ipynb).

### Serving

You can then use it in production with [tensorflow model server](https://www.tensorflow.org/tfx/serving/docker).

```python
import requests

from kerod.core.standard_fields import DatasetField

url = 'https://my_server:XXX/v1/models/serving:predict'

image = resize_to_min_dim(inputs['image'], 800.0, 1300.0)
image_information = tf.cast(tf.shape(image)[:2], dtype=tf.float32)

# Will perform a query for a single batch but you can perform query on batch
inputs = {
  DatasetField.IMAGES: tf.expand_dims(images, axis=0).numpy().tolist(),
  DatasetField.IMAGES_INFO: tf.expand_dims(image_information, axis=0).numpy().tolist(),
}

headers = {"content-type": "application/json"}
response = requests.post(url, data=json.dumps(inputs), headers=headers)
outputs = json.loads(response.text)['outputs']
```

The outputs will have the following format:

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

## Roadmap

### Futur Improvements

- Compute anchors once and slice the non usefull anchors. The anchors are computed at each inference which is useless. I should generate them on the maximum grid and slice them as done in tensorpack.
- The sampling is done in the graph. It may be worth it to remove it and place it in the tf.Dataset (like tensorpack) ?
