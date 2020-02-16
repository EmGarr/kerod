<h3 align="center">
<p>OD - Faster R-CNN for TensorFlow 2.X 
</h3>

[![Build Status](https://img.shields.io/travis/TheAlgorithms/Python.svg?label=Travis%20CI&logo=travis&style=flat-square)](https://travis-ci.com/EmGarr/od)
[![codecov.io](https://codecov.io/gh/EmGarr/od/coverage.svg?branch=master)](https://codecov.io/gh/EmGarr/od/?branch=master)
_________________

[Read Latest Documentation](https://emgarr.github.io/od/) - [Browse GitHub Code Repository](https://github.com/EmGarr/od)
_________________


Od is pure tensorflow2.x implementation of object detection algorithms (Faster R-CNN) aiming production.

It aims to build a clear, reusable, tested, simple and documented codebase for tensorflow 2.X.

Many ideas have been based on [google object detection](https://github.com/tensorflow/models/tree/master/research/object_detection) and [tensorpack](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN). If you need to have good performances I'll advise to choose [tensorpack](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) (This is actually the same developer than Detectron2) for now but my aim is too match its benchmarks soon.

`Warning`: It is still a work in progress and some breaking changes could arrive soon.

## Features

- As powerful and concise as Keras
- Low barrier to entry for educators and practitioners
- Documentation
- Simple (again)
- Handle batch in training and inference

### Algorithms

- [x] [Feature Pyramidal Network](https://arxiv.org/abs/1612.03144) 
- [ ] [Mask-RCNN](https://arxiv.org/abs/1703.06870) (Much of the pieces are already here just need to put everything together. It will arrive soon.)
- [ ] [Object Relation Network for object detection](https://arxiv.org/abs/1711.11575): aims to replace the fast-rcnn head multiclass nms. Will allow to make a better usage of the GPU (The NMS is used on CPU and block the serving efficiency).
- [ ] [Cascade R-CNN](https://arxiv.org/abs/1906.09756)
- [ ] [Efficient Det](https://arxiv.org/pdf/1911.09070.pdf) - will be converted in 2 stage mode instead of 1 stage 
- [ ] [Max pool nms](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cai_MaxpoolNMS_Getting_Rid_of_NMS_Bottlenecks_in_Two-Stage_Object_Detectors_CVPR_2019_paper.pdf) will make the network more efficient on GPU.

### No configuration file

The code is (I hope) as simple as possible. You won't find any configuration file. All the parameters have already been chosen for you. If you need to change something simply code it and create a new layer.

Why: In deep learning each parameter is important. You must think thoroughly before a change on how it will impact your model. Here, the code base is super simple just rewrite the blocks that you need and create new layers using the power of Keras. Also, it makes the code easier to read.

## Installation

This repo is tested on Python 3.6 and 3.7 and TensorFlow 2.1.

You may want to install 'od' in a [virtual environment](https://docs.python.org/3/library/venv.html) or with [pyenv](https://github.com/pyenv/pyenv). Create a virtual environment with the version of Python you wanna use and activate it.

### With pip

```bash
pip install git+https://github.com/EmGarr/od.git
```

### From source

```bash
git clone https://github.com/EmGarr/od.git
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

## Examples

### Basic blocks

It provides simple blocks to create the state of the art object detection algorithms.

```python
from od.model.backbone.fpn import Pyramid
from od.model.backbone.resnet import ResNet50
from od.model.detection.fast_rcnn import FastRCNN
from od.model.detection.rpn import RegionProposalNetwork
from od.model import factory


# Inputs of our model
batch_size = 2
images, images_information, ground_truths = factory.build_input_layers(training=True, batch_size=batch_size)

# Keras layers
num_classes = 20
resnet = ResNet50(input_tensor=images, weights='imagenet')
pyramid = Pyramid()(resnet.outputs)
rois, _ = RegionProposalNetwork()([pyramid, images_information, ground_truths], training=True)
outputs = FastRCNN(num_classes + 1)([pyramid, rois, images_information, ground_truths],
                                        training=True)
model_faster_rcnn = tf.keras.Model(inputs=[images, images_information, ground_truths],
                                       outputs=outputs)
```

### Notebooks

You can find examples in the [notebooks folder](./notebooks). There are no runners shipped with the library.

- Pascal VOC training example [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmGarr/od/blob/master/notebooks/pascal_voc_training_fpn50.ipynb)
- Coco example [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmGarr/od/blob/master/notebooks/coco_training.ipynb). Training an algorithm has never been so easy just run the cells.

### Serving

#### Export

To export a model for tensorflow serving:

```python
import tensorflow as tf

from od.model.faster_rcnn import build_fpn_resnet50_faster_rcnn

model_faster_rcnn_inference = build_fpn_resnet50_faster_rcnn(num_classes, None, training=False)
model_faster_rcnn_inference.load_weights('my_amazing_weights.h5')
model_faster_rcnn_inference.save('serving_model')
```

#### Serving
You can then use it in production with [tensorflow model server](https://www.tensorflow.org/tfx/serving/docker).

```python
import requests

from od.core.standard_fields import DatasetField

API_ENDPOINT = 'https://my_server:XXX/'

image = resize_to_min_dim(inputs['image'], 800.0, 1300.0)
image_information = tf.cast(tf.shape(image)[:2], dtype=tf.float32)

# Will perform a query for a single batch but you can perform query on batch
inputs = {
  DatasetField.IMAGES: tf.expand_dims(images, axis=0).numpy().tolist(),
  DatasetField.IMAGES_INFO: tf.expand_dims(image_information, axis=0).numpy().tolist(),
}

url_serving = os.path.join(API_ENDPOINT, "v1/models/serving:predict")
headers = {"content-type": "application/json"}
response = requests.post(url_serving, data=json.dumps(signature), headers=headers)
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

### Implementation

- [ ] evaluation (MAP)
- [ ] Mixed Precision
- [ ] Improved fit loop using gradient tape

### Futur Improvements

- Compute anchors once and slice the non usefull anchors. The anchors are computed at each inference which is useless. I should generate them on the maximum grid and slice them as done in tensorpack.
- The sampling is done in the graph. It may be worth it to remove it and place it in the tf.Dataset (like tensorpack) ?
