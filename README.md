# OD
_________________

[![Build Status](https://img.shields.io/travis/TheAlgorithms/Python.svg?label=Travis%20CI&logo=travis&style=flat-square)](https://travis-ci.com/EmGarr/od)
[![codecov.io](https://codecov.io/gh/EmGarr/od/coverage.svg?branch=master)](https://codecov.io/gh/EmGarr/od/?branch=master)
_________________

[Read Latest Documentation](https://emgarr.github.io/od/) - [Browse GitHub Code Repository](https://github.com/EmGarr/od)
_________________


Od is pure tensorflow2.x implementation of object detection algorithms aiming production.

It aims to build a clear, reusable, tested, simple and documented codebase for tensorflow.

## Installation

Support python: 3.6, 3.7.

```bash
pip install git+https://github.com/EmGarr/od.git
```

## Example

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

## Notebooks

Pascal VOC training example [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmGarr/od/blob/master/notebooks/pascal_voc_training_fpn50.ipynb)

## WIP

### Implementation

- [x] base network in eager
- [x] base network in graph mode
- [x] training
- [ ] evaluation (MAP)
- [ ] visualisation of the predictions
- [ ] Mixed Precision

### Improvements

Compute anchors once and slice the non usefull anchors

### Algorithms

- [x] [Feature Pyramidal Network](https://arxiv.org/abs/1612.03144) 
- [ ] [Mask-RCNN](https://arxiv.org/abs/1703.06870) (easy to implement)
- [ ] [Cascade R-CNN](https://arxiv.org/abs/1906.09756)
- [ ] [Object Relation Network for object detection](https://arxiv.org/abs/1711.11575): aims to replace the fast-rcnn head multiclass nms. Will
allow to make a better usage of the GPU (The NMS is used on CPU and block the serving efficiency).
- [ ] [Efficient Det](https://arxiv.org/pdf/1911.09070.pdf) - will be converted in 2 stage mode instead of 1 stage 
- [ ] [Max pool nms](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cai_MaxpoolNMS_Getting_Rid_of_NMS_Bottlenecks_in_Two-Stage_Object_Detectors_CVPR_2019_paper.pdf) will make the network more efficient on GPU.

Many ideas have been based on [google object detection](https://github.com/tensorflow/models/tree/master/research/object_detection) and [tensorpack](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN). 

