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


# Inputs of our model
batch_size = 2
images = tf.keras.layers.Input(shape=(None, None, 3),
                                batch_size=batch_size,
                                name='images')
images_information = tf.keras.layers.Input(shape=(2),
                                            batch_size=batch_size,
                                            name='images_info')

ground_truths = {
    BoxField.BOXES:
        tf.keras.layers.Input(shape=(None, 4), batch_size=batch_size, name='bbox'),
    BoxField.LABELS:
        tf.keras.layers.Input(shape=(None,),
                              batch_size=batch_size,
                              dtype=tf.int32,
                              name='label'),
    BoxField.WEIGHTS:
        tf.keras.layers.Input(shape=(None,),
                              batch_size=batch_size,
                              dtype=tf.float32,
                              name='weights'),
    BoxField.NUM_BOXES:
        tf.keras.layers.Input(shape=(batch_size),
                              batch_size=batch_size,
                              dtype=tf.int32,
                              name='num_boxes')
}
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

### Algorithms

- [x] [Feature Pyramidal Network](https://arxiv.org/abs/1612.03144) 
- [ ] [Mask-RCNN](https://arxiv.org/abs/1703.06870) (easy to implement)
- [ ] [Cascade R-CNN](https://arxiv.org/abs/1906.09756)
- [ ] [Object Relation Network for object detection](https://arxiv.org/abs/1711.11575): aims to replace the fast-rcnn head multiclass nms. Will
allow to make a better usage of the GPU (The NMS is used on CPU and block the serving efficiency).
- [ ] [Efficient Det](https://arxiv.org/pdf/1911.09070.pdf) - will be converted in 2 stage mode instead of 1 stage 
- [ ] [Max pool nms](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cai_MaxpoolNMS_Getting_Rid_of_NMS_Bottlenecks_in_Two-Stage_Object_Detectors_CVPR_2019_paper.pdf) will make the network more efficient on GPU.

Many ideas have been based on [google object detection](https://github.com/tensorflow/models/tree/master/research/object_detection) and [tensorpack](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN). 

