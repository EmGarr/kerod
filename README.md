# OD
_________________

[![Build Status](https://img.shields.io/travis/TheAlgorithms/Python.svg?label=Travis%20CI&logo=travis&style=flat-square)](https://travis-ci.com/EmGarr/od)
[![codecov.io](https://codecov.io/gh/EmGarr/od/coverage.svg?branch=master)](https://codecov.io/gh/EmGarr/od/?branch=master)
_________________

[Read Latest Documentation](https://emgarr.github.io/od/) - [Browse GitHub Code Repository](https://github.com/EmGarr/od)
_________________


Od is pure tensorflow2.0 implementation of object detection algorithms aiming production:

It is still ongoing but aims to build a clear, reusable, tested, simple and documented codebase.

## Installation

Support python: 3.6, 3.7.

```bash
pip install git+https://github.com/EmGarr/od.git
```

## WIP

### Implementation

- [x] base network in eager
- [x] base network in graph mode
- [ ] training
- [ ] evaluation (MAP)

### Algorithms

- [x] [Feature Pyramidal Network](https://arxiv.org/abs/1612.03144) 
- [ ] [Mask-RCNN](https://arxiv.org/abs/1703.06870) (easy to implement)
- [ ] [Object Relation Network for object detection](https://arxiv.org/abs/1711.11575): aims to replace the fast-rcnn head multiclass nms. Will
allow to make a better usage of the GPU (The NMS is used on CPU and block the serving efficiency).
- [ ] [Efficient Det](https://arxiv.org/pdf/1911.09070.pdf) - will be converted in 2 stage mode instead of 1 stage 
- [ ] [Max pool nms](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cai_MaxpoolNMS_Getting_Rid_of_NMS_Bottlenecks_in_Two-Stage_Object_Detectors_CVPR_2019_paper.pdf) will make the network more efficient on GPU.

Many ideas have been based on [google object detection](https://github.com/tensorflow/models/tree/master/research/object_detection) and [tensorpack](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN). 

