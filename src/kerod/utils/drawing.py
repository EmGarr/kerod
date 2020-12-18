from typing import List, Union

import matplotlib.colors as pltc
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class BoxDrawer:
    """Outputs a copy of images but draws on top of the pixels zero or more bounding boxes specified
    by the locations in boxes. The coordinates of the each bounding box in boxes are encoded as
    [y_min, x_min, y_max, x_max]. The bounding box coordinates can be:

    1. floats in [0.0, 1.0] relative to the width and height of the underlying image
    2. already resized to the corresponding size

    Arguments:

    - *classes*: The ordered list of classes that we could display
    """

    def __init__(self, classes: List[str]):
        self._classes = classes
        all_colors = [color for color in pltc.cnames.keys()]
        self._colors = [
            all_colors[i] for i in np.random.randint(0, len(all_colors), size=len(classes))
        ]

    def __call__(self,
                 images,
                 boxes,
                 labels: Union[np.array, List[List[int]], tf.Tensor],
                 scores: Union[np.array, List[List[float]], tf.Tensor],
                 num_valid_detections: Union[np.array, List[int], tf.Tensor],
                 resize=True):
        """Outputs a copy of images but draws on top of the pixels zero or more bounding boxes specified
        by the locations in boxes. The coordinates of the each bounding box in boxes are encoded as
        [y_min, x_min, y_max, x_max]. The bounding box coordinates can be:

        1. floats in [0.0, 1.0] relative to the width and height of the underlying image
        2. already resized to the corresponding size

        Arguments:

        - *images*: An image (tf.Tensor or numpy array) with shape [batch, height, width, 3]
        - *boxes*: An array (tf.Tensor, or Numpy array) of boxes with the following shape [batch, num_boxes, (y_min, x_min, y_max, x_max)]
        are as described by `1`. If set to false the boxes won't be resized.
        - *labels*: Label of the predicted boxes.
        - *scores*: Score of the predicted boxes.
        - *num_valid_detections*: Number of boxes that we need to display. By default it will display all
        the boxes. This argument is useful whenever we are inference mode. The network perform padding
        operation and num_valid_detections is the value which allow to know which boxes were padded.
        - *resize*: Allow to resize the bounding boxes to the proper size. If set to true the inputs
        """
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        elif tf.is_tensor(labels):
            labels = labels.numpy().tolist()
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        elif tf.is_tensor(scores):
            scores = scores.numpy().tolist()
        if isinstance(num_valid_detections, np.ndarray):
            num_valid_detections = num_valid_detections.tolist()
        elif tf.is_tensor(num_valid_detections):
            num_valid_detections = num_valid_detections.numpy().tolist()

        for im, bb, cls, scr, nvd in zip(images, boxes, labels, scores, num_valid_detections):
            labels = [self._classes[int(ind)] for ind in cls]
            colors = [self._colors[int(ind)] for ind in cls]
            draw_bounding_boxes(im,
                                bb,
                                scores=scr,
                                labels=labels,
                                num_valid_detections=nvd,
                                resize=resize,
                                colors=colors)


def draw_bounding_boxes(image,
                        boxes,
                        labels: list = None,
                        scores: np.array = None,
                        num_valid_detections: int = None,
                        colors: List[str] = None,
                        resize=True):
    """Outputs a copy of images but draws on top of the pixels zero or more bounding boxes specified
    by the locations in boxes. The coordinates of the each bounding box in boxes are encoded as
    [y_min, x_min, y_max, x_max]. The bounding box coordinates can be:

    1. floats in [0.0, 1.0] relative to the width and height of the underlying image
    2. already resized to the corresponding size

    Arguments:

    - *image*: An image (tf.Tensor or numpy array)with shape [height, width, 3]
    - *boxes*: An array (tf.Tensor, or Numpy array) of boxes with the following shape [num_boxes, (y_min, x_min, y_max, x_max)]
    - *labels*: A list of string corresponding to the predicted label.
    - *scores*: A list of scores predicted
    - *num_valid_detections*: Number of boxes that we need to display. By default it will display all
    the boxes. This argument is useful whenever we are inference mode. The network perform padding
    operation and num_valid_detections is the value which allow to know which boxes were padded.
    - *colors*: A list of matplotlib colors of length = num_boxes
    - *resize*: Allow to resize the bounding boxes to the proper size. If set to true the inputs
    are as described by `1`. If set to false the boxes won't be resized.

    """
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if isinstance(boxes, tf.Tensor):
        boxes = boxes.numpy()

    plt.figure(figsize=(15, 15))
    plt.imshow(image.astype(np.uint8))
    axes = plt.gca()

    if isinstance(boxes, tf.Tensor):
        boxes = boxes.numpy()

    if resize:
        boxes[:, 0::2] *= image.shape[0]
        boxes[:, 1::2] *= image.shape[1]

    if num_valid_detections is not None:
        boxes = boxes[:num_valid_detections]

    for i, box in enumerate(boxes):
        x_y = (box[1], box[0])
        width = box[3] - box[1]
        height = box[2] - box[0]

        color = 'r' if colors is None else colors[i]
        # Draw the boxes
        r = patches.Rectangle(x_y, width, height, fill=False, edgecolor=color)
        axes.add_patch(r)

        if labels is not None:
            score = scores[i] if scores is not None else 1
            axes.annotate(f'{labels[i]}_{str(round(score, 4))}',
                          x_y,
                          color=color,
                          weight='bold',
                          fontsize=10,
                          ha='center',
                          va='center')
