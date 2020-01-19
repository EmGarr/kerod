import numpy as np
import tensorflow as tf

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def draw_bounding_boxes(image,
                        boxes,
                        labels: list = None,
                        scores: np.array = None,
                        num_valid_detections: int = None,
                        resize=True):
    """Outputs a copy of images but draws on top of the pixels zero or more bounding boxes specified
    by the locations in boxes. The coordinates of the each bounding box in boxes are encoded as
    [y_min, x_min, y_max, x_max]. The bounding box coordinates can be:

    1. floats in [0.0, 1.0] relative to the width and height of the underlying image
    2. already resized to the corresponding size

    Arguments:

    - *image*: An image (tf.Tensor or numpy array)with shape [height, width, 3]
    - *boxes*: An array (tf.Tensor, or Numpy array) of boxes with the following shape [num_boxes, (y_min, x_min, y_max, x_max)] 
    - *resize*: Allow to resize the bounding boxes to the proper size. If set to true the inputs
    are as described by `1`. If set to false the boxes won't be resized.
    - *labels*: A list of string corresponding to the predicted label.
    - *scores*: A list of scores predicted
    - *num_valid_detections*: Number of boxes that we need to display. By default it will display all
    the boxes. This argument is useful whenever we are inference mode. The network perform padding
    operation and num_valid_detections is the value which allow to know which boxes were padded.
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

        # Draw the boxes
        r = patches.Rectangle(x_y, width, height, fill=False, edgecolor='r')
        axes.add_patch(r)

        if labels is not None:
            score = scores[i] if scores is not None else 1
            axes.annotate(f'{labels[i]}_{str(score)}',
                          x_y,
                          color='r',
                          weight='bold',
                          fontsize=10,
                          ha='center',
                          va='center')
