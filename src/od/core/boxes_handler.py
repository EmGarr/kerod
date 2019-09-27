from enum import Enum
import tensorflow as tf


class BoxField(Enum):
    BOXES = 'boxes'
    SCORES = 'scores'
    LABELS = 'labels'
    MASKS = 'masks'
    WEIGHTS = 'weights'
    NUM_BOXES = 'num_boxes'


class BoxHandler():

    def __init__(self, boxes):
        self.data = {BoxField.BOXES: boxes}
        self._batch_size = tf.shape(boxes)[0]
        num_dim = len(tf.shape(boxes))
        if  num_dim != 3:
            raise ValueError(f'You provide boxes with shape {num_dim}. BoxHandler can only \
            handle boxes with shape [batch_size, num_boxes, 4] or [num_boxes, 4]')

    def add_field(self, field, field_data):
        self.data[field] = field_data

    @property
    def boxes(self):
        return self.data[BoxField.BOXES]

    @property
    def scores(self):
        return self.data[BoxField.SCORES]

    @property
    def masks(self):
        return self.data[BoxField.MASKS]

    @property
    def weights(self):
        return self.data[BoxField.WEIGHTS]

    @property
    def num_boxes(self):
        return self.data[BoxField.NUM_BOXES]

    @property
    def boxes_center_coordinates(self):
        y_min, x_min, y_max, x_max = tf.split(value=self.boxes, num_or_size_splits=4, axis=1)
        width = x_max - x_min
        height = y_max - y_min
        ycenter = y_min + height / 2.
        xcenter = x_min + width / 2.
        return tf.concat([ycenter, xcenter, height, width], axis=-1)

    def to_list(self):
        """ Return a list of dict representing a box with its information (BoxField)

        :returns: List of dict
        """
        num_boxes = self.data[BoxField.NUM_BOXES]
        boxes = [{} for _ in range(self._batch_size)]
        for field, data in self.data.items():
            if data != BoxField.NUM_BOXES:
                splitted_field = tf.split(data, num_or_size_splits=self._batch_size)
                for i in range(self._batch_size):
                    unpadded_field = splitted_field[i][:num_boxes[i]]
                    unpadded_field = tf.squeeze(unpadded_field, [0])
                    boxes[i][field] = unpadded_field
        return boxes 
