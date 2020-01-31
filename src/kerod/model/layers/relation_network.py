import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras.losses import BinaryCrossentropy

from od.core.argmax_matcher import ArgMaxMatcher
from od.core.box_coder import encode_boxes_faster_rcnn
from od.core.box_ops import compute_dimensional_relative_geometry, compute_iou
from od.core.standard_fields import BoxField, LossField
from od.core.target_assigner import TargetAssigner, batch_assign_targets


def angle_defn(pos, i, d_model_size):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model_size))
    return pos * angle_rates


def positional_encoding(position, d_model_size, dtype=tf.float32):
    """Positional encoding is added to give the model some information about the relative position
    of the item among the others. The positional encoding vector is added to the embedding vector.
    Embeddings represent a token in a d-dimensional space where tokens with similar meaning will be
    closer to each other. But the embeddings do not encode the relative position of an item among the
    other. So after adding the positional encoding, items will be closer to each other based
    on the similarity of their meaning and their position, in the d-dimensional space.
    """

    # create the sinusoidal pattern for the positional encoding
    angle_rads = angle_defn(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model_size)[np.newaxis, :], d_model_size)

    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = tf.cast(np.concatenate([sines, cosines], axis=-1), dtype=dtype)
    return pos_encoding


class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding is added to give the model some information about the relative position
    of the item among the others. The positional encoding vector is added to the embedding vector.
    Embeddings represent a token in a d-dimensional space where tokens with similar meaning will be
    closer to each other. But the embeddings do not encode the relative position of an item among the
    other. So after adding the positional encoding, items will be closer to each other based
    on the similarity of their meaning and their position, in the d-dimensional space.

    Arguments:

    - *units*: Positive integer, dimensionality of the output space of the relation_feature module.
    """

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self._units = units

    def build(self, input_shape):
        num_boxes = input_shape[1]
        self.encodings = [positional_encoding(i, self._units, dtype=self.dtype) for i in range(num_boxes)]
        # Preload them
        self.encodings = tf.stack(self.encodings)

        super().build(input_shape)

    def call(self, scores):
        """

        Argument:

        - scores: A tensor of float32 and shape [batch, num_boxes]
        """
        arg_ordered_scores = tf.argsort(scores,
                                        axis=-1,
                                        direction='DESCENDING',
                                        stable=False,
                                        name=None)
        encoding = tf.gather(self.encodings, arg_ordered_scores)
        return tf.stop_gradient(encoding)

    def get_config(self):
        base_config = super().get_config()
        base_config['units'] = self._units
        return base_config


class RelationFeature(tf.keras.layers.Layer):
    """Implementation of object relation module from the paper [Relation Networks for Object Detection](https://arxiv.org/pdf/1711.11575.pdf)
    It processes a set of objects simultaneously through interaction between their
    appearance feature and geometry,  thus  allowing  modeling  of  their  relations.

    Arguments:

    - *units*: Positive integer, dimensionality of the output space of the relation_feature module. 
    """

    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self._units = units
        self.relative_geometry_dense = KL.Dense(units)
        self.geometry_dense_and_relu = KL.Dense(1, activation='relu')

        self.weights_key = KL.Dense(units, name='Wk')
        self.weights_question = KL.Dense(units, name='Wq')
        self.weights_v = KL.Dense(units, name='Wv')

    def call(self, inputs):
        """It processes a set of objects simultaneously through interaction between their
        appearance feature and geometry,  thus  allowing  modeling  of  their  relations.

        Arguments:

        1. *feature_map*: A tensor of float32 and shape [batch, nb_boxes, nb_channel]
        2. *boxes*: A tensor of float32 and shape [batch, nb_boxes, (x_min, y_min, x_max, y_max)]

        Return:

        A tensor of float32 and shape [batch, nb_boxes, self._units]
        """
        feature_map = inputs[0]
        boxes = inputs[1]

        # shape = [batch, nb_boxes, nb_boxes]
        geometry_weights = self.compute_geometry_weights(boxes)

        # shape = [batch, nb_boxes, nb_boxes]
        appearance_weights = self.compute_appearance_weights(feature_map)

        # shape = [batch, nb_boxes, nb_boxes]
        weights = tf.maximum(geometry_weights * tf.exp(appearance_weights), 1e-4)

        #It computes the impact from each objects on the others
        relation_weights = weights / tf.expand_dims(tf.reduce_sum(weights, axis=2), axis=2)

        # shape = [batch, nb_boxes, self._units]
        return tf.matmul(relation_weights, self.weights_v(feature_map))

    def compute_geometry_weights(self, boxes):
        """
        Arguments:

        - *boxes*: A tensor of float32 and shape [batch, nb_boxes, (x_min, y_min, x_max, y_max)]

        Return:

        A tensor of float32 and shape [batch, nb_boxes, nb_boxes]
        """
        # [batch_size, nb_boxes, nb_boxes, 4]
        relative_geometry = compute_dimensional_relative_geometry(boxes)
        # TODO use positional_encoding
        # [batch_size, nb_boxes, nb_boxes, units]
        embed_relative_geometry = self.relative_geometry_dense(relative_geometry)
        # The zero trimming operation restricts relations only between
        # object of certain geometric relation ship.
        # [batch_size, nb_boxes, nb_boxes, 1]
        geometry_weights = self.geometry_dense_and_relu(embed_relative_geometry)
        return tf.squeeze(geometry_weights, axis=-1)

    def compute_appearance_weights(self, feature_map):
        """This operation project feature_map_m and feature_map_n into a subspace to measure how well
        they match.

        Arguments:

        - *feature_map*: A tensor of float32 and shape [batch, nb_boxes, nb_channel]

        Return:

        A tensor of float32 and shape [batch, nb_boxes, nb_boxes] which represent how
        well the boxes match
        """
        key = self.weights_key(feature_map)
        question = self.weights_question(feature_map)
        return tf.matmul(key, question, transpose_b=True) / tf.sqrt(
            tf.constant(self._units, self.dtype))


class DuplicateRemovalLayer(tf.keras.layers.Layer):
    """The duplicate removal layer is a module which learn to perfom a smart NMS.

    Arguments:

    - *iou_threshold*: TODO EXPLAIN THE UTILITY OF IOU_THRESHOLD
    - *units*: Positive integer, dimensionality of the internal space of the duplicate removal.
    The default value (128) is from the [paragraph 4.3 in the paper](https://arxiv.org/pdf/1711.11575.pdf).
    - *serving*: Will allow to bypass the save_model behavior the graph in serving mode.
        Currently, the issue is that in training the ground_truths are passed to the call method but
        not in inference. For the serving only the `images` and `images_information` are defined.
        It means the inputs link to the ground_truths won't be defined in serving. However, in tensorflow
        when the `training` arguments is defined int the method `call`, `tf.save_model.save` method
        performs a check on the graph for training=False and training=True.
        However, we don't want this check to be perform because our ground_truths inputs aren't defined.

    ```python
    def call(self, inputs, training=None):
        tensor = inputs[0] # your tensors
        if training:
            ground_truths = inputs[1] # inputs 1 is not defined so an exception is raised
    ```

    to avoid that the serving arguments is used

    ```python
    def call(self, inputs, training=None):
        tensor = inputs[0] # your tensors
        if training and not self.serving:
            ground_truths = inputs[1] # inputs 1 is not defined so an exception is raised
    ```

    TODO find a better way than serving to avoid this issue
    """

    def __init__(self, iou_threshold, units=128, serving=False, **kwargs):
        super().__init(**kwargs)
        self._units = units
        self.serving = serving
        # Weights
        self._weighted_scores = KL.Dense(1)
        self._weights_score_rank = KL.Dense(units)
        self._weights_feature = KL.Dense(units)
        self._relation_feature = RelationFeature(units=units)
        self._feature_embedding = KL.Dense(units)
        self._scores_positional_encoding = PositionalEncoding(units)

        # Loss
        self._iou_threshold = iou_threshold
        self._target_assigner = TargetAssigner(compute_iou,
                                               ArgMaxMatcher(iou_threshold),
                                               encode_boxes_faster_rcnn,
                                               dtype=self._compute_dtype)

    def call(self, inputs, training=None):
        """It processes a set of objects simultaneously through interaction between their
        appearance feature and geometry,  thus  allowing  modeling  of  their  relations.

        Arguments:

        1. *feature_map*: A tensor of float32 and shape [batch, nb_boxes, nb_channel]
        2. *boxes*: A tensor of float32 and shape [batch, nb_boxes, (x_min, y_min, x_max, y_max)]
        3. *scores*: TODO

        Return:

        A tensor of float32 and shape [batch, nb_boxes, self._units]
        """

        feature_map = inputs[0]
        boxes = inputs[1]
        scores = inputs[2]

        embedded_scores = self._scores_positional_encoding(scores)
        embeddeding = self._weights_score_rank(embedded_scores) + self._weights_feature(feature_map)

        # [batch_size, num_boxes, self.units]
        embedding_feature_and_geometry = self._relation_feature([embeddeding, boxes])
        # [batch_size, num_boxes, 1]
        logits = self._weighted_scores(embedding_feature_and_geometry)

        if training and not self.serving:
            ground_truths = inputs[3]
            loss = self.compute_loss(logits, boxes, ground_truths)
            self.add_metric(loss, name=f'{self.name}_at_{self._iou_threshold}', aggregation='mean')
            self.add_loss(loss)

        return scores * tf.keras.activations.sigmoid(logits)

    def compute_loss(self, scores_logits, boxes, ground_truths):
        gt_boxes = tf.unstack(ground_truths[BoxField.BOXES])
        num_boxes = tf.unstack(ground_truths[BoxField.NUM_BOXES])
        ground_truths = [{BoxField.BOXES: b[:nb[0]]} for b, nb in zip(gt_boxes, num_boxes)]
        y_true, weights, _ = batch_assign_targets(self._target_assigner, boxes, ground_truths)

        loss = BinaryCrossentropy(from_logits=True)(
            y_true[LossField.CLASSIFICATION],
            scores_logits,
            sample_weights=weights[LossField.CLASSIFICATION])
        return loss

    def get_config(self):
        base_config = super().get_config()
        base_config['iou_threshold'] = self._iou_threshold
        base_config['units'] = self._units
        base_config['serving'] = self.serving
        return base_config
