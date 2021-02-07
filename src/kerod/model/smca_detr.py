from typing import Dict

import tensorflow as tf
import tensorflow_addons as tfa
from kerod.core.box_ops import (convert_to_center_coordinates, convert_to_xyxy_coordinates)
from kerod.core.losses import L1Loss
from kerod.core.matcher import hungarian_matching
from kerod.core.similarity import DetrSimilarity
from kerod.core.standard_fields import BoxField, DatasetField
from kerod.core.target_assigner import TargetAssigner
from kerod.model.backbone.resnet import ResNet50, ResNet50PytorchStyle
from kerod.model.detr import compute_detr_metrics
from kerod.model.layers import (DynamicalWeightMaps, PositionEmbeddingSine, Transformer,
                                SMCAReferencePoints)
from kerod.model.post_processing.post_processing_detr import \
    post_processing as detr_postprocessing
from kerod.utils.documentation import remove_unwanted_doc
from tensorflow.python.keras.engine import data_adapter

__pdoc__ = {}


class SMCA(tf.keras.Model):
    """Build a single scale SCMA model according to the paper
    [Fast Convergence of DETR with Spatially Modulated Co-Attention](https://arxiv.org/pdf/2101.07448.pdf).

    In what is it different from DETR ?

    Just imagine that your object queries are learned anchors.
    Those learned "anchors" will modulate the attention map during
    the coattention stage of the decoder. They will help to target
    faster some sweet spots which leads to a speed up by 10
    of the training. It maintains the same performance than DETR.

    You can use it as follow:

    ```python
    model = SMCAR50(80)
    base_lr = 0.1
    optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr)
    model.compile(optimizer=optimizer, loss=None)
    model.fit(ds_train, validation_data=ds_test, epochs=11,)
    ```

    Arguments:
        num_classes: The number of classes of your dataset
            (**do not include the background class** it is handle for you)
        backbone: A vision model like ResNet50.
        num_queries: number of object queries, ie detection slot.
            This is the maximal number of objects
            SCMA can detect in a single image. For COCO, we recommend 300 queries.

    Call arguments:
        inputs: Tuple
            1. images: A 4-D tensor of float32 and shape [batch_size, None, None, 3]
            2. image_informations: A 1D tensor of float32 and shape [(height, width),].
                It contains the shape of the image without any padding.
            3. images_padding_mask: A 3D tensor of int8 and shape [batch_size, None, None]
                composed of 0 and 1 which allows to know where a padding has been applied.
        training: Is automatically set to `True` in train mode

    Call returns:
        logits: A Tensor of shape [batch_size, h, num_classes + 1] class logits
        boxes: A Tensor of shape [batch_size, h, 4]

    where h is num_queries * transformer_decoder.num_layers if
    training is true and num_queries otherwise.
    """

    def __init__(self, num_classes, backbone, num_queries=300, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = 256

        self.backbone = backbone
        self.input_proj = tf.keras.layers.Conv2D(self.hidden_dim, 1)
        self.pos_embed = PositionEmbeddingSine(output_dim=self.hidden_dim)
        num_heads = 8
        self.transformer_num_layers = 6
        self.transformer = Transformer(num_layers=self.transformer_num_layers,
                                       d_model=self.hidden_dim,
                                       num_heads=num_heads,
                                       dim_feedforward=2048)

        # MCMA layers
        self.dyn_weight_map = DynamicalWeightMaps()
        self.ref_points = SMCAReferencePoints(self.hidden_dim, num_heads)

        self.bbox_embed = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dense(4, dtype=tf.float32)  # (x1, y1, x2, y2)
        ])
        self.class_embed = tf.keras.layers.Dense(num_classes + 1, dtype=tf.float32)

        # Will create a learnable embedding matrix for all our queries
        # It is a matrix of [num_queries, self.hidden_dim]
        # The embedding layers
        self.query_embed = tf.keras.layers.Embedding(num_queries, self.hidden_dim)
        self.all_the_queries = tf.range(num_queries)

        # Loss computation
        self.weight_class, self.weight_l1, self.weight_giou = 2, 5, 2
        similarity_func = DetrSimilarity(self.weight_class, self.weight_l1, self.weight_giou)
        self.target_assigner = TargetAssigner(similarity_func,
                                              hungarian_matching,
                                              lambda gt, pred: gt,
                                              negative_class_weight=1.0)

        # Losses
        self.giou = tfa.losses.GIoULoss(reduction=tf.keras.losses.Reduction.NONE)
        self.l1 = L1Loss(reduction=tf.keras.losses.Reduction.NONE)
        self.focal_loss = tfa.losses.SigmoidFocalCrossEntropy(
            alpha=0.25, gamma=2, reduction=tf.keras.losses.Reduction.NONE, from_logits=True)

        # Metrics
        self.giou_metric = tf.keras.metrics.Mean(name="giou_last_layer")
        self.l1_metric = tf.keras.metrics.Mean(name="l1_last_layer")
        self.focal_loss_metric = tf.keras.metrics.Mean(name="focal_loss_last_layer")
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.precision_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        # Object recall = foreground
        self.recall_metric = tf.keras.metrics.Mean(name="object_recall")

    @property
    def metrics(self):
        return [
            self.loss_metric, self.giou_metric, self.l1_metric, self.focal_loss_metric,
            self.precision_metric, self.recall_metric
        ]

    def call(self, inputs, training=None):
        """Perform an inference in training.

        Arguments:
            inputs: Tuple
                1. images: A 4-D tensor of float32 and shape [batch_size, None, None, 3]
                2. image_informations: A 1D tensor of float32 and shape [(height, width),].
                    It contains the shape of the image without any padding.
                3. images_padding_mask: A 3D tensor of int8 and shape
                    [batch_size, None, None] composed of 0 and 1 which allows
                    to know where a padding has been applied.
            training: Is automatically set to `True` in train mode

        Returns:
            logits: A Tensor of shape [batch_size, num_queries, num_classes + 1] class logits
            boxes: A Tensor of shape [batch_size, num_queries, 4]

        where h is num_queries * transformer_decoder.num_layers if
        training is true and num_queries otherwise.
        """
        images = inputs[DatasetField.IMAGES]
        images_padding_masks = inputs[DatasetField.IMAGES_PMASK]
        batch_size = tf.shape(images)[0]
        # The preprocessing dedicated to the backbone is done inside the model.
        x = self.backbone(images)[-1]
        features_mask = tf.image.resize(tf.cast(images_padding_masks[..., None], tf.float32),
                                        tf.shape(x)[1:3],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        features_mask = tf.cast(features_mask, tf.bool)
        # Positional_encoding for the backbone
        pos_embed = self.pos_embed(features_mask)
        # [batch_size, num_queries, self.hidden_dim]
        all_the_queries = tf.tile(self.all_the_queries[None], (batch_size, 1))
        # [batch_size, num_queries, self.hidden_dim]
        query_embed = self.query_embed(all_the_queries)
        h_backbone_out, w_backbone_out = tf.shape(x)[1], tf.shape(x)[2]
        x = self.input_proj(x)

        # Flatten the position embedding and the spatial tensor
        # to allow the preprocessing by the Transformer
        # [batch_size, h * w,  self.hidden_dim]
        x = tf.reshape(x, (batch_size, -1, self.hidden_dim))
        pos_embed = tf.reshape(pos_embed, (batch_size, -1, self.hidden_dim))
        # Flatten the padding masks
        features_mask = tf.reshape(features_mask, (batch_size, -1))

        ref_points, ref_points_presigmoid = self.ref_points(query_embed)

        # dyn_weight_map_per_head = G in the paper
        dyn_weight_map_per_head = self.dyn_weight_map(h_backbone_out, w_backbone_out, ref_points)
        dyn_weight_map_per_head = tf.math.log(dyn_weight_map_per_head + 10e-4)  # log G

        decoder_out, _ = self.transformer(x,
                                          pos_embed,
                                          query_embed,
                                          key_padding_mask=features_mask,
                                          coattn_mask=dyn_weight_map_per_head,
                                          training=training)

        logits = self.class_embed(decoder_out)
        boxes = self.bbox_embed(decoder_out)

        if training:
            # In training all the outputs of the decoders are stacked together.
            # We tile the reference_points to match those outputs
            ref_points_presigmoid = tf.tile(ref_points_presigmoid,
                                            (1, self.transformer_num_layers, 1))

        # Add initial center to constrain  the bounding boxes predictions
        offset = tf.concat(
            [ref_points_presigmoid,
             tf.zeros((batch_size, tf.shape(ref_points_presigmoid)[1], 2))],
            axis=-1)
        boxes = tf.nn.sigmoid(boxes + offset)

        return {
            BoxField.SCORES: logits,
            BoxField.BOXES: boxes,
        }

    def compute_loss(
        self,
        ground_truths: Dict[str, tf.Tensor],
        y_pred: Dict[str, tf.Tensor],
        input_shape: tf.Tensor,
    ) -> int:
        """Apply the GIoU, L1 and SCC to each layers of the transformer decoder

        Args:
            ground_truths: see output kerod.dataset.preprocessing for the doc
            y_pred: A dict
                - *scores: A Tensor of shape [batch_size, num_queries, num_classes + 1] class logits
                - *bbox*: A Tensor of shape [batch_size, num_queries, 4]
            input_shape: [height, width] of the input tensor.
                It is the shape of the images will all the padding included.
                It is used to normalize the ground_truths boxes.
        """
        normalized_boxes = ground_truths[BoxField.BOXES] / tf.tile(input_shape[None], [1, 2])
        centered_normalized_boxes = convert_to_center_coordinates(normalized_boxes)
        ground_truths = {
            # We add one because the background is not counted in ground_truths [BoxField.LABELS]
            BoxField.LABELS:
                ground_truths[BoxField.LABELS] + 1,
            BoxField.BOXES:
                centered_normalized_boxes,
            BoxField.WEIGHTS:
                ground_truths[BoxField.WEIGHTS],
            BoxField.NUM_BOXES:
                ground_truths[BoxField.NUM_BOXES]
        }
        boxes_per_lvl = tf.split(y_pred[BoxField.BOXES], self.transformer_num_layers, axis=1)
        logits_per_lvl = tf.split(y_pred[BoxField.SCORES], self.transformer_num_layers, axis=1)

        y_pred_per_lvl = [{
            BoxField.BOXES: boxes,
            BoxField.SCORES: logits
        } for boxes, logits in zip(boxes_per_lvl, logits_per_lvl)]

        num_boxes = tf.cast(tf.reduce_sum(ground_truths[BoxField.NUM_BOXES]), tf.float32)
        loss = 0
        # Compute the Giou, L1 and SCC at each layers of the transformer decoder
        for i, y_pred in enumerate(y_pred_per_lvl):
            # Logs the metrics for the last layer of the decoder
            compute_metrics = i == self.transformer_num_layers - 1
            loss += self._compute_loss(y_pred,
                                       ground_truths,
                                       num_boxes,
                                       compute_metrics=compute_metrics)
        return loss

    def _compute_loss(
        self,
        y_pred: Dict[str, tf.Tensor],
        ground_truths: Dict[str, tf.Tensor],
        num_boxes: int,
        compute_metrics=False,
    ):
        y_true, weights = self.target_assigner.assign(y_pred, ground_truths)

        # Caveats GIoU is buggy and if the batch_size is 1 and the sample_weight
        # is provided will raise an error
        giou = self.giou(convert_to_xyxy_coordinates(y_true[BoxField.BOXES]),
                         convert_to_xyxy_coordinates(y_pred[BoxField.BOXES]),
                         sample_weight=weights[BoxField.BOXES])

        l1 = self.l1(y_true[BoxField.BOXES],
                     y_pred[BoxField.BOXES],
                     sample_weight=weights[BoxField.BOXES])

        cls_labels = tf.one_hot(
            y_true[BoxField.LABELS],
            depth=self.num_classes + 1,
            dtype=tf.float32,
        )
        focal_loss = self.focal_loss(cls_labels,
                                     y_pred[BoxField.SCORES],
                                     sample_weight=weights[BoxField.LABELS])

        giou = self.weight_giou * tf.reduce_sum(giou) / num_boxes

        l1 = self.weight_l1 * tf.reduce_sum(l1) / num_boxes

        focal_loss = self.weight_class * tf.reduce_sum(focal_loss) / tf.reduce_sum(
            weights[BoxField.LABELS])

        if compute_metrics:
            self.giou_metric.update_state(giou)
            self.l1_metric.update_state(l1)
            self.focal_loss_metric.update_state(focal_loss)
            self.precision_metric.update_state(y_true[BoxField.LABELS],
                                               y_pred[BoxField.SCORES],
                                               sample_weight=weights[BoxField.LABELS])

            recall = compute_detr_metrics(y_true[BoxField.LABELS], y_pred[BoxField.SCORES])
            self.recall_metric.update_state(recall)
        return giou + l1 + focal_loss

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, ground_truths, _ = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            input_shape = tf.cast(tf.shape(x[DatasetField.IMAGES])[1:3], self.compute_dtype)
            loss = self.compute_loss(ground_truths, y_pred, input_shape)

            loss += self.compiled_loss(None, y_pred, None, regularization_losses=self.losses)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.loss_metric.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        x, ground_truths, _ = data_adapter.unpack_x_y_sample_weight(data)

        # To compute the loss we need to get the results of each decoder layer
        # Setting training to True will provide it
        y_pred = self(x, training=True)
        input_shape = tf.cast(tf.shape(x[DatasetField.IMAGES])[1:3], self.compute_dtype)
        loss = self.compute_loss(ground_truths, y_pred, input_shape)
        loss += self.compiled_loss(None, y_pred, None, regularization_losses=self.losses)
        self.loss_metric.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        """Perform an inference and returns the boxes, scores and labels associated.
        Background is discarded the max and argmax operation are performed.
        It means that if background was predicted the second maximum score would
        be outputed.

        Example: background + 3 classes
        [0.54, 0.40, 0.03, 0.03] => score = 0.40, label = 0 (1 - 1)


        "To optimize for AP, we override the prediction of these slots
        with the second highest scoring class, using the corresponding confidence"
        Part 4. Experiments of Object Detection with Transformers

        Returns:
            boxes: A Tensor of shape [batch_size, self.num_queries, (y1,x1,y2,x2)]
                containing the boxes with the coordinates between 0 and 1.
            scores: A Tensor of shape [batch_size, self.num_queries] containing
                the score of the boxes.
            classes: A Tensor of shape [batch_size, self.num_queries]
                containing the class of the boxes [0, num_classes).
        """
        data = data_adapter.expand_1d(data)
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        y_pred = self(x, training=False)
        boxes_without_padding, scores, labels = detr_postprocessing(
            y_pred[BoxField.BOXES],
            y_pred[BoxField.SCORES],
            x[DatasetField.IMAGES_INFO],
            tf.shape(x[DatasetField.IMAGES])[1:3],
        )
        return boxes_without_padding, scores, labels


class SMCAR50(SMCA):

    def __init__(self, num_classes, num_queries=100, **kwargs):
        resnet = ResNet50(input_shape=[None, None, 3], weights='imagenet')
        super().__init__(num_classes, resnet, num_queries=num_queries, **kwargs)


class SMCAR50Pytorch(SMCA):

    def __init__(self, num_classes, num_queries=100, **kwargs):
        resnet = ResNet50PytorchStyle(input_shape=[None, None, 3], weights='imagenet')
        super().__init__(num_classes, resnet, num_queries=num_queries, **kwargs)


remove_unwanted_doc(SMCA, __pdoc__)
remove_unwanted_doc(SMCAR50, __pdoc__)
remove_unwanted_doc(SMCAR50Pytorch, __pdoc__)
