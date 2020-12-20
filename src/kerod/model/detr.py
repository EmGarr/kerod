import tensorflow as tf
import numpy as np

from tensorflow.python.keras.engine import data_adapter
from kerod.model.layers.transformer import Transformer
from kerod.core.matcher import hungarian_matching
from kerod.utils.training import apply_kernel_regularization
from kerod.model.backbone.resnet import ResNet50PytorchStyle
from kerod.core.standard_fields import BoxField, DatasetField
from kerod.core.target_assigner import TargetAssigner
from kerod.core.similarity import DetrSimilarity
from kerod.core.box_ops import convert_to_center_coordinates
from kerod.model.layers.positional_encoding import PositionEmbeddingLearned
from tensorflow_addons.losses.giou_loss import GIoULoss
from tensorflow.keras.losses import MeanAbsoluteError, SparseCategoricalCrossentropy


class DeTr(tf.keras.Model):
    """Build a DeTr model according to the paper
    [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    You can use it as follow:

    ```python
    model = Detr(80)
    base_lr = 0.1
    optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr)
    model_faster_rcnn.compile(optimizer=optimizer, loss=None)
    model_faster_rcnn.fit(ds_train, validation_data=ds_test, epochs=11,)
    ```

    Arguments:

    - *num_classes*: The number of classes of your dataset
    (**do not include the background class** it is handle for you)
    - *num_queries*: number of object queries, ie detection slot. This is the maximal number of objects
    DETR can detect in a single image. For COCO, we recommend 100 queries.

    """

    def __init__(self, num_classes, backbone, num_queries=100, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = 256

        self.l2 = tf.keras.regularizers.l2(1e-4)
        self.backbone = backbone
        self.input_proj = tf.keras.layers.Conv2D(self.hidden_dim, 1)
        self.pos_embed = PositionEmbeddingLearned(output_dim=256)
        self.transformer = Transformer(d_model=self.hidden_dim)

        self.bbox_embed = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dense(4, activation='sigmoid', dtype=tf.float32)  # (x1, y1, x2, y2)
        ])
        self.class_embed = tf.keras.layers.Dense(num_classes + 1, dtype=tf.float32)

        # Will create a learnable embedding matrix for all our queries
        # It is a matrix of [num_queries, self.hidden_dim]
        # The embedding layers
        self.query_embed = tf.keras.layers.Embedding(num_queries, self.hidden_dim)
        self.all_the_queries = tf.range(num_queries)

        # Loss computation
        self.target_assigner = TargetAssigner(DetrSimilarity(), hungarian_matching,
                                              lambda gt, pred: gt)

        # Relative classification weight applied to the no-object category
        # It down-weight the log-probability term of a no-object
        # by a factor 10 to account for class imbalance
        self.eos_coef = 0.1
        self.eos_vector = np.ones((self.num_classes + 1))
        self.eos_vector[0] = self.eos_coef
        self.eos_vector = tf.constant(self.eos_vector, dtype=tf.float32)

        # Losses
        self.giou = GIoULoss(reduction=tf.keras.losses.Reduction.NONE)
        self.mae = MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        self.scc = SparseCategoricalCrossentropy(from_logits=True)

        # Metrics
        self.giou_metric = tf.keras.metrics.Mean(name="giou")
        self.mae_metric = tf.keras.metrics.Mean(name="mae")
        self.scc_metric = tf.keras.metrics.Mean(name="scc")
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.precision_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.recall_metric = tf.keras.metrics.Mean(name="recall")

    @property
    def metrics(self):
        return [
            self.loss_metric, self.giou_metric, self.mae_metric, self.scc_metric,
            self.precision_metric, self.recall_metric
        ]

    def call(self, images, training=None):
        """Perform an inference in training.

        Arguments:

        - *images*: A Tensor of shape [batch_size, height, width, 3]

        - *training*: Is automatically set to `True` in train mode

        Returns:

        - *classification_pred*: A Tensor of shape [batch_size, num_queries, num_classes + 1] representig
        the class probability.
        - *localization_pred*: A Tensor of shape [batch_size, num_queries, 4]
        """
        batch_size = tf.shape(images)[0]
        # The preprocessing dedicated to the backbone is done inside the model.
        x = self.backbone(images)[-1]
        if training:
            apply_kernel_regularization(self.l2, self.backbone)

        # Add positional_encoding to backbon
        pos_embed = self.pos_embed(x)
        # [batch_size, num_queries, self.hidden_dim]
        all_the_queries = tf.tile(self.all_the_queries[None], (batch_size, 1))
        # [batch_size, num_queries, self.hidden_dim]
        query_embed = self.query_embed(all_the_queries)
        # add positional_encoding to x [batch_size, h, w, self.hidden_dim]
        x = self.input_proj(x)

        # Flatten the position embedding and the spatial tensor
        # to allow the preprocessing by the Transformer
        # [batch_size, h * w,  self.hidden_dim]
        x = tf.reshape(x, (batch_size, -1, self.hidden_dim))
        pos_embed = tf.reshape(pos_embed, (batch_size, -1, self.hidden_dim))

        decoder_out, _ = self.transformer((x, None, pos_embed, query_embed))
        localization_pred = self.bbox_embed(decoder_out)
        classification_pred = self.class_embed(decoder_out)

        return {
            BoxField.LABELS: classification_pred,
            BoxField.BOXES: localization_pred,
        }

    def compute_loss(self, ground_truths, y_pred, input_shape):
        normalized_boxes = ground_truths[BoxField.BOXES] / tf.tile(
            tf.expand_dims(input_shape, axis=0), [1, 2])

        ground_truths = {
            # We add one because the background is not counted in ground_truths [BoxField.LABELS]
            BoxField.LABELS:
                ground_truths[BoxField.LABELS] + 1,
            BoxField.BOXES:
                normalized_boxes,
            BoxField.WEIGHTS:
                ground_truths[BoxField.WEIGHTS],
            BoxField.NUM_BOXES:
                ground_truths[BoxField.NUM_BOXES]
        }
        y_true, weights = self.target_assigner.assign(y_pred, ground_truths)

        num_boxes = tf.cast(tf.reduce_sum(ground_truths[BoxField.NUM_BOXES]), tf.float32)
        # Reduce the class imbalanced by applying dividing the weights
        # by self.eos for the non object (pos 0)
        weights[BoxField.LABELS] = weights[BoxField.LABELS] * self.eos_vector
        # Caveats GIoU is buggy and if the batch_size is 1 and the sample_weight
        # is provided will raise an error
        giou = self.giou(y_true[BoxField.BOXES],
                         y_pred[BoxField.BOXES],
                         sample_weight=weights[BoxField.BOXES])

        # L1 with coordinates in y_cent, x_cent, w, h
        mae = self.mae(convert_to_center_coordinates(y_true[BoxField.BOXES]),
                       convert_to_center_coordinates(y_pred[BoxField.BOXES]),
                       sample_weight=weights[BoxField.BOXES])

        # SparseCategoricalCrossentropy
        scc = self.scc(y_true[BoxField.LABELS],
                       y_pred[BoxField.LABELS],
                       sample_weight=weights[BoxField.LABELS])
        self.scc_metric.update_state(scc)

        giou = tf.reduce_sum(giou) / num_boxes
        self.giou_metric.update_state(giou)

        mae = tf.reduce_sum(mae) / num_boxes
        self.mae_metric.update_state(mae)

        recall = compute_detr_metrics(y_true[BoxField.LABELS], y_pred[BoxField.LABELS])
        self.recall_metric.update_state(recall)

        self.precision_metric.update_state(y_true[BoxField.LABELS],
                                           y_pred[BoxField.LABELS],
                                           sample_weight=weights[BoxField.LABELS])

        return giou, mae, scc

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, ground_truths, _ = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(x[DatasetField.IMAGES], training=True)
            # They are added automatically to self.losses
            # Normalize the boxes between 0 and 1
            input_shape = tf.cast(tf.shape(x[DatasetField.IMAGES])[1:3], self.compute_dtype)
            giou, mae, scc = self.compute_loss(ground_truths, y_pred, input_shape)

            reg_loss = self.compiled_loss(None, y_pred, None, regularization_losses=self.losses)
            loss = reg_loss + giou + mae + scc

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.loss_metric.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        x, ground_truths, _ = data_adapter.unpack_x_y_sample_weight(data)

        y_pred = self(x[DatasetField.IMAGES], training=False)
        # They are added automatically to self.losses
        # Normalize the boxes between 0 and 1
        input_shape = tf.cast(tf.shape(x[DatasetField.IMAGES])[1:3], self.compute_dtype)
        giou, mae, scc = self.compute_loss(ground_truths, y_pred, input_shape)

        reg_loss = self.compiled_loss(None, y_pred, None, regularization_losses=self.losses)
        loss = reg_loss + giou + mae + scc
        self.loss_metric.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        data = data_adapter.expand_1d(data)
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        return self(x[DatasetField.IMAGES], training=False)


class DeTrResnet50Pytorch(DeTr):

    def __init__(self, num_classes, num_queries=100, **kwargs):
        resnet = ResNet50PytorchStyle(input_shape=[None, None, 3], weights='imagenet')
        super().__init__(num_classes, resnet, num_queries=num_queries, **kwargs)


def compute_detr_metrics(y_true: tf.Tensor, y_pred: tf.Tensor):
    """Useful metrics that allows to track how behave the training.

    Arguments:

    - *y_true*: A one-hot encoded vector with shape [batch_size, num_object_queries, num_classes]
    - *y_pred*: A tensor with shape [batch_size, num_object_queries, num_classes],
    representing the classification logits.

    Returns:

    - *recall*: Among all the boxes that we had to find how much did we found.
    """
    #Even if the softmax has not been applyed the argmax can be usefull
    prediction = tf.argmax(y_pred, axis=-1, name='label_prediction', output_type=tf.int32)
    correct = tf.cast(prediction == y_true, tf.float32)
    # Compute accuracy and false negative on all the foreground boxes
    fg_inds = tf.where(y_true > 0)

    recall = tf.reduce_mean(tf.gather_nd(correct, fg_inds), name='recall')
    return recall