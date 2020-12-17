import tensorflow as tf

from tensorflow.python.keras.engine import data_adapter
from kerod.model.layers.transformer import Transformer
from kerod.core.matcher import hungarian_matching
from kerod.utils.training import apply_kernel_regularization
from kerod.model.backbone.resnet import ResNet50PytorchStyle
from kerod.core.standard_fields import BoxField
from kerod.core.target_assigner import TargetAssigner
from kerod.core.similarity import DetrSimilarity
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
            tf.keras.layers.Dense(4, activation='sigmoid')  # (x1, y1, x2, y2)
        ])
        self.class_embed = tf.keras.layers.Dense(num_classes + 1)

        # Will create a learnable embedding matrix for all our queries
        # It is a matrix of [num_queries, self.hidden_dim]
        # The embedding layers
        self.query_embed = tf.keras.layers.Embedding(num_queries, self.hidden_dim)
        self.all_the_queries = tf.range(num_queries)

        # Loss computation
        self.target_assigner = TargetAssigner(DetrSimilarity(), hungarian_matching,
                                              lambda gt, pred: gt)

        self.giou = GIoULoss(reduction=tf.keras.losses.Reduction.NONE)
        self.mae = MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        self.scc = SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

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
        classification_pred = tf.nn.softmax(self.class_embed(decoder_out))

        return classification_pred, localization_pred

    def compute_loss(self, ground_truths, y_pred):
        ground_truths = {
            # We add one because the background is not counted in ground_truths[BoxField.LABELS]
            BoxField.LABELS:
                ground_truths[BoxField.LABELS] + 1,
            BoxField.BOXES:
                ground_truths[BoxField.BOXES],
            BoxField.WEIGHTS:
                ground_truths[BoxField.WEIGHTS],
            BoxField.NUM_BOXES:
                ground_truths[BoxField.NUM_BOXES]
        }
        y_true, weights = self.target_assigner.assign(y_pred, ground_truths)
        giou = self.giou(y_true[BoxField.BOXES], y_pred[BoxField.BOXES],
                         sample_weight=weights[BoxField.BOXES])
        mae = self.mae(y_true[BoxField.BOXES],
                       y_pred[BoxField.BOXES],
                       sample_weight=weights[BoxField.BOXES])

        scc = self.scc(y_true[BoxField.LABELS],
                       y_pred[BoxField.LABELS],
                       sample_weight=weights[BoxField.LABELS])
        return tf.reduce_mean(tf.reduce_sum(giou * mae + scc, axis=1))

    def train_step(self, data):
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided. These utilities will be exposed
        # publicly.
        data = data_adapter.expand_1d(data)
        x, ground_truths, _ = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # All the losses are computed in the call. It's weird but it those the job
            # They are added automatically to self.losses
            loss = self.compute_loss(ground_truths, y_pred)

            reg_loss = self.compiled_loss(None, y_pred, None, regularization_losses=self.losses)
            loss += reg_loss
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        return {m.name: m.result() for m in self.metrics}


class DeTrResnet50Pytorch(DeTr):

    def __init__(self, num_classes, num_queries=100, **kwargs):
        resnet = ResNet50PytorchStyle(input_shape=[None, None, 3], weights='imagenet')
        super().__init__(num_classes, resnet, num_queries=num_queries, **kwargs)
