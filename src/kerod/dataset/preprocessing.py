import tensorflow as tf

from kerod.core import constants
from kerod.core.standard_fields import BoxField, DatasetField
from kerod.dataset.utils import filter_crowded_boxes, filter_bad_area
from kerod.dataset import augmentation as aug


def resize_to_min_dim(image, short_edge_length, max_dimension):
    """Resize an image given to the min size maintaining the aspect ratio.

    If one of the image dimensions is bigger than the max_dimension after resizing, it will scale
    the image such that its biggest dimension is equal to the max_dimension.

    Arguments :

    - *image*: A np.array of size [height, width, channels].
    - *short_edge_length*: minimum image dimension.
    - *max_dimension*: If the resized largest size is over max_dimension. Will use to max_dimension
    to compute the resizing ratio.

    Returns:
    - *resized_image*: The input image resized with the aspect_ratio preserved in float32

    Raises:

    ValueError: If the max_dimension is above `kerod.core.constants.MAX_IMAGE_SIZE`
    """
    if max_dimension > constants.MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"The max_dimension can only be inferior or equal to {constants.MAX_IMAGE_DIMENSION}")
    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    im_size_min = tf.minimum(height, width)
    im_size_max = tf.maximum(height, width)
    scale = short_edge_length / im_size_min
    # Prevent the biggest axis from being more than MAX_SIZE
    if tf.math.round(scale * im_size_max) > max_dimension:
        scale = max_dimension / im_size_max

    target_height = tf.cast(height * scale, dtype=tf.int32)
    target_width = tf.cast(width * scale, dtype=tf.int32)
    return tf.image.resize(tf.expand_dims(image, axis=0),
                           size=[target_height, target_width],
                           method=tf.image.ResizeMethod.BILINEAR)[0]


def preprocess(inputs, bgr=True, horizontal_flip=True, random_crop_size=None, padded_mask=False):
    """This operations performs a classical preprocessing operations for localization datasets:

    - COCO
    - Pascal Voc

    You can download easily those dataset using [tensorflow dataset](https://www.tensorflow.org/datasets/catalog/overview).

    Arguments:

    - *inputs*: It can be either a [FeaturesDict](https://www.tensorflow.org/datasets/api_docs/python/tfds/features/FeaturesDict) or a dict.
    but it should have the following structures.

    ```python
    inputs = FeaturesDict({
        'image': Image(shape=(None, None, 3), dtype=tf.uint8),
        'objects': Sequence({
            'area': Tensor(shape=(), dtype=tf.int64), # area
            'bbox': BBoxFeature(shape=(4,), dtype=tf.float32), # The values are between 0 and 1
            'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=80),
        }),
    })
    ```
    - *bgr*: Convert your input image to BGR (od.model.faster_rcnn.FasterRcnnFPNResnet50 needs it).
    If you have open your image with `tf.image.decode_image` will open an image in RGB. However,
    OpenCV will open it in BGR by default.

    - *horizontal_flip*: Activate the random horizontal flip.
    - *random_crop_size*: 1-D tensor with size the rank of `image` (e.g: (400, 600, 0)).
    - *padded_mask*: If set to true return a mask of 1 of the image. When padded
    we will know which parts is from the original image.

    Returns:

    - *inputs*:
        1. image: A 3D tensor of float32 and shape [None, None, 3]
        2. image_informations: A 1D tensor of float32 and shape [(height, width),]. It contains the shape
        of the image without any padding. It can be usefull if it followed by a `padded_batch` operations.
        The models needs those information in order to clip the boxes to the proper dimension.
        3. images_padding_mask: If padded_mask set to true return a 2D tensor of int8 and shape [None, None, 3].
        Mask of the image if a padding is performed we will know where the original image was.
    - *ground_truths*:
        1. BoxField.BOXES: A tensor of shape [num_boxes, (y1, x1, y2, x2)] and resized to the image shape
        2. BoxField.LABELS: A tensor of shape [num_boxes, ]
        3. BoxField.NUM_BOXES: A tensor of shape (). It is usefull to unpad the data in case of a batched training
    """

    image = inputs['image'][:, :, ::-1] if bgr else inputs['image']
    image = tf.cast(image, tf.float32)

    targets = inputs['objects']

    if horizontal_flip:
        image, targets[BoxField.BOXES] = aug.random_horizontal_flip(image, targets[BoxField.BOXES])

    if random_crop_size is not None:
        if tf.shape(image)[0] < random_crop_size[0] or tf.shape(image)[1] < random_crop_size[1]:
            image = resize_to_min_dim(image, max(random_crop_size), 1333.0)
        image, targets = aug.random_random_crop(image, random_crop_size, targets)

    if 'is_crowd' in targets:
        targets = filter_crowded_boxes(targets)

    targets = filter_bad_area(targets)

    image = resize_to_min_dim(image, 800.0, 1333.0)
    image_information = tf.cast(tf.shape(image)[:2], dtype=tf.float32)

    inputs = {DatasetField.IMAGES: image, DatasetField.IMAGES_INFO: image_information}
    if padded_mask:
        inputs[DatasetField.IMAGES_PMASK] = tf.ones((tf.shape(image)[0], tf.shape(image)[1]),
                                                    dtype=tf.int8)

    ground_truths = {
        BoxField.BOXES: targets[BoxField.BOXES] * tf.tile(image_information[tf.newaxis], [1, 2]),
        BoxField.LABELS: tf.cast(targets[BoxField.LABELS], tf.int32),
        BoxField.NUM_BOXES: tf.shape(targets[BoxField.LABELS]),
        BoxField.WEIGHTS: tf.fill(tf.shape(targets[BoxField.LABELS]), 1.0)
    }
    return inputs, ground_truths


def expand_dims_for_single_batch(inputs, ground_truths):
    """In order to train your model you need to add a batch dimension to the output of the preprocess
    function. For a single batch operation this method is faster:

    - `expand_dims`:

    ```python
    ds_train = tfds.load(name="voc", split="train", shuffle_files=True)
    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.map(expand_dims_for_single_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ```

    > Execution time: 0.002636657891998766

    - `batch`

    ```python
    ds_train = tfds.load(name="voc", split="train", shuffle_files=True)
    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.batch(1)
    ```

    > Execution time: 0.004332915792008862

    - `padded_batch`

    ```python
    ds_train = tfds.load(name="voc", split="train", shuffle_files=True)
    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.padded_batch(batch_size, padded_shapes=padded_shapes)
    ```

    > Execution time: 0.0055130551019974515

    Returns:

    - *inputs*: The features and the ground_truths are mixed together
        1. DatasetField.IMAGES: A 3D tensor of float32 and shape [None, None, 3]
        2. DatasetField.IMAGES_INFO: A 1D tensor of float32 and shape [(height, width),]. It contains the shape
        of the image without any padding. It can be usefull if it followed by a `padded_batch` operations.
        The models needs those information in order to clip the boxes to the proper dimension.

    - *ground_truths*:
        1. BoxField.BOXES: A tensor of shape [1, num_boxes, (y1, x1, y2, x2)] and resized to the image shape
        2. BoxField.LABELS: A tensor of shape [1, num_boxes, ]
        3. BoxField.NUM_BOXES: A tensor of shape [1, 1]. It is usefull to unpad the data in case of a batched training
        4. BoxField.WEIGHTS: A tensor of shape [1]
    """
    inputs = {
        DatasetField.IMAGES: inputs[DatasetField.IMAGES][None],
        DatasetField.IMAGES_INFO: inputs[DatasetField.IMAGES_INFO][None]
    }

    ground_truths = {
        BoxField.BOXES: ground_truths[BoxField.BOXES][None],
        BoxField.LABELS: ground_truths[BoxField.LABELS][None],
        BoxField.NUM_BOXES: ground_truths[BoxField.NUM_BOXES][None],
        BoxField.WEIGHTS: ground_truths[BoxField.WEIGHTS][None]
    }
    return inputs, ground_truths
