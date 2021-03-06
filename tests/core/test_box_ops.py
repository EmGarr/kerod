import tensorflow as tf
import numpy as np

from kerod.core import box_ops


def test_transform_fpcoor_for_tf():
    """This test make sure the extracted boxes are properly aligned. Please read this thread
    for more information:
    https://github.com/tensorflow/tensorflow/issues/26278

    """
    arr = np.arange(25).astype('float32').reshape(5, 5)
    input4D = tf.reshape(arr, [1, 5, 5, 1])
    resize = tf.image.resize(input4D, [10, 10], method='bilinear')[0, :, :, 0]
    # We are targeting the box [1, 1, 3, 3]
    expected_crop_value = resize[2:6, 2:6]
    normalized_boxes = box_ops.transform_fpcoor_for_tf(tf.constant([[1, 1, 3, 3]], tf.float32),
                                                       [5, 5], [4, 4])
    crop = tf.image.crop_and_resize(input4D, normalized_boxes, [0], [4, 4])
    crop = tf.reshape(crop, (4, 4))
    assert np.array_equal(crop.numpy(), expected_crop_value.numpy())


def test_convert_to_center_coordinates():
    boxes = tf.constant([[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]])
    centers_sizes = box_ops.convert_to_center_coordinates(boxes)
    expected_centers_sizes = np.array([[15, 12.5, 10, 5], [0.35, 0.25, 0.3, 0.3]])
    np.testing.assert_allclose(centers_sizes, expected_centers_sizes)
    boxes_out = box_ops.convert_to_xyxy_coordinates(centers_sizes)
    np.testing.assert_allclose(boxes_out, boxes)


def test_compute_area():
    boxes = tf.constant([[0.0, 0.0, 10.0, 20.0], [1.0, 2.0, 3.0, 4.0]])
    exp_output = [200.0, 4.0]
    areas_output = box_ops.compute_area(boxes)
    np.testing.assert_allclose(areas_output, exp_output)


def test_compute_intersection():
    boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]])
    exp_output = [[2.0, 0.0, 6.0], [1.0, 0.0, 5.0]]
    intersect_output = box_ops.compute_intersection(boxes1, boxes2)
    np.testing.assert_allclose(intersect_output, exp_output)


def test_compute_iou_2d_tensor():
    boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]])
    exp_output = [[2.0 / 16.0, 0, 6.0 / 400.0], [1.0 / 16.0, 0.0, 5.0 / 400.0]]
    iou_output = box_ops.compute_iou(boxes1, boxes2)
    np.testing.assert_allclose(iou_output, exp_output)


def test_compute_iou_3d_tensor():
    boxes1 = tf.constant([[[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]],
                          [[4.0, 3.0, 7.0, 5.0], [0, 0, 0, 0]]])
    boxes2 = tf.constant([[[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]],
                          [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]]])
    exp_output = [[[2.0 / 16.0, 0, 6.0 / 400.0], [1.0 / 16.0, 0.0, 5.0 / 400.0]],
                  [[2.0 / 16.0, 0, 6.0 / 400.0], [0, 0, 0]]]
    iou_output = box_ops.compute_iou(boxes1, boxes2)
    np.testing.assert_allclose(iou_output, exp_output)


def test_compute_giou_3d_tensor():
    boxes1 = tf.constant([[[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]],
                          [[4.0, 3.0, 7.0, 5.0], [0, 0, 0, 0]]])
    boxes2 = tf.constant([[[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]],
                          [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]]])
    exp_iou = np.array([[[2.0 / 16.0, 0, 6.0 / 400.0], [1.0 / 16.0, 0.0, 5.0 / 400.0]],
                        [[2.0 / 16.0, 0, 6.0 / 400.0], [0, 0, 0]]])
    exp_term2 = np.array([[[4. / 20, 125 / 132, 0.], [12 / 28., 84 / 90., 0.]],
                          [[4. / 20, 125. / 132, 0.], [36. / 48, 224. / 225, 0.]]])
    iou_output = box_ops.compute_giou(boxes1, boxes2, mode='giou')
    np.testing.assert_allclose(iou_output, exp_iou - exp_term2)


def test_compute_iou_works_on_empty_inputs():
    boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]])
    boxes_empty = tf.zeros((0, 4))
    iou_empty_1 = box_ops.compute_iou(boxes1, boxes_empty)
    iou_empty_2 = box_ops.compute_iou(boxes_empty, boxes2)
    iou_empty_3 = box_ops.compute_iou(boxes_empty, boxes_empty)
    np.testing.assert_array_equal(iou_empty_1.shape, (2, 0))
    np.testing.assert_array_equal(iou_empty_2.shape, (0, 3))
    np.testing.assert_array_equal(iou_empty_3.shape, (0, 0))


def test_normalize_boxes_coordinates():
    boxes = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    expected_boxes = np.array([[1, 3 / 10, 7 / 4, 5 / 10], [5 / 4, 6 / 10, 10 / 4, 7 / 10]],
                              np.float32)
    out_boxes = box_ops.normalize_box_coordinates(boxes, 4, 10)
    np.testing.assert_array_equal(out_boxes, expected_boxes)


def test_clip_boxes():
    boxes = tf.constant([[[-1, -1, 2, 1], [-1, 0, 2, 1]], [[-1, -1, 2, 1], [-1, 0, 2, 1]]],
                        tf.float32)
    image_shape = tf.constant([[1, 2], [2, 1]])
    expected_boxes = [[[0, 0, 1, 1], [0, 0, 1, 1]], [[0, 0, 2, 1], [0, 0, 2, 1]]]

    clipped_boxes = box_ops.clip_boxes(boxes, image_shape)
    np.testing.assert_array_equal(clipped_boxes, expected_boxes)


def test_flip_left_right():
    boxes = tf.constant([[0.4, 0.3, 0.6, 0.6], [0.5, 0.6, 0.9, 0.65]])
    expected_boxes = np.array([[0.4, 0.4, 0.6, 0.7], [0.5, 0.35, 0.9, 0.4]], np.float32)
    flipped_boxes = box_ops.flip_left_right(boxes)
    np.testing.assert_allclose(flipped_boxes, expected_boxes)
