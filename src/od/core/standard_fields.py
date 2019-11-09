from enum import Enum


class BoxField():
    BOXES = 'boxes'
    KEYPOINTS = 'keypoints'
    LABELS = 'labels'
    MASKS = 'masks'
    NUM_BOXES = 'num_boxes'
    SCORES = 'scores'
    WEIGHTS = 'weights'

class LossField():
    CLASSIFICATION = 'classification'
    INSTANCE_SEGMENTATION = 'instance_segmentation'
    LOCALIZATION = 'localization'
