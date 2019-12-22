from od.model.faster_rcnn import build_fpn_resnet50_faster_rcnn

def test_build_fpn_resnet50_faster_rcnn():
    num_classes = 20
    batch_size = 3
    model = build_fpn_resnet50_faster_rcnn(num_classes, batch_size)
