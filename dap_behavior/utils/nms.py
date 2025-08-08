import numpy as np

def nms_cpu(boxes, scores, iou_threshold):
    """
    Perform non-maximum suppression (NMS) on the bounding boxes using CPU.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes with shape (N, 4).
        scores (numpy.ndarray): Array of scores with shape (N,).
        iou_threshold (float): Intersection over Union (IoU) threshold for NMS.

    Returns:
        numpy.ndarray: Indices of the bounding boxes to keep.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    # Compute the area of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort the bounding boxes by their scores in descending order
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Compute IoU of the kept box with the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU less than the threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)