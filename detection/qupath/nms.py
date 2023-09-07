import torch
import numpy as np


def nms(bounding_boxes, confidence_score, threshold_iou=0.5, threshold_iom=0.5, return_idxs=False):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        if return_idxs:
            return None
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)
    score = score * score * areas

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)
    idxs = []

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        idxs.append(index)

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        union = (areas[index] + areas[order[:-1]] - intersection)
        ratio_iou = intersection / union
        minimum = [min(areas[index], area) for area in areas[order[:-1]]]
        ratio_iom = intersection / minimum

        left = np.where(np.logical_and(ratio_iou < threshold_iou, ratio_iom < threshold_iom))
        order = order[left]

    if return_idxs:
        return idxs
    return picked_boxes, picked_score
