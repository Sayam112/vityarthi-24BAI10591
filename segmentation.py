import cv2
import numpy as np


def threshold_segmentation(image, threshold=127):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, result = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    return result


def kmeans_segmentation(image, clusters=3):
    if clusters < 2:
        raise ValueError("Clusters must be at least 2.")

    pixels = image.reshape((-1, 3)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        clusters,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(centers)
    result = centers[labels.flatten()]
    result = result.reshape(image.shape)

    return result
