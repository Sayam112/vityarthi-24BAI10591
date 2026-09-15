import cv2
import numpy as np
from skimage.feature import hog


def harris_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    response = cv2.cornerHarris(gray, 2, 3, 0.04)

    result = image.copy()
    threshold = 0.01 * response.max()
    result[response > threshold] = [0, 0, 255]

    return result, int(np.sum(response > threshold))


def sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    result = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    descriptor_count = 0 if descriptors is None else len(descriptors)

    return result, len(keypoints), descriptor_count


def hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features, visualization = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True
    )

    visualization = np.clip(
        visualization * 255,
        0,
        255
    ).astype(np.uint8)

    return visualization, features


def feature_summary(image):
    result = {}

    _, harris_count = harris_corners(image)
    _, sift_count, descriptor_count = sift_features(image)
    _, hog_vector = hog_features(image)

    result["Harris corners"] = harris_count
    result["SIFT keypoints"] = sift_count
    result["SIFT descriptors"] = descriptor_count
    result["HOG feature length"] = len(hog_vector)

    return result
