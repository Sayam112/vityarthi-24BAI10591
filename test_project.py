import os
import tempfile

import cv2
import numpy as np

from image_processing import (
    grayscale,
    gaussian_filter,
    median_filter,
    histogram_equalization,
    fourier_transform,
    canny_edges
)

from feature_extraction import (
    harris_corners,
    sift_features,
    hog_features
)

from segmentation import (
    threshold_segmentation,
    kmeans_segmentation
)

from pattern_analysis import (
    kmeans_pattern_analysis,
    pca_analysis,
    save_pca_csv
)

from motion_analysis import (
    frame_difference,
    optical_flow
)


def create_test_image():
    image = np.zeros(
        (120, 160, 3),
        dtype=np.uint8
    )

    cv2.rectangle(
        image,
        (20, 20),
        (80, 80),
        (255, 255, 255),
        -1
    )

    cv2.circle(
        image,
        (120, 60),
        25,
        (150, 150, 150),
        -1
    )

    return image


def test_image_processing():
    image = create_test_image()

    assert grayscale(image).shape == (120, 160)
    assert gaussian_filter(image).shape == image.shape
    assert median_filter(image).shape == image.shape
    assert histogram_equalization(image).shape == (120, 160)
    assert fourier_transform(image).shape == (120, 160)
    assert canny_edges(image).shape == (120, 160)


def test_features():
    image = create_test_image()

    harris, _ = harris_corners(image)
    sift, keypoints, descriptors = sift_features(image)
    hog_image, features = hog_features(image)

    assert harris.shape == image.shape
    assert sift.shape == image.shape
    assert keypoints >= 0
    assert descriptors >= 0
    assert hog_image.ndim == 2
    assert len(features) > 0


def test_segmentation():
    image = create_test_image()

    threshold = threshold_segmentation(image)
    segmented = kmeans_segmentation(image, 2)

    assert threshold.shape == (120, 160)
    assert segmented.shape == image.shape


def test_pattern_analysis():
    image = create_test_image()

    centers, labels = kmeans_pattern_analysis(image, 2)
    transformed, variance = pca_analysis(image)

    assert centers.shape == (2, 3)
    assert len(labels) > 0
    assert transformed.shape[1] == 2
    assert len(variance) == 2


def test_pca_file():
    image = create_test_image()
    transformed, _ = pca_analysis(image)

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "pca.csv")
        save_pca_csv(path, transformed)
        assert os.path.isfile(path)


def test_motion():
    image1 = create_test_image()
    image2 = image1.copy()

    cv2.circle(
        image2,
        (50, 100),
        10,
        (255, 255, 255),
        -1
    )

    difference = frame_difference(
        image1,
        image2
    )

    flow = optical_flow(
        image1,
        image2
    )

    assert difference.shape == (120, 160)
    assert flow.shape == image1.shape


if __name__ == "__main__":
    test_image_processing()
    test_features()
    test_segmentation()
    test_pattern_analysis()
    test_pca_file()
    test_motion()

    print("All tests passed successfully.")
