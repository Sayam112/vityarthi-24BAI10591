import argparse
import os

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
    hog_features,
    feature_summary
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


OUTPUT_DIR = "output"


def ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_image(name, image):
    ensure_output()
    path = os.path.join(OUTPUT_DIR, name)

    if not cv2.imwrite(path, image):
        raise IOError(f"Could not save {path}")

    print(f"Saved: {path}")


def create_sample_image():
    image = np.zeros((400, 600, 3), dtype=np.uint8)

    cv2.rectangle(
        image,
        (60, 80),
        (260, 280),
        (255, 255, 255),
        -1
    )

    cv2.circle(
        image,
        (430, 180),
        90,
        (120, 120, 120),
        -1
    )

    cv2.line(
        image,
        (30, 350),
        (570, 350),
        (200, 200, 200),
        5
    )

    return image


def load_image(path):
    if path is None:
        print("No image supplied. Creating a synthetic sample image.")
        return create_sample_image()

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(path)

    if image is None:
        raise ValueError("The selected file is not a readable image.")

    return image


def enhancement_menu(image):
    while True:
        print("\n--- Image Enhancement ---")
        print("1. Grayscale")
        print("2. Gaussian Filter")
        print("3. Median Filter")
        print("4. Histogram Equalization")
        print("5. Fourier Transform")
        print("6. Canny Edge Detection")
        print("7. Back")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            save_image("grayscale.png", grayscale(image))
        elif choice == "2":
            save_image("gaussian.png", gaussian_filter(image))
        elif choice == "3":
            save_image("median.png", median_filter(image))
        elif choice == "4":
            save_image("histogram_equalized.png", histogram_equalization(image))
        elif choice == "5":
            save_image("fourier_spectrum.png", fourier_transform(image))
        elif choice == "6":
            save_image("canny_edges.png", canny_edges(image))
        elif choice == "7":
            return
        else:
            print("Invalid choice.")


def feature_menu(image):
    while True:
        print("\n--- Feature Extraction ---")
        print("1. Harris Corners")
        print("2. SIFT")
        print("3. HOG")
        print("4. Feature Summary")
        print("5. Back")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            result, count = harris_corners(image)
            save_image("harris_corners.png", result)
            print(f"Harris corners detected: {count}")

        elif choice == "2":
            result, keypoints, descriptors = sift_features(image)
            save_image("sift_features.png", result)
            print(f"SIFT keypoints: {keypoints}")
            print(f"SIFT descriptors: {descriptors}")

        elif choice == "3":
            result, features = hog_features(image)
            save_image("hog_visualization.png", result)
            print(f"HOG feature length: {len(features)}")

        elif choice == "4":
            summary = feature_summary(image)
            for key, value in summary.items():
                print(f"{key}: {value}")

        elif choice == "5":
            return

        else:
            print("Invalid choice.")


def segmentation_menu(image):
    while True:
        print("\n--- Image Segmentation ---")
        print("1. Threshold Segmentation")
        print("2. K-Means Segmentation")
        print("3. Back")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            result = threshold_segmentation(image)
            save_image("threshold_segmentation.png", result)

        elif choice == "2":
            raw = input("Number of clusters [2-6, default 3]: ").strip()
            clusters = int(raw) if raw else 3

            if not 2 <= clusters <= 6:
                print("Clusters must be between 2 and 6.")
                continue

            result = kmeans_segmentation(image, clusters)
            save_image("kmeans_segmentation.png", result)

        elif choice == "3":
            return

        else:
            print("Invalid choice.")


def pattern_menu(image):
    while True:
        print("\n--- Pattern Analysis ---")
        print("1. K-Means Pattern Clustering")
        print("2. PCA")
        print("3. Back")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            raw = input("Number of clusters [2-6, default 3]: ").strip()
            clusters = int(raw) if raw else 3

            centers, labels = kmeans_pattern_analysis(
                image,
                clusters
            )

            print("\nCluster centers:")
            print(np.round(centers, 2))
            print(f"Number of labelled samples: {len(labels)}")

        elif choice == "2":
            transformed, variance = pca_analysis(image)

            path = os.path.join(
                OUTPUT_DIR,
                "pca_result.csv"
            )

            save_pca_csv(path, transformed)

            print(f"Saved: {path}")
            print(
                "Explained variance ratio:",
                np.round(variance, 4)
            )

        elif choice == "3":
            return

        else:
            print("Invalid choice.")


def motion_menu(image):
    print("\n--- Motion Analysis ---")
    print("This module needs two frames.")
    print("Press Enter to create a synthetic second frame.")
    second_path = input(
        "Enter second frame path (or press Enter): "
    ).strip()

    if second_path:
        second = load_image(second_path)

        if second.shape != image.shape:
            second = cv2.resize(
                second,
                (image.shape[1], image.shape[0])
            )
    else:
        second = image.copy()

        cv2.circle(
            second,
            (450, 200),
            60,
            (255, 255, 255),
            -1
        )

    save_image(
        "frame_difference.png",
        frame_difference(image, second)
    )

    save_image(
        "optical_flow.png",
        optical_flow(image, second)
    )


def run_all(image):
    print("\nRunning representative operations...")

    save_image("grayscale.png", grayscale(image))
    save_image("gaussian.png", gaussian_filter(image))
    save_image("median.png", median_filter(image))
    save_image(
        "histogram_equalized.png",
        histogram_equalization(image)
    )
    save_image(
        "fourier_spectrum.png",
        fourier_transform(image)
    )
    save_image("canny_edges.png", canny_edges(image))

    harris, _ = harris_corners(image)
    save_image("harris_corners.png", harris)

    sift, _, _ = sift_features(image)
    save_image("sift_features.png", sift)

    hog_image, _ = hog_features(image)
    save_image("hog_visualization.png", hog_image)

    threshold = threshold_segmentation(image)
    save_image("threshold_segmentation.png", threshold)

    segmented = kmeans_segmentation(image, 3)
    save_image("kmeans_segmentation.png", segmented)

    transformed, variance = pca_analysis(image)
    save_pca_csv(
        os.path.join(OUTPUT_DIR, "pca_result.csv"),
        transformed
    )

    print(
        "PCA explained variance:",
        np.round(variance, 4)
    )

    second = image.copy()
    cv2.circle(
        second,
        (450, 200),
        60,
        (255, 255, 255),
        -1
    )

    save_image(
        "frame_difference.png",
        frame_difference(image, second)
    )

    save_image(
        "optical_flow.png",
        optical_flow(image, second)
    )

    print("Representative operations completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Computer Vision Image Analysis Toolkit"
    )

    parser.add_argument(
        "--image",
        help="Path to input image"
    )

    args = parser.parse_args()

    try:
        ensure_output()
        image = load_image(args.image)

        print("\n========================================")
        print(" COMPUTER VISION IMAGE ANALYSIS TOOLKIT")
        print("========================================")

        while True:
            print("\nMain Menu")
            print("1. Image Enhancement")
            print("2. Feature Extraction")
            print("3. Image Segmentation")
            print("4. Pattern Analysis")
            print("5. Motion Analysis")
            print("6. Run All Basic Operations")
            print("7. Exit")

            choice = input("Enter choice: ").strip()

            if choice == "1":
                enhancement_menu(image)
            elif choice == "2":
                feature_menu(image)
            elif choice == "3":
                segmentation_menu(image)
            elif choice == "4":
                pattern_menu(image)
            elif choice == "5":
                motion_menu(image)
            elif choice == "6":
                run_all(image)
            elif choice == "7":
                print("Project execution completed.")
                break
            else:
                print("Invalid choice. Please select 1-7.")

    except (FileNotFoundError, ValueError, IOError) as error:
        print(f"Error: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
