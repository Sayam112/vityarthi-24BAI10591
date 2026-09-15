import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def image_pixel_samples(image, sample_size=3000):
    pixels = image.reshape((-1, 3)).astype(np.float32)

    if len(pixels) > sample_size:
        indices = np.linspace(
            0,
            len(pixels) - 1,
            sample_size,
            dtype=int
        )
        pixels = pixels[indices]

    return pixels


def kmeans_pattern_analysis(image, clusters=3):
    samples = image_pixel_samples(image)

    model = KMeans(
        n_clusters=clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(samples)

    return model.cluster_centers_, labels


def pca_analysis(image, components=2):
    samples = image_pixel_samples(image)

    components = min(
        components,
        samples.shape[0],
        samples.shape[1]
    )

    model = PCA(n_components=components)
    transformed = model.fit_transform(samples)

    return transformed, model.explained_variance_ratio_


def save_pca_csv(path, transformed):
    header = ",".join(
        [f"PC{i + 1}" for i in range(transformed.shape[1])]
    )

    np.savetxt(
        path,
        transformed,
        delimiter=",",
        header=header,
        comments=""
    )
