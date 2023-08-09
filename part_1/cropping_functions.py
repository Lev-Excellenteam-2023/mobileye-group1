import pandas as pd
import numpy as np
import sklearn.cluster
import matplotlib.pyplot as plt


def find_center_and_radius(cropped_image):
    """
    Find the center and radius of a circle in an image.

    :param cropped_image: Input cropped image as a numpy array.
    :return: Tuple containing center and radius of the circle.
    """
    cropped_image_1d = cropped_image.reshape(-1, 1)

    num_clusters = 2
    kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters, random_state=0).fit(cropped_image_1d)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    plt.scatter(cropped_image_1d, labels, c=labels, cmap='rainbow')
    plt.scatter(cluster_centers, np.arange(num_clusters), marker='x', s=200, linewidths=3, color='black')
    plt.show()

    # Choose the cluster with the highest amount of points
    cluster = np.argmin(np.bincount(labels))

    # Find the center and radius of the circle
    # sum of all points in the cluster divided by the number of points in the cluster
    center = np.sum(cluster_centers[cluster]) / len(cluster_centers[cluster])


def big_crop(image: np.ndarray, value:tuple, color: str) -> np.ndarray:
    """
    Crop the input image based on a specific value
    that's known as a location of a traffic light.

    :param image: A NumPy array representing the input image.
    :param value: A tuple containing the (x, y) coordinates for cropping center.
    :param color: A string indicating the color channel to consider for cropping ('r' or 'g').

    :return: A cropped portion of the input image based on the provided conditions.
    :raises ValueError: If an invalid color value is provided (not 'r' or 'g').
    """
    max_y = image.shape[0]
    max_x = image.shape[1]
    max_x_value = min(value[1] + 30, max_x)
    min_x_value = max(value[1] - 30, 0)

    if color == 'r':
        max_y_value = min(value[0] + 80, max_y)
        min_y_value = max(value[0] - 30, 0)
    elif color == 'g':
        max_y_value = min(value[0] + 30, max_y)
        min_y_value = max(value[0] - 80, 0)
    else:
        raise Exception("Invalid color. Expected 'r' or 'g'.")
    return image[min_y_value:max_y_value, min_x_value:max_x_value]

