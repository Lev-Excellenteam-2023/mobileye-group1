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
    kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters, random_state=0).fit(cropped_image_1d.reshape(-1, 1))
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    plt.scatter(cropped_image_1d, labels, c=labels, cmap='rainbow')
    plt.scatter(cluster_centers, np.arange(num_clusters), marker='x', s=200, linewidths=3, color='black')
    plt.show()



