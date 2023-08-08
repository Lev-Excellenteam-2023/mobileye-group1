import os.path
import consts
from typing import List, Optional, Union, Dict, Tuple
import json
import argparse
from pathlib import Path
from scipy.ndimage import gaussian_filter

import numpy as np
from scipy import signal as sg
from scipy.ndimage import maximum_filter
from PIL import Image
import matplotlib.pyplot as plt

RED_X_COORDINATES = List[int]
RED_Y_COORDINATES = List[int]
GREEN_X_COORDINATES = List[int]
GREEN_Y_COORDINATES = List[int]
X_COORDINATES = List[int]
Y_COORDINATES = List[int]


def gaussian_kernel_2d(kernel_size, sigma):
    """
    Generate a 2D Gaussian kernel.

    :param kernel_size: Size of the kernel.
    :param sigma: Standard deviation of the Gaussian distribution.
    :return: 2D Gaussian kernel.
    """
    x, y = np.mgrid[-kernel_size // 2 + 1: kernel_size // 2 + 1, -kernel_size // 2 + 1: kernel_size // 2 + 1]
    g = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def gaussian_blur(image: np.ndarray) -> np.ndarray:
    """
    Apply Gaussian blur to an input image.

    :param image: Input image as a numpy array.
    :return: Blurred image as a numpy array.
    """
    blurred_image = sg.convolve(image, gaussian_kernel_2d(7, 1), mode='same', method='fft')

    return blurred_image


def find_red_coordinates(c_image: np.ndarray) -> Tuple[RED_X_COORDINATES, RED_Y_COORDINATES]:
    """
    Find coordinates of dominant red regions in an image.

    :param c_image: Input RGB image as a numpy array.
    :return: Tuple containing red x-coordinates and y-coordinates.
    """
    c_image = np.uint8(c_image * 255)

    r_image = c_image[:, :, 0]
    g_image = c_image[:, :, 1]
    b_image = c_image[:, :, 2]
    red_image = np.zeros((r_image.shape[0], r_image.shape[1]))

    for i in range(r_image.shape[0]):
        for j in range(r_image.shape[1]):
            if (0.8 * r_image[i][j] > g_image[i][j] > 0.4 * r_image[i][j] and 0.7 * r_image[i][j] > b_image[i][j] > 0.4 * \
                    r_image[i][j]):
                red_image[i][j] = r_image[i][j] - g_image[i][j] / 3 - b_image[i][j] / 4
                # check pixels around this point, if they are white, and not included in red_image, add them
                if 3 < i < r_image.shape[0] - 3 and 3 < j < r_image.shape[1] - 3:
                    for a in range(i - 2, i + 2):
                        for b in range(j - 2, j + 2):
                            if red_image[i][j] == 0 and r_image[a][b] > 0.8 and g_image[a][b] > 0.8 and b_image[a][b] > 0.8:
                                red_image[a][b] = (r_image[a][b] - g_image[a][b] / 3 - b_image[a][b] / 4) * 0.6


    blurred_image = gaussian_blur(red_image)


    return blurred_image / 255


def find_green_coordinates(c_image: np.ndarray) -> Tuple[GREEN_X_COORDINATES, GREEN_Y_COORDINATES]:
    """
    Find coordinates of dominant green regions in an image.

    :param c_image: Input RGB image as a numpy array.
    :return: Tuple containing green x-coordinates and y-coordinates.
    """
    c_image = np.uint8(c_image * 255)
    r_image = c_image[:, :, 0]
    g_image = c_image[:, :, 1]
    b_image = c_image[:, :, 2]
    green_image = 1.5 * g_image + b_image / 2 - 1.5 * r_image

    kernel = np.array([[1/25, 1/25, 1 / 25, 1/25, 1/25],
                       [1/25, 1/25, 1 / 25, 1/25, 1/25],
                       [1/25, 1/25, 1 / 25, 1/25, 1/25],
                       [1/25, 1/25, 1 / 25, 1/25, 1/25],
                       [1/25, 1/25, 1 / 25, 1/25, 1/25]])



    blurred_image = sg.convolve(green_image, kernel, mode='same', method='direct')

    #blurred_image = gaussian_blur(green_image)

    return blurred_image / 255

# deprecated
# def find_traffic_light_kernel() -> np.ndarray:
#     kernel = Image.open("green_light_2.png")
#     numpy_kernel = np.array(kernel)
#
#     return numpy_kernel


def max_suppression(image: np.ndarray, min_value: float, kernel_size: int = 150) -> np.ndarray:
    """
    Apply non-maximum suppression to an input image.

    :param image: Input image as a numpy array.
    :param min_value: Minimum value for suppression.
    :param kernel_size: Size of the maximum filter kernel.
    :return: Suppressed image as a numpy array.
    """
    max_image = maximum_filter(image, size=kernel_size, mode='constant')
    values = compare_max_supression(image, max_image, min_value)
    values = filter_points(values)
    return values


def compare_max_supression(image: np.ndarray, max_image: np.ndarray, min_value: float) -> Tuple[X_COORDINATES, Y_COORDINATES]:
    """
    Compare an image with a maximum image and perform suppression based on a threshold.

    :param image: Original input image as a numpy array.
    :param max_image: Image obtained after applying the maximum filter.
    :param min_value: Minimum value for suppression.
    :return: Tuple containing x-coordinates and y-coordinates of suppressed points.
    """
    x_coordinates = []
    y_coordinates = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x][y] == max_image[x][y] and max_image[x][y] > min_value:
                x_coordinates.append(x)
                y_coordinates.append(y)
    return x_coordinates, y_coordinates

def filter_points(values: Tuple[X_COORDINATES, Y_COORDINATES]) -> Tuple[X_COORDINATES, Y_COORDINATES]:
    values = ([value_0 for value_0, value_1 in zip(values[0], values[1]) if value_0 <= 410 and value_0 > 5],
              [value_1 for value_0, value_1 in zip(values[0], values[1]) if value_0 <= 410 and value_0 > 5])
    return values


def filter_green_points(c_image: np.ndarray, values: Tuple[X_COORDINATES, Y_COORDINATES]) -> Tuple[X_COORDINATES, Y_COORDINATES]:
    """
        Filter points based on conditions involving color channels.

        :param c_image: Input RGB image as a numpy array.
        :param values: Tuple of x-coordinates and y-coordinates.
        :return: Tuple of filtered x-coordinates and y-coordinates.
        """
    c_image = np.uint8(c_image * 255)
    r_image = c_image[:, :, 0]
    g_image = c_image[:, :, 1]
    b_image = c_image[:, :, 2]
    new_values = ([], [])
    for i in range(len(values[0])):
        x = values[0][i]
        y = values[1][i]

        if r_image[x][y] < 0.4 * g_image[x][y]:
            new_values[0].append(x)
            new_values[1].append(y)

    return new_values