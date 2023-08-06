import os.path
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


def preprocess_image(c_image: np.ndarray) -> np.ndarray:
    # TODO: add preprocessing
    blurred_image = sg.convolve(c_image, gaussian_kernel_3d(21, 1), mode='same')
    blurred_image_new = blurred_image.astype("uint32")
    plt.imshow(blurred_image_new)
    plt.show()

    #kernel = Image.open("green_light_1.png")
    kernel = [[[-4/9, 3/9, 1/9]],
               [[-6/9, 4/9, 2/9]],
               [[-4/9, 3/9, 1/9]]]
    numpy_kernel = np.array(kernel)


    normalized_kernel = numpy_kernel / np.sum(numpy_kernel)
    new_image = sg.convolve(blurred_image, numpy_kernel, mode='same')

    uint8_image = new_image.astype("uint32")
    plt.imshow(uint8_image, cmap='hot')
    plt.show()

    green_channel = uint8_image[:, :, 1]  # Green channel is at index 1 (0-based index)
    thresholded_green = green_channel > 100

    return uint8_image


def gaussian_kernel_3d(kernel_size, sigma):
    x, y, z = np.mgrid[-kernel_size // 2 + 1: kernel_size // 2 + 1, -kernel_size // 2 + 1: kernel_size // 2 + 1,
              -kernel_size // 2 + 1: kernel_size // 2 + 1]
    g = np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def find_red_coordinates(image: np.ndarray) -> Tuple[RED_X_COORDINATES, RED_Y_COORDINATES]:
    pass


def find_green_coordinates(image: np.ndarray) -> Tuple[GREEN_X_COORDINATES, GREEN_Y_COORDINATES]:
    pass


def find_traffic_light_kernel(image: np.ndarray) -> np.ndarray:
    new_image = image[72:148]
    plt.imshow(new_image)
    plt.show()
