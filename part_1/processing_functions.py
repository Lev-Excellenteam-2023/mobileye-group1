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
    #blurred_image = gaussian_blur(c_image)
    kernel = find_traffic_light_kernel()
    convoluted_image = sg.convolve(c_image, kernel, mode='same', method='fft')
    image_uint8 = np.uint8(convoluted_image)
    plt.imshow(image_uint8, cmap='hot')
    plt.show()

    return convoluted_image


def gaussian_kernel_3d(kernel_size, sigma):
    x, y, z = np.mgrid[-kernel_size // 2 + 1: kernel_size // 2 + 1, -kernel_size // 2 + 1: kernel_size // 2 + 1,
              -kernel_size // 2 + 1: kernel_size // 2 + 1]
    g = np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def gaussian_blur(image: np.ndarray) -> np.ndarray:
    blurred_image = sg.convolve(image, gaussian_kernel_3d(21, 1), mode='same')
    return blurred_image


def find_red_coordinates(image: np.ndarray) -> Tuple[RED_X_COORDINATES, RED_Y_COORDINATES]:
    pass


def find_green_coordinates(image: np.ndarray) -> Tuple[GREEN_X_COORDINATES, GREEN_Y_COORDINATES]:
    pass


def find_traffic_light_kernel() -> np.ndarray:
    kernel = Image.open("green_light_2.png")
    numpy_kernel = np.array(kernel)
    normalized_kernel = numpy_kernel / np.sum(numpy_kernel)

    return normalized_kernel
