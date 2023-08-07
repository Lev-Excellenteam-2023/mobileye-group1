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


def preprocess_image(c_image: np.ndarray) -> np.ndarray:
   pass


def gaussian_kernel_3d(kernel_size, sigma):
    x, y, z = np.mgrid[-kernel_size // 2 + 1: kernel_size // 2 + 1, -kernel_size // 2 + 1: kernel_size // 2 + 1,
              -kernel_size // 2 + 1: kernel_size // 2 + 1]
    g = np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()

def gaussian_kernel_2d(kernel_size, sigma):
    x, y = np.mgrid[-kernel_size // 2 + 1: kernel_size // 2 + 1, -kernel_size // 2 + 1: kernel_size // 2 + 1]
    g = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def gaussian_blur(image: np.ndarray) -> np.ndarray:
    blurred_image = sg.convolve(image, gaussian_kernel_2d(7, 1), mode='same', method='fft')

    return blurred_image



def find_red_coordinates(c_image: np.ndarray) -> Tuple[RED_X_COORDINATES, RED_Y_COORDINATES]:
    c_image = np.uint8(c_image * 255)
    r_image = c_image[:, :, 0]
    g_image = c_image[:, :, 1]
    b_image = c_image[:, :, 2]
    red_image = r_image - b_image / 3 - g_image / 3

    blurred_image = gaussian_blur(red_image)

    return blurred_image / 255
def find_green_coordinates(c_image: np.ndarray) -> np.ndarray:

    r_image = c_image[:, :, 0]
    g_image = c_image[:, :, 1]
    b_image = c_image[:, :, 2]


def find_green_coordinates(c_image: np.ndarray) -> Tuple[GREEN_X_COORDINATES, GREEN_Y_COORDINATES]:
    c_image = np.uint8(c_image * 255)
    r_image = c_image[:, :, 0]
    g_image = c_image[:, :, 1]
    b_image = c_image[:, :, 2]
    green_image = g_image + b_image / 2 - r_image

    blurred_image = gaussian_blur(green_image)

    return blurred_image / 255


def find_traffic_light_kernel() -> np.ndarray:
    kernel = Image.open("green_light_2.png")
    numpy_kernel = np.array(kernel)

    return numpy_kernel

def max_suppression(image: np.ndarray, min_value: float, kernel_size: int = 50) -> np.ndarray:
    max_image = maximum_filter(image, size=kernel_size, mode='constant')
    values = compare_max_supression(image, max_image, min_value)
    return values

def compare_max_supression(image: np.ndarray, max_image: np.ndarray, min_value: float) -> Tuple[RED_X_COORDINATES, RED_Y_COORDINATES, GREEN_X_COORDINATES, GREEN_Y_COORDINATES]:

    # return pixels that have same value in both images
    x_coordinates = []
    y_coordinates = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x][y] == max_image[x][y] and max_image[x][y] > min_value:
                x_coordinates.append(x)
                y_coordinates.append(y)
    # see dots on image
    #plt.imshow(image)
    #plt.scatter(green_y_coordinates, green_x_coordinates, s=20, color = 'green', marker='x')
    #plt.show()
    #plt.clf()
    return x_coordinates, y_coordinates


