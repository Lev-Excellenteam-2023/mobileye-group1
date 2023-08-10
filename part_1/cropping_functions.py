from typing import Any, Dict, List

import pandas as pd
import numpy as np
import processing_functions
import sklearn.cluster
import cv2
import matplotlib.pyplot as plt
import hashlib
import json
from pathlib import Path


def find_center_and_radius(cropped_image):
    """
    Find the center and radius of a circle in an image.

    :param cropped_image: Input cropped image as a numpy array.
    :return: Tuple containing center and radius of the circle.
    """

    contoured_image = np.uint8(cropped_image * 255)
    copied_image = contoured_image.copy()
    contours, hierarchy = cv2.findContours(contoured_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # plt.imshow(copied_image)
    # plt.show()

    contoured_canvas = contoured_image.copy()
    cv2.drawContours(contoured_canvas, contours, -1, (255, 255, 255), 1)
    # Draw all contours in white
    # plt.imshow(contoured_canvas)
    # plt.show()

    # find the most circular contour
    circularity = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if 1400 > area > 5:
            circularity.append(4 * np.pi * area / (perimeter ** 2))

    circularity = np.array(circularity)
    hashed_image = hashlib.sha256(cropped_image).hexdigest()

    if len(circularity) == 0:
        # return center of image and radius 20
        # save contour image
        center = [cropped_image.shape[0] // 2, cropped_image.shape[1] // 2]
        radius = 5
        cv2.circle(copied_image, tuple(center), radius, (255, 255, 255), 1)
        # save cropped image
        cv2.imwrite(f'./Test Results/{hashed_image}.png', copied_image)
        return [cropped_image.shape[0] // 2, cropped_image.shape[1] // 2], radius
    circular_contour = contours[np.argmax(circularity)]

    # Draw the most circular contour in white
    contoured_canvas = np.zeros_like(contoured_image, dtype=np.uint8)
    cv2.drawContours(contoured_canvas, [circular_contour], -1, (255, 255, 255), 1)
    # plt.imshow(contoured_canvas)
    # plt.show()

    # Find the center and radius of the circle
    (x, y), radius = cv2.minEnclosingCircle(circular_contour)

    if radius > 21:
        center = [cropped_image.shape[0] // 2, cropped_image.shape[1] // 2]
        radius = 5
        cv2.circle(copied_image, tuple(center), radius, (255, 255, 255), 1)
        # save cropped image
        cv2.imwrite(f'./Test Results/{hashed_image}.png', copied_image)
        return [cropped_image.shape[0] // 2, cropped_image.shape[1] // 2], radius

    # the following code is for visualization purposes only
    # center = (int(x), int(y))
    # radius = int(radius)
    # new_image = np.zeros_like(contoured_image, dtype=np.uint8)
    # cv2.circle(new_image, center, radius, (255, 0, 0), 1)
    # plt.imshow(new_image, cmap='gray')
    # plt.title("Circle Center and Radius")
    # plt.show()

    # draw the circle on the original image
    center = [int(x), int(y)]
    cv2.circle(copied_image, tuple(center), int(radius), (255, 0, 0), 1)
    cv2.imwrite(f'./Test Results/{hashed_image}.png', copied_image)
    return center, int(radius)


def big_crop(image: np.ndarray, value: tuple, color: str) -> np.ndarray:
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
    max_x_value = int(min(value[1] + 30, max_x - 1))
    min_x_value = int(max(value[1] - 30, 0))
    max_y_value = int(min(value[0] + 30, max_y - 1))
    min_y_value = int(max(value[0] - 30, 0))

    image = image[min_y_value:max_y_value, min_x_value:max_x_value, :]

    return convert_to_1_chanel(image, color), min_x_value, min_y_value


def convert_to_1_chanel(image: np.ndarray, color: str) -> np.ndarray:
    """
    Convert the input image to a single channel based on the specified color.

    :param image: A NumPy array representing the input image.
    :param color: A string indicating the color channel to extract ('r' or 'g').

    :return: A single-channel image containing the extracted color information.
    """
    if color == 'r':
        return processing_functions.find_red_coordinates(image)
    else:
        return processing_functions.find_green_coordinates(image)


def calculate_traffic_light_coordinates(center, radius, color):
    """
    Calculate the cropping coordinates for a traffic light based on its center, radius, and color.

    :param center: A tuple representing the (x, y) coordinates of the traffic light's center.
    :param radius: The radius of the traffic light.
    :param color: The color of the traffic light ('r' for red or 'g' for green).

    :return: A tuple containing the left, right, top, and low coordinates for cropping.

    """
    right_x = center[0] + 2 * radius
    left_x = center[0] - 2 * radius
    if color == 'g':
        # For green, we need to go up to include all the traffic light
        up_y = center[1] - 2 * radius - (8 * radius)
        down_y = center[1] + 2 * radius
    elif color == 'r':
        # For red, we need to go down to include all the traffic light
        up_y = center[1] - 2 * radius
        down_y = center[1] + 2 * radius + (8 * radius)
    else:
        raise ValueError("Invalid color. Please provide 'green' or 'red'.")

    return left_x, right_x, up_y, down_y


def final_image_crop(image: np.ndarray, left_x: int, right_x: int, up_y: int, down_y: int) -> np.ndarray:
    """
     Crop the input image based on the specified cropping coordinates.

    :param image: A NumPy array representing the input image.
    :param left_x: The leftmost column index for cropping.
    :param right_x: The rightmost column index for cropping.
    :param up_y: The top row index for cropping.
    :param down_y: The bottom row index for cropping.

    :return: A cropped portion of the input image.
    """
    max_y = image.shape[0]
    max_x = image.shape[1]
    max_x_value = min(right_x, max_x)
    min_x_value = max(left_x, 0)
    max_y_value = min(down_y, max_y)
    min_y_value = max(up_y, 0)
    image = image[int(min_y_value):int(max_y_value), int(min_x_value):int(max_x_value), :]
    image = np.uint8(image)
    return image


def find_json_polygons(json_path) -> List[List]:
    """

    Extracts polygons labeled as 'traffic light' from a JSON file.

    :param json_path: Path to the JSON file.
    :return: List of polygons (lists of coordinates).
    """
    gt_data: Dict[str, Any] = json.loads(Path(json_path).read_text())
    what: List[str] = ['traffic light']
    objects = [o for o in gt_data['objects'] if o['label'] in what]
    polygons = []
    for object in objects:
        polygons.append(object['polygon'])
    return polygons


def squared_polygon_coordinates(polygon):
    xs = [coordinate[0] for coordinate in polygon]
    ys = [coordinate[1] for coordinate in polygon]
    return [min(xs), max(xs), min(ys), max(ys)]
