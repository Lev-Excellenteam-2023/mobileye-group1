import numpy as np


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

