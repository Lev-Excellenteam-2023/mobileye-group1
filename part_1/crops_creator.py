from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from consts import CROP_DIR, CROP_RESULT, SEQ, IS_TRUE, IGNOR, CROP_PATH, X0, X1, Y0, Y1, COLOR, SEQ_IMAG, COL, X, Y, \
    GTIM_PATH, IMAG_PATH, JSON_PATH

from pandas import DataFrame

import cropping_functions


def make_crop(x_values, y_values, image_paths, colors):
    """
    The function that creates the crops from the image.
    Your return values from here should be the coordinates of the crops in this format (x0, x1, y0, y1, crop content):
    'x0'  The bigger x value (the right corner)
    'x1'  The smaller x value (the left corner)
    'y0'  The smaller y value (the lower corner)
    'y1'  The bigger y value (the higher corner)
    """
    image: np.ndarray = np.array(Image.open(image_paths), dtype=np.float32)
    light_image, min_x, min_y = cropping_functions.big_crop(image, tuple((x_values, y_values)), colors)
    center, radius = cropping_functions.find_center_and_radius(light_image)
    center[0] = center[0] + min_x
    center[1] = center[1] + min_y
    left_x, right_x, low_y, top_y = cropping_functions.calculate_traffic_light_coordinates(center, radius,
                                                                                               colors)
    final_crop = cropping_functions.final_image_crop(image, left_x, right_x, top_y, low_y)

    return left_x, right_x, low_y, top_y, final_crop


def check_crop(crops, x0, x1, y0, y1, json_path):
    """
    Here you check if your crop contains a traffic light or not.
    Try using the ground truth to do that (Hint: easier than you think for the simple cases, and if you found a hard
    one, just ignore it for now :). )
    """
    polygons = cropping_functions.find_json_polygons(json_path)

    return True, True


def create_crops(df: DataFrame) -> DataFrame:
    # Your goal in this part is to take the coordinates you have in the df, run on it, create crops from them, save them
    # in the 'data' folder, then check if crops you have found is correct (meaning the TFL is fully contained in the
    # crops) by comparing it to the ground truth and in the end right all the result data you have in the following
    # DataFrame (for doc about each field and its input, look at 'CROP_RESULT')
    #
    # *** IMPORTANT ***
    # All crops should be the same size or smaller!!!

    # creates a folder for you to save the crops in, recommended not must
    if not CROP_DIR.exists():
        CROP_DIR.mkdir()

    # For documentation about each key end what it means, click on 'CROP_RESULT' and see for each value what it means.
    # You wanna stick with this DataFrame structure because its output is the same as the input for the next stages.
    result_df = DataFrame(columns=CROP_RESULT)

    # A dict containing the row you want to insert into the result DataFrame.
    result_template: Dict[Any] = {SEQ: '', IS_TRUE: '', IGNOR: '', CROP_PATH: '', X0: '', X1: '', Y0: '', Y1: '',
                                  COL: ''}
    for index, row in df.iterrows():
        result_template[SEQ] = row[SEQ_IMAG]
        result_template[COL] = row[COLOR]

        # example code:
        x0, x1, y0, y1, crop = make_crop(df[X][index], df[Y][index], df[IMAG_PATH][index], df[COLOR][index])
        result_template[X0], result_template[X1], result_template[Y0], result_template[Y1] = x0, x1, y0, y1
        crop_path: str = '/data/crops/my_crop_unique_name.probably_containing_the original_image_name+somthing_unique'
        # crops.save(CROP_DIR / crop_path)
        result_template[CROP_PATH] = crop_path
        plt.imshow(crop)
        plt.show()
        result_template[IS_TRUE], result_template[IGNOR] = check_crop(crop, x0, x1, y0, y1, df[JSON_PATH][index])

        # added to current row to the result DataFrame that will serve you as the input to part 2 B).
        result_df = result_df._append(result_template, ignore_index=True)
    return result_df
