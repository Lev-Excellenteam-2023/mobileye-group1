import os
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
    image: np.ndarray = np.array(Image.open(image_paths), dtype=np.float64)
    light_image, min_x, min_y = cropping_functions.big_crop(image, tuple((x_values, y_values)), colors)
    center, radius = cropping_functions.find_center_and_radius(light_image)
    center[0] = center[0] + min_x
    center[1] = center[1] + min_y
    left_x, right_x, up_y, down_y = cropping_functions.calculate_traffic_light_coordinates(center, radius, colors)
    final_crop = cropping_functions.final_image_crop(image, left_x, right_x, up_y, down_y)

    return left_x, right_x, up_y, down_y, final_crop


def check_crop_helper(min_x, max_x, min_y, max_y, x0, x1, y0, y1):
    o1 = x0 <= min_x and x1 >= max_x and y0 <= min_y and y1 >= max_y
    o2 = x0 >= min_x and x1 <= max_x and y0 >= min_y and y1 <= max_y
    o3 = min_x <= x0 <= max_x and min_y <= y0 <= max_y
    o4 = min_x <= x1 <= max_x and min_y <= y0 <= max_y
    o5 = min_x <= x1 <= max_x and min_y <= y1 <= max_y
    o6 = min_x <= x0 <= max_x and min_y <= y1 <= max_y
    o7 = x0 <= min_x <= x1 and y0 <= min_y <= y1
    o8 = x0 <= max_x <= x1 and y0 <= min_y <= y1
    o9 = x0 <= max_x <= x1 and y0 <= max_y <= y1
    o10 = x0 <= min_x <= x1 and y0 <= max_y <= y1
    o11 = min_x <= x0 and x1 <= max_x and y0 <= min_y and max_y <= y1
    o12 = min_y <= y0 and y1 <= max_y and x0 <= min_x and max_x <= x1
    return o1 or o2 or o3 or o4 or o5 or o6 or o7 or o8 or o9 or o10 or o11 or o12


def check_crop(x0, x1, y0, y1, json_path):
    """
    Here you check if your crop contains a traffic light or not.
    Try using the ground truth to do that (Hint: easier than you think for the simple cases, and if you found a hard
    one, just ignore it for now :). )
    """
    polygons = cropping_functions.find_json_polygons(json_path)
    flag = False
    for polygon in polygons:
        min_x, max_x, min_y, max_y = cropping_functions.squared_polygon_coordinates(polygon)
        # if x0 <= min_x and x1 >= max_x and y0 <= min_y and y1 >= max_y:
        #     flag =True

        if check_crop_helper(min_x, max_x, min_y, max_y, x0, x1, y0, y1):
            flag =True

    return flag, False


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
    zooms = []
    for index, row in df.iterrows():
        result_template[SEQ] = row[SEQ_IMAG]
        result_template[COL] = row[COLOR]



        # example code:
        x0, x1, y0, y1, crop = make_crop(df[X][index], df[Y][index], df[IMAG_PATH][index], df[COLOR][index])
        result_template[X0], result_template[X1], result_template[Y0], result_template[Y1] = x0, x1, y0, y1
        image_name = df[IMAG_PATH][index].split('/')
        crop_path: str = '../data/crops/' + str(index) + '_' + image_name[len(image_name) - 1]
        result_template[CROP_PATH] = crop_path
        # plt.imshow(crop)
        # plt.show()

        result_template[IS_TRUE], result_template[IGNOR] = check_crop(x0, x1, y0, y1, df[JSON_PATH][index])
        zooms.append(20 / (x1 - x0)) if (x1 - x0 ) * 3 == y1 - y0 and x1 - x0 > 0 else zooms.append(0)

        # added to current row to the result DataFrame that will serve you as the input to part 2 B).
        result_df = result_df._append(result_template, ignore_index=True)


        # new_crop = np.resize(crop, (60, 20, 3))
        os.makedirs(os.path.dirname(crop_path), exist_ok=True)
        image_crop = Image.fromarray(crop)
        resized_image = image_crop.resize((20, 60))
        resized_image.save(crop_path)

    return result_df, zooms
