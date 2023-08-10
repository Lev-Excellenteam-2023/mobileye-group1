import csv
import os

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image


def check_column_b(csv_filename):
    current_directory = os.getcwd()

    with open(csv_filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        for row in csv_reader:
            # Assuming the second column (index 1) is column B
            if len(row) > 1 and row[1] == 'True':
                # Construct the relative path from current_directory to image
                relative_image_path = os.path.abspath(os.path.join('..', row[3][3:]))  # Construct relative path
                image_path = os.path.abspath(os.path.join(current_directory, relative_image_path))
                if os.path.exists(image_path):
                    image = np.array(Image.open(image_path))
                    plt.imshow(image)
                    plt.title(row[0] + ':   ' + row[4] + '   ' + row[5] + '   ' + row[6] + '   ' + row[7] + '   ')
                    plt.show()



# Provide the CSV file name as an argument
csv_filename = r'..\data\\attention_results\\crop_results.csv'
check_column_b(csv_filename)
