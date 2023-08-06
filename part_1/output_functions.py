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
DEFAULT_TEST_RESULT_FOLDER: str = 'Test Results/'


def show_convolution(original_image: np.ndarray, convoluted_image: np.ndarray, fig_name: str):

    plt.subplot(2, 1, 1)
    plt.imshow(original_image)

    plt.title("Normal Image")
    plt.axis('off')

    # Second subplot - right side
    plt.subplot(2, 1, 2)
    plt.imshow(convoluted_image)
    plt.title("Filtered Image")
    plt.axis('off')

    plt.show()
    new_file_name = DEFAULT_TEST_RESULT_FOLDER + os.path.basename(fig_name)[:-4] + '_convolution.png'
    plt.savefig(new_file_name)

