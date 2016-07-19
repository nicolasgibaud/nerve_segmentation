# -*- coding: utf-8 -*-
"""Loading tools

Module containing the functions to import and format the data.
"""

import sys
sys.path.append("C:/Users/admin/Documents/Codes/") # if not in PYTHHONPATH

from nerve_segmentation.project_config import DATA_DIR, ROOT_DIR
import os
import matplotlib.pyplot as plt


def create_images_generator(data_path=None):
    """
    Returns a generator that yield the numpy array of the images files contained
    in repository at path *data_path*
    """
    if data_path is None:
        data_path = os.path.join(DATA_DIR, "raw", "train")

    im_files = [os.path.join(data_path, f) for f in os.listdir(data_path)]

    for im_file in im_files:
        try:
            im_array = plt.imread(im_file)
            yield im_array
        except IOError:
            pass
