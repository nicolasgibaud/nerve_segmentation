# -*- coding: utf-8 -*-
"""Loading tools

Module containing the functions to import and format the data.
"""

import sys
try:
    sys.path.append("C:/Users/admin/Documents/Codes/") # if not in PYTHHONPATH
except ImportError:
    pass

try:
    sys.path.append("/home/nicolas") # if not in PYTHHONPATH
except ImportError:
    pass


from nerve_segmentation.project_config import DATA_DIR, ROOT_DIR, IMAGE_SIZE,\
                                        IMAGE_LENGTH, IMAGE_WIDTH
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_image_number_from_path(im_path):
    im_number = os.path.split(im_path)[-1]
    im_number = im_number.split(".")[0]
    return im_number


def create_images_generator(data_path=None):
    """
    Returns a generator that yield the numpy array of the images files contained
    in repository at path *data_path*
    """
    if data_path is None:
        data_path = os.path.join(DATA_DIR, "raw", "train", "images")

    im_files = np.array([os.path.join(data_path, f) for f in sorted(os.listdir(data_path))])
    number_im_files = [get_image_number_from_path(im) for im in im_files]
    number_im_files_filled = [num.split("_")[0] + "_" + num.split("_")[1].zfill(4) \
                                    for num in number_im_files]
    sorting_mask= np.array(number_im_files_filled).argsort()
    im_files_sorted = im_files[sorting_mask]
    for im_file in im_files_sorted:
        try:
            im_array = plt.imread(im_file)
            print "loading image : %s" %im_file
            yield im_array
        except IOError:
            pass

def create_dense_labels_from_sparse(sparse_labels_vector, image_size=IMAGE_SIZE):
    """
    Takes as input a list of the sparse labels and returns the corresponding
    dense array of 0 and 1 as a 1-D vector

    Parameters
    ----------
    sparse_labels : list
        the list of sparse labels

    Returns
    -------

    Raises
    ------
    """
    dense_labels_vector = np.zeros(image_size)

    for i in range(0, len(sparse_labels_vector)-1, 2):
        labels_1_mask = range(sparse_labels_vector[i], sparse_labels_vector[i] +\
                                    sparse_labels_vector[i+1])
        dense_labels_vector[labels_1_mask] = 1

    return dense_labels_vector


def transform_vector_to_matrix(vector, image_length=IMAGE_LENGTH, \
                    image_width=IMAGE_WIDTH):
        matrix = [vector[i*IMAGE_WIDTH:(i+1)*IMAGE_WIDTH] for i in range(IMAGE_LENGTH)]
        return matrix

def import_labels_csv(file_path=None):
    if file_path is None:
        file_path = os.path.join(DATA_DIR, "raw", "train_masks.csv")
    sparse_labels = pd.read_csv(file_path)
    return sparse_labels


def convert_string_labels_to_list(string_labels):
    list_labels_string = string_labels.split(' ')
    list_labels_int = [int(elt) for elt in list_labels_string]
    return list_labels_int


def create_labels_generator(file_path=None):
    """
    Returns a generator that yield the numpy array of the images files contained
    in repository at path *data_path*
    """
    labels_csv = import_labels_csv(file_path)
    for im_num in range(len(labels_csv)):
        labels = labels_csv["pixels"].iloc[im_num]
        try:
            labels_int = convert_string_labels_to_list(labels)
        except AttributeError: #Nan
            labels_int = []
        yield labels_int


def create_image_and_labels_generator(images_path=None, file_path=None):
    images_generator = create_images_generator(data_path=images_path)
    labels_generator = create_labels_generator(file_path=file_path)

    while True:
        yield images_generator.next(), labels_generator.next()
