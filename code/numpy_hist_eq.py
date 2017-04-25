from keras.models import Model, load_model
from keras.models import load_model
from keras.models import model_from_json
from keras.utils.io_utils import HDF5Matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cPickle, gzip, pickle, h5py
import argparse
import os, time, sys

# TODO remove temporary fix
# train_file_name,  dataset_keyword = '../data_small_100.h5', 'data_small'
train_file_name,  dataset_keyword = '../Exp_17.ptw.h5', 'data_float32'

def hist_eq():
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

def data_preprocess(data):
    flag_isMultipleVids = 0
    if len(data.shape) == 4:
        sample,frames,height,width = data.shape
        data = data.reshape((sample*frames,height,width))
        flag_isMultipleVids = 1
    maxVal = np.max(data, axis = -1)
    maxVal = np.max(maxVal, axis = -1)
    minVal = np.min(data, axis = -1)
    minVal = np.min(minVal, axis = -1)
    for i in range(len(maxVal)):
        data[i,...] = (data[i,...]-minVal[i]) / (maxVal[i]- minVal[i]+0.001)
    if flag_isMultipleVids == 1:
        data = data.reshape((sample,frames,height,width))
    return data

# data_slice_size = 202
# train_split, valid_split, test_split = 7, 1.5, 1.5

train_set_data = HDF5Matrix(train_file_name, dataset_keyword)


image = train_set_data[1000]


hist,bins = np.histogram(image,500)

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

# plt.plot(cdf_normalized, color = 'b')
# plt.hist(img.flatten(),1000, color = 'r')

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint32')

image_equalized = np.interp(image, bins[:-1], cdf)
