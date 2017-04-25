# Evaluate a saved model
from keras.models import Model, load_model
from keras.models import load_model
from keras.models import model_from_json
from keras.utils.io_utils import HDF5Matrix

# from main import read_data

import numpy as np
import tensorflow as tf
import cPickle, gzip, pickle, h5py
import argparse
import os, time, sys

# TODO remove temporary fix
train_file_name,  dataset_keyword = '../data_small_100.h5', 'data_small'

def read_data():
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

    global train_file_name,  dataset_keyword
    data_slice_size = 202
    train_split, valid_split, test_split = 7, 1.5, 1.5
    train_set_data = HDF5Matrix(train_file_name, dataset_keyword, start=0, \
                                                                  end=int( train_split *data_slice_size/10), \
                                                                  normalizer=lambda x: data_preprocess(x))
    valid_set_data = HDF5Matrix(train_file_name, dataset_keyword, start=int( train_split *data_slice_size/10), \
                                                                  end=int( (train_split+valid_split) *data_slice_size/10), \
                                                                  normalizer=lambda x: data_preprocess(x))
    test_set_data = HDF5Matrix(train_file_name, dataset_keyword, start=int( (train_split+valid_split) *data_slice_size/10), \
                                                                 end= 1 *data_slice_size, \
                                                                 normalizer=lambda x: data_preprocess(x))
    return (train_set_data, train_set_data, valid_set_data, valid_set_data, test_set_data, test_set_data)

# later...
experiment_num = '3'
experiment_root = './exp'+experiment_num+'/'

# load json and create model
json_file = open(experiment_root+'model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(experiment_root+"model.h5")
print("Loaded model from disk")

# Loading data
train_set_data, train_set_data, valid_set_data, valid_set_data, test_set_data, test_set_data = read_data()
X = test_set_data
Y = X

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
