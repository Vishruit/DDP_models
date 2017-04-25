# Evaluate a saved model
from keras.models import Model, load_model
from keras.models import load_model
from models import model_from_json

import numpy as np
import tensorflow as tf
import cPickle, gzip, pickle, h5py
import argparse
import os, time, sys
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

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
