# Matplotlib imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reference counted: Helps with segmentation fault
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# Keras Imports
import keras
from keras import initializers
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback, ProgbarLogger, ReduceLROnPlateau
from keras.callbacks import LambdaCallback, CSVLogger
from keras.layers import Input, Dense, ZeroPadding2D, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Activation, Dropout
from keras.layers import Reshape, Conv2D, UpSampling3D, Conv3D, MaxPooling3D
from keras.layers.core import Lambda
from keras.metrics import categorical_accuracy, binary_accuracy
from keras.models import Model, load_model, model_from_json
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.utils.io_utils import HDF5Matrix
from keras.utils.np_utils import normalize
from keras.utils.vis_utils import plot_model

from keras_contrib.layers import Deconvolution3D

# Miscellaneous
import numpy as np
import tensorflow as tf
import cPickle, gzip, pickle
import argparse, h5py
import os, time, sys
