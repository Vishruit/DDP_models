import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from keras import initializers
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback, ProgbarLogger, ReduceLROnPlateau
from keras.callbacks import LambdaCallback, CSVLogger
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Activation, Dropout
from keras.layers import Reshape,  Conv2D, UpSampling3D, Conv3D, MaxPooling3D
from keras.layers.core import Lambda
from keras.metrics import categorical_accuracy, binary_accuracy
from keras.models import Model, load_model
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.utils.io_utils import HDF5Matrix
from keras.utils.np_utils import normalize
from keras.utils.vis_utils import plot_model
import numpy as np
import tensorflow as tf
import cPickle, gzip, pickle, h5py
import argparse
import os, time, sys


# tf.device('/gpu:0')

# model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# del model  # deletes the existing model
# model = load_model('my_model.h5')


class TestCallback(Callback):
    def __init__(self, test_data, valid_data):
        self.test_data = test_data
        self.valid_data = valid_data

    def on_epoch_end(self, epoch, logs={}):
        global batch_size
        x_val, y_val = self.valid_data
        x_test, y_test = self.test_data
        train_loss, train_acc = logs.get('loss'), logs.get('binary_accuracy')
        val_loss, val_acc = self.model.evaluate(x_val, y_val, verbose=0, batch_size=batch_size)
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
        print('\n \x1b[6;30;42m :=:> \x1b[0m train_loss: {0:.3f}, train_acc: {1:.2f}|| val_loss: {2:.3f}, val_acc: {3:.2f} || test_loss: {4:.3f}, test_acc: {5:.2f}\n'.format(np.asscalar(train_loss), np.asscalar(train_acc), np.asscalar(val_loss), np.asscalar(val_acc), np.asscalar(test_loss), np.asscalar(test_acc)))

        #TODO TODO
        video_index = [1,5,10,15,20,25,30]
        frame_index = [1,5,10,25,40,50,60,75,90,99]
        decoded_imgs = model.predict(x_test[video_index], batch_size=batch_size)
        for (video, vid_it) in zip(video_index, range(len(video_index))):
            plt.figure(figsize=(20, 4))
            for i in range(len(frame_index)):
                ax = plt.subplot(2, len(frame_index), i + 1)
                plt.imshow(x_test[video].reshape(frames, 256, 320)[frame_index[i],...])
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                ax = plt.subplot(2, len(frame_index), i + len(frame_index) + 1)
                plt.imshow(decoded_imgs[vid_it].reshape(frames, 256, 320)[frame_index[i],...])
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                plt.savefig( visualization_filepath+ 'reconstruction_vid_'+str(video)+'_Epoch_'+epoch+'.png' )

class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.model.validation_data[0])
        self.aucs.append(roc_auc_score(self.model.validation_data[1], y_pred))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def define_model(init,lr,verbose,restart):
    channels = 1
    global dropout_rate, frames, height, width, visualization_filepath

    input_img = Input(shape=(frames, height, width))
    x = Reshape((frames, height, width, 1))(input_img)

    # x = BatchNormalization(mode=2, axis=1, input_shape=(ROWS, COLS, CHANNELS))
    # x = BatchNormalization(mode=2, axis=1)

    x = Conv3D(16, (3, 3, 3), activation='relu', padding='same', kernel_initializer=init)(x)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)

    x = Conv3D(8, (3, 3, 3), activation='relu', padding='same', kernel_initializer=init)(x)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)

    x = Conv3D(8, (3, 3, 3), activation='relu', padding='same', kernel_initializer=init)(x)
    encoded = MaxPooling3D((5, 2, 2), padding='same')(x)

    # at this point the representation is (8, 4, 4) i.e. 128-dimensional

    x = Conv3D(8, (3, 3, 3), activation='relu', padding='same', kernel_initializer=init)(encoded)
    x = UpSampling3D((5, 2, 2))(x)

    x = Conv3D(8, (3, 3, 3), activation='relu', padding='same', kernel_initializer=init)(x)
    x = UpSampling3D((2, 2, 2))(x)

    x = Conv3D(16, (3, 3, 3), activation='relu', padding='same', kernel_initializer=init)(x)
    x = UpSampling3D((2, 2, 2))(x)

    decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)
    decoded = Reshape((frames, height, width))(decoded)

    # x = Dropout(dropout_rate)(x)
    # POOL3_flattened = Flatten()(DROP3)
    # FC1 = Dense(1024, activation='relu', kernel_initializer=init)(POOL3_flattened)
    # BN = BatchNormalization()(SOFTMAX_precomputation)
    # SOFTMAX = Activation('softmax')(BN)

    model = Model(input_img, decoded)
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=[binary_accuracy])

    # model.compile(optimizer=adam, loss='categorical_crossentropy', \
    #                 class_mode="categorical", metrics=[categorical_accuracy]) # TODO binary_crossentropy

    model.summary()

    load_model_weights(model,restart)

    if verbose:
        print model.summary()
        # grapher.plot(model, './visualizations/model_grapher.png')
        plot_model(model, to_file=visualization_filepath+'plot_model.png')
    return model

def load_model_weights(model, restart=False):
    global filepath_best_weights
    # load weights
    if restart:
        model.load_weights(filepath_best_weights)

def argAssigner(args):
    # TODO check the data types
    global lr,batch_size,init_Code,model_initializer,save_dir, verbose,debug,restart
    lr = float(args.lr)
    batch_size = int(args.batch_size)
    save_dir = args.save_dir
    init_Code = int(args.init)
    # TODO check normal or uniform
    model_initializer = initializers.glorot_normal(seed=None) if init_Code == 1 else initializers.he_normal(seed=None)
    verbose = args.verbose
    restart = args.restart
    debug = args.debug

def argParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',default=0.0001, help='Initial learning rate (eta)', type=float)
    parser.add_argument('--batch_size',default=2, help='Batch size, -1 for vanilla gradient descent')
    parser.add_argument('--init',default=1, help='Initializer: 1 for Xavier init and 2 for He init')
    parser.add_argument('--save_dir',default='./save_dir/', help='Saves model parameters in this directory')
    # Custom debugging args
    parser.add_argument('-d','--debug', help='For devs only, takes in no arguments', action="store_true")
    parser.add_argument('-v',"--verbose", help="Increase output verbosity",action="store_true")
    parser.add_argument('-r',"--restart", help="Restarts the network",action="store_true")
    args = parser.parse_args()

    if args.verbose:
        print("verbosity turned on")
        for k in args.__dict__:
            print ('\x1b[6;30;42m' + str(k) + '\x1b[0m' + '\t\t' + str(args.__dict__[k]))

    return args, args.__dict__

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def read_data():
    def data_preprocess(data):
        flag_isMultipleVids = 0
        if len(data.shape) == 4:
            video,sample,height,width = data.shape
            data = data.reshape((video*sample,height,width))
            flag_isMultipleVids = 1
        maxVal = np.max(data, axis = -1)
        maxVal = np.max(maxVal, axis = -1)
        minVal = np.min(data, axis = -1)
        minVal = np.min(minVal, axis = -1)
        for i in range(len(maxVal)):
            data[i,...] = (data[i,...]-minVal[i]) / (maxVal[i]- minVal[i]+0.001)
        if flag_isMultipleVids == 1:
            data = data.reshape((video,sample,height,width))
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
    # return (train_set_data, train_set_labels, valid_set_data, valid_set_labels, test_set_data, test_set_labels)
    return (train_set_data, train_set_data, valid_set_data, valid_set_data, test_set_data, test_set_data)


def ensure_dir(files):
    for f in files:
        d = os.path.dirname(f)
        if not os.path.exists(d):
            os.makedirs(d)
    return 1

def plot_group(history):
    # print history_record
    # print(history_record.history.keys())
    # ['loss', 'val_binary_accuracy', 'lr', 'val_loss', 'binary_accuracy']
    # summarize history for accuracy
    global experiment_root
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig( experiment_root + 'plot_valtrain_acc.jpg')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig( experiment_root + 'plot_valtrain_loss.jpg')

if __name__ == "__main__":
    # parse the arguments from the command line
    args, argDict = argParser()
    argAssigner(args)

    dataset_path = './data_small_100.h5'
    train_file_name,  dataset_keyword = '../data_small_100.h5', 'data_small'
    image_height, image_width, image_depth=32,32,3
    frames, height, width = 100, 256, 320
    dropout_rate = 0.7
    data_augmentation=False
    append_CSVfile_FLAG = False
    #data = (nsamples, 202*100*256*320) float32

    experiment_num = '2'
    experiment_root = './exp'+experiment_num+'/'
    visualization_filepath = './exp'+experiment_num+'/visualizations/'
    filepath_best_weights='./exp'+experiment_num+'/save_dir/weights.best.hdf5'
    filepath_chpkt_weights = './exp'+experiment_num+'/save_dir/CheckPoint/'
    filepath_csvLogger = './exp'+experiment_num+'/save_dir/CheckPoint/csv_log_file.csv'

    # filepath="./save_dir/exp1/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

    ensure_dir([visualization_filepath,filepath_best_weights, filepath_chpkt_weights, experiment_root])

    model = define_model(model_initializer, lr, verbose,restart=restart)

    img_size = 32
    num_channels = 3
    num_classes = 10
    total_training_epochs = 200
    train_set_data, train_set_labels, valid_set_data, valid_set_labels, test_set_data, test_set_labels = read_data()
    (x_train, x_valid, x_test, y_train, y_valid, y_test) = (train_set_data, valid_set_data, test_set_data, train_set_labels, valid_set_labels, test_set_labels)

    # Callbacks
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=verbose, mode='auto')
    checkpointer = ModelCheckpoint(filepath=filepath_best_weights, verbose=1, save_best_only=True)  # checkpoint
    chkpt = ModelCheckpoint(filepath_chpkt_weights+'weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    tensorboard = TensorBoard(log_dir=experiment_root+'logs2', histogram_freq=0, write_graph=False)  # Run using tensorboard --logdir=./logs
    testcallback = TestCallback((x_test, y_test), (x_valid, y_valid))
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    csvLogger = CSVLogger(filepath_csvLogger, separator=',', append=append_CSVfile_FLAG)
    # histories = Histories()

    callback_list = [es, checkpointer, chkpt, tensorboard, testcallback, reduceLR, csvLogger]

    if not debug:
        history = model.fit(x_train, y_train,
                                epochs=total_training_epochs,
                                batch_size=batch_size,
                                shuffle='batch',
                                validation_data=(x_valid, y_valid),
                                callbacks=callback_list,
                                initial_epoch=0)
    else:
        history = model.fit(x_train, x_train,\
                        epochs=2,\
                        batch_size=2,\
                        shuffle='batch',\
                        validation_data=(x_valid, x_valid), \
                        callbacks=callback_list)

    with open('model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)

    # print(history.history.keys())
    plot_group(history)

    decoded_imgs = model.predict(x_test, batch_size=batch_size)
    # video_index = [1,5,10,50,100,150,200]
    video_index = [1,5,10,15,20,25,30]
    frame_index = [1,5,10,25,40,50,60,75,90,99]

    for video in video_index:
        plt.figure(figsize=(20, 4))
        print('Processing video:',video)
        for i in range(len(frame_index)):
            # decoded_imgs = autoencoder.predict(x_test[i].reshape(1, x_test.shape[1], x_test.shape[2], x_test.shape[3])
            # display original
            ax = plt.subplot(2, len(frame_index), i + 1)
            # TODO remove hard links
            # print(x_test.shape)
            plt.imshow(x_test[video].reshape(frames, 256, 320)[frame_index[i],...])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            #plt.savefig('original.jpg')
            # display reconstruction
            ax = plt.subplot(2, len(frame_index), i + len(frame_index) + 1)
            plt.imshow(decoded_imgs[video].reshape(frames, 256, 320)[frame_index[i],...])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.savefig( visualization_filepath+ 'reconstruction_vid'+str(video)+'.png' )
