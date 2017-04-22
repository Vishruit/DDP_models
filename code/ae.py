import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers import Input, Dense,Reshape,  Conv2D, MaxPooling2D, UpSampling3D, Conv3D, MaxPooling3D
from keras.models import Model
from keras.layers.core import Lambda
from keras.utils.io_utils import HDF5Matrix
from keras.utils.np_utils import normalize
import keras.backend as K
import h5py
from keras.models import load_model
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback, CSVLogger
import sys
import numpy as np
import matplotlib.pyplot as plt
tf.device('/gpu:0')

# Add commentsgfswsrfssgsdhge
from keras.datasets import mnist
import numpy as np

# with h5py.File('../data_small.h5', 'r') as hdf:
#     # data = hdf['data_small'][:,:150,:,:]
#     data = hdf['data_small'][:10]
# print ('Hi1')
# # data = data.reshape((len(data)*data.shape[1]),1,data.shape[2],data.shape[3])
# data = data.reshape(len(data),data.shape[1],data.shape[2],data.shape[3],1)
# print ('Hi2')
# print(data.shape)
#
#
# x_train = data[:int(6*len(data)/10)]
# x_test = data[int(6*len(data)/10):]
# print(x_train.shape, x_test.shape)

def data_preprocess(data):
    # data = data.reshape((len(data)*data.shape[1]),data.shape[2],data.shape[3],1)
    # print (data.shape)
    # print (data.dtype, type(data))
    maxVal = np.max(data)
    data = data / (maxVal+0.00001)
    # sys.exit()
    # data = data.astype('float32')
    # normalize(data, axis=1, order=2)
    # data = data[:int(6*len(data)/10)]
    # print(x_train.shape)
    # x_test = data[int(6*len(data)/10):]
    # maxVal= K.max(data)
    # print("#####################################")
    # data = data / maxVal
    # print(type(data), data.dtype, K.is_keras_tensor(data))
    # data = np.array
    return data


with h5py.File('../data_small_100.h5', 'r') as hdf:
    # data = hdf['data_small'][:20]
    # print(len(hdf['data_small']))
    data_slice_size = 202
    x_train = HDF5Matrix('../data_small_100.h5', 'data_small', start=0, end=int(6*data_slice_size/10), normalizer=lambda x: data_preprocess(x))
    x_test = HDF5Matrix('../data_small_100.h5', 'data_small', start=int(6*data_slice_size/10), end=data_slice_size, normalizer=lambda x: data_preprocess(x))
    # print (data_hdf.shape, type(data_hdf))


print(x_train.shape)
print ('Hi3')


frames = x_train.shape[1]
height = x_train.shape[2]
width = x_train.shape[3]

input_img = Input(shape=(frames, height, width))
x = Reshape((x_train.shape[1],x_train.shape[2],x_train.shape[3], 1))(input_img)

x = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(x)
x = MaxPooling3D((2, 2, 2), padding='same')(x)
x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(x)
x = MaxPooling3D((2, 2, 2), padding='same')(x)
x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling3D((5, 2, 2), padding='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(encoded)
x = UpSampling3D((5, 2, 2))(x)
x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(x)
x = UpSampling3D((2, 2, 2))(x)
x = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(x)
x = UpSampling3D((2, 2, 2))(x)
decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)
decoded = Reshape((x_train.shape[1],x_train.shape[2],x_train.shape[3]))(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()

chkpt = ModelCheckpoint('./CheckPoint/weights_adam.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
# Plot the loss after every epoch.
plot_loss_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: plt.plot(np.arange(epoch),
                      logs['loss']))
csvLogger = CSVLogger('./CheckPoint/csv_log_file_50_5_adamReal_200.csv', separator=',', append=False)
# callback_list = [chkpt, reduce_lr, plot_loss_callback, csvLogger]
callback_list = [chkpt, reduce_lr, csvLogger]


autoencoder.fit(x_train, x_train,\
                epochs=50,\
                batch_size=2,\
                shuffle='batch',\
                validation_data=(x_test, x_test), \
                callbacks=callback_list)

# decoded_imgs = autoencoder.predict(x_test[:n])

n = 10
# decoded_imgs = autoencoder.predict(x_test[0:n])
plt.figure(figsize=(20, 4))
for i in range(n):
    decoded_imgs = autoencoder.predict(x_test[i].reshape(1, x_test.shape[1], x_test.shape[2], x_test.shape[3]))
    # display original
    ax = plt.subplot(2, n, i + 1)

    # TODO remove hard links
    print(x_test[i].shape)
    plt.imshow(x_test[i].reshape(frames, 256, 320)[i,...])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #plt.savefig('original.jpg')

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs.reshape(frames, 256, 320)[i,...])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig('reconstruction1_50_5_adamReal_200.png')
