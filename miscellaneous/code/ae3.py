import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling3D, Conv3D, MaxPooling3D
from keras.models import Model
import h5py
from keras.models import load_model
import tensorflow as tf
tf.device('/gpu:0')

# http://stackoverflow.com/questions/35074549/how-to-load-a-model-from-an-hdf5-file-in-keras

# https://keras.io/getting-started/faq/#how-can-i-run-keras-on-gpu
# model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# del model  # deletes the existing modelzsdgdzsbzs
#
# # returns a compiled model
# # identical to the previous one
# model = load_model('my_model.h5')

from keras.datasets import mnist
import numpy as np

# (x_train, _), (x_test, _) = mnist.load_data()

with h5py.File('../data_small.h5', 'r') as hdf:
    # data = hdf['data_small'][:,:150,:,:]
    data = hdf['data_small'][:1]
print ('Hi1')
# data = data.reshape((len(data)*data.shape[1]),1,data.shape[2],data.shape[3])
data = data.reshape(len(data),data.shape[1],data.shape[2],data.shape[3],1)
print ('Hi2')
print(data.shape)


x_train = data[:int(6*len(data)/10)]
x_test = data[int(6*len(data)/10):]
# print(x_train.shape, x_test.shape)

# TODO incase memory is less
# del data
maxVal= np.max(data)
# TODO normalization + mem issues
x_train = x_train.astype('float32') / maxVal
x_test = x_test.astype('float32') / maxVal
# x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
# x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))
print ('Hi3')

# input_img = Input(shape=(1, data.shape[2],data.shape[3]))
input_img = Input(shape=(data.shape[1],data.shape[2],data.shape[3],1))

x = Conv3D(16, 3, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling3D((2, 2, 2), border_mode='same')(x)
x = Conv3D(8, 3, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling3D((2, 2, 2), border_mode='same')(x)
x = Conv3D(8, 3, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling3D((3, 2, 2), border_mode='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Conv3D(8, 3, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling3D((3, 2, 2))(x)
x = Conv3D(8, 3, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling3D((2, 2, 2))(x)
x = Conv3D(16, 3, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling3D((2, 2, 2))(x)
decoded = Conv3D(1, 3, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.summary()


autoencoder.fit(x_train, x_train,\
                nb_epoch=10,\
                batch_size=10,\
                shuffle=True,\
                validation_data=(x_test, x_test),\
                callbacks=)

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)

    # TODO remove hard links
    plt.imshow(x_test[i].reshape(256, 320))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #plt.savefig('original.jpg')

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(256, 320))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig('reconstruction.png')
