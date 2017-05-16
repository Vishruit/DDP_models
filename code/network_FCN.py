from imports_lib import *
from keras.layers import Conv2DTranspose

inputData = Input(batch_shape=(None,240,320,3))

#First Layer
pad1 = ZeroPadding2D((100,100))(inputData)
conv1_1 = Conv2D(64,(3,3),activation='relu',padding='valid')(pad1)
conv1_2 = Conv2D(64,(3,3),activation='relu',padding='same')(conv1_1)
pool1 = MaxPooling2D((2,2), strides=(2,2))(conv1_2)

#Second Convolution
conv2_1 = Conv2D(128,(3,3),activation='relu',padding='same')(pool1)
conv2_2 = Conv2D(128,(3,3),activation='relu',padding='same')(conv2_1)
pool2 = MaxPooling2D((2,2), strides=(2,2))(conv2_2)

#Third Convolution
conv3_1 = Conv2D(256,(3,3),activation='relu',padding='same')(pool2)
conv3_2 = Conv2D(256,(3,3),activation='relu',padding='same')(conv3_1)
conv3_3 = Conv2D(256,(3,3),activation='relu',padding='same')(conv3_2)
pool3 = MaxPooling2D((2,2), strides=(2,2))(conv3_3)

#Fourth Convolution
conv4_1 = Conv2D(512,(3,3),activation='relu',padding='same')(pool3)
conv4_2 = Conv2D(512,(3,3),activation='relu',padding='same')(conv4_1)
conv4_3 = Conv2D(512,(3,3),activation='relu',padding='same')(conv4_2)
pool4 = MaxPooling2D((2,2), strides=(2,2))(conv4_3)

#Fifth Convolution
conv5_1 = Conv2D(512,(3,3),activation='relu',padding='same')(pool4)
conv5_2 = Conv2D(512,(3,3),activation='relu',padding='same')(conv5_1)
conv5_3 = Conv2D(512,(3,3),activation='relu',padding='same')(conv5_2)
pool5 = MaxPooling2D((2,2), strides=(2,2))(conv5_3)

#Fully Convolutional Layers 32s
fc6 = Conv2D(4096,(7,7),activation='relu',padding='valid')(pool5)
drop6 = Dropout(0.5)(fc6)
fc7 = Conv2D(4096,(1,1),activation='relu',padding='valid')(drop6)
drop7 = Dropout(0.5)(fc7)
score_fr = Conv2D(21,(1,1),padding='valid')(drop7)

#Deconv Layer
upscore  = Conv2DTranspose(21,(64,64),output_shape=(inputData.get_shape()),strides=(32,32),bias=False)(score_fr)

# flat = Flatten()(upscore)
# pred32 = Dense(21,activation='softmax')(flat)
# model = Model(input=inputData, output=[pred32])
model = Model(inputData,upscore)
model.summary()
