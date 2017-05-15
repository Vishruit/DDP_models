inputData = Input(batch_shape=(None,3,None,None))

#First Layer
pad1 = ZeroPadding2D((100,100))(inputData)
conv1_1 = Convolution2D(64,3,3,activation='relu',border_mode='valid')(pad1)
conv1_2 = Convolution2D(64,3,3,activation='relu',border_mode='same')(conv1_1)
pool1 = MaxPooling2D((2,2), strides=(2,2))(conv1_2)

#Second Convolution
conv2_1 = Convolution2D(128,3,3,activation='relu',border_mode='same')(pool1)
conv2_2 = Convolution2D(128,3,3,activation='relu',border_mode='same')(conv2_1)
pool2 = MaxPooling2D((2,2), strides=(2,2))(conv2_2)

#Third Convolution
conv3_1 = Convolution2D(256,3,3,activation='relu',border_mode='same')(pool2)
conv3_2 = Convolution2D(256,3,3,activation='relu',border_mode='same')(conv3_1)
conv3_3 = Convolution2D(256,3,3,activation='relu',border_mode='same')(conv3_2)
pool3 = MaxPooling2D((2,2), strides=(2,2))(conv3_3)

#Fourth Convolution
conv4_1 = Convolution2D(512,3,3,activation='relu',border_mode='same')(pool3)
conv4_2 = Convolution2D(512,3,3,activation='relu',border_mode='same')(conv4_1)
conv4_3 = Convolution2D(512,3,3,activation='relu',border_mode='same')(conv4_2)
pool4 = MaxPooling2D((2,2), strides=(2,2))(conv4_3)

#Fifth Convolution
conv5_1 = Convolution2D(512,3,3,activation='relu',border_mode='same')(pool4)
conv5_2 = Convolution2D(512,3,3,activation='relu',border_mode='same')(conv5_1)
conv5_3 = Convolution2D(512,3,3,activation='relu',border_mode='same')(conv5_2)
pool5 = MaxPooling2D((2,2), strides=(2,2))(conv5_3)

#Fully Convolutional Layers 32s
fc6 = Convolution2D(4096,7,7,activation='relu',border_mode='valid')(pool5)
drop6 = Dropout(0.5)(fc6)
fc7 = Convolution2D(4096,1,1,activation='relu',border_mode='valid')(drop6)
drop7 = Dropout(0.5)(fc7)
score_fr = Convolution2D(21,1,1,border_mode='valid')(drop7)

#Deconv Layer
upscore  = Deconvolution2D(21,64,64,output_shape=(inputData.get_shape()),subsample=(32,32),bias=False)(score_fr)

flat = Flatten()(upscore)
pred32 = Dense(21,activation='softmax')(flat)
model = Model(input=inputData, output=[pred32])
