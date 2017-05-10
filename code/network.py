from imports_lib import *
from utils import *
from constants import *
from main import *
from utils import load_model_weights

def define_model(init,lr,verbose,restart):
    channels = 1
    global dropout_rate, frames, height, width, visualization_filepath

    input_img = Input(shape=(frames, height, width))
    x = Reshape((frames, height, width, 1))(input_img)

    # x = BatchNormalization(mode=2, axis=1, input_shape=(ROWS, COLS, CHANNELS))
    # x = BatchNormalization(mode=2, axis=1)
    # x = BatchNormalization()(x)

    # x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer=init)(x)
    # x = MaxPooling3D((2, 2, 2), padding='same')(x)

    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer=init)(x)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)

    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer=init)(x)
    x = MaxPooling3D((5, 2, 2), padding='same')(x)

    x = Conv3D(16, (3, 3, 3), activation='relu', padding='same', kernel_initializer=init)(x)
    x = MaxPooling3D((1, 2, 2), padding='same')(x)

    x = Conv3D(8, (3, 3, 3), activation='relu', padding='same', kernel_initializer=init)(x)
    x = MaxPooling3D((1, 2, 2), padding='same')(x)

    x = Conv3D(8, (3, 3, 3), activation='relu', padding='same', kernel_initializer=init)(x)
    encoded = MaxPooling3D((5, 2, 2), padding='same')(x)

    # at this point the representation is (8, 4, 4) i.e. 128-dimensional

    x = Conv3D(8, (3, 3, 3), activation='relu', padding='same', kernel_initializer=init)(encoded)
    x = UpSampling3D((5, 2, 2))(x)

    x = Conv3D(8, (3, 3, 3), activation='relu', padding='same', kernel_initializer=init)(x)
    x = UpSampling3D((1, 2, 2))(x)

    x = Conv3D(16, (3, 3, 3), activation='relu', padding='same', kernel_initializer=init)(x)
    x = UpSampling3D((1, 2, 2))(x)

    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer=init)(x)
    x = UpSampling3D((5, 2, 2))(x)

    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer=init)(x)
    x = UpSampling3D((2, 2, 2))(x)

    # x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer=init)(x)
    # x = UpSampling3D((2, 2, 2))(x)

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

    #if verbose:
        #print (model.summary()
        # )grapher.plot(model, './visualizations/model_grapher.png')
        #plot_model(model, to_file=visualization_filepath+'plot_model.png')
    return model
