from imports_lib import *
from utils import *
from network import *
from constants import *


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
    data_slice_size = 100
    train_split, valid_split, test_split = 1.5, 1.5, 7
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


if __name__ == "__main__":

    # parse the arguments from the command line
    args, argDict = argParser()
    lr, batch_size, init_Code, num_epochs, model_initializer, save_dir, verbose, debug, restart = argAssigner(args)

    ensure_dir([visualization_filepath, filepath_best_weights, filepath_chpkt_weights, experiment_root])
    ensure_dir([visualization_filepath_test_time])

    # import keras.backend.tensorflow_backend as K

    # with K.tf.device('/gpu:0'):
    #    config = K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    #    config.gpu_options.allow_growth = True
    #    K.set_session(K.tf.Session(config=config))
    #    model = define_model(model_initializer, lr, verbose,restart=restart)
    model = define_model(model_initializer, lr, verbose,restart=restart)

    total_training_epochs = num_epochs
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
    #callback_list = [es, checkpointer, chkpt, tensorboard, reduceLR, csvLogger]

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

    # print(history.history.keys())
    plot_group(history)

    decoded_imgs = model.predict(x_test, batch_size=batch_size)

    plot_video(decoded_imgs)

    save_model_and_weights(model)

    # Save models
    # with open('model.json', 'w') as outfile:
    #     json.dump(model.to_json(), outfile)
