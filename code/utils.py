# Utility functions
from imports_lib import *
from constants import *

def argAssigner(args):
    # TODO check the data types
    global lr,batch_size,init_Code,num_epochs,model_initializer,save_dir, verbose,debug,restart
    lr = float(args.lr)
    batch_size = int(args.batch_size)
    num_epochs = int(args.num_epochs)
    save_dir = args.save_dir
    init_Code = int(args.init)
    # TODO check normal or uniform
    model_initializer = initializers.glorot_normal(seed=None) if init_Code == 1 else initializers.he_normal(seed=None)
    verbose = args.verbose
    restart = args.restart
    debug = args.debug
    return lr,batch_size,init_Code,num_epochs,model_initializer,save_dir, verbose,debug,restart

def argParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',default=0.0001, help='Initial learning rate (eta)', type=float)
    parser.add_argument('--batch_size',default=2, help='Batch size, -1 for vanilla gradient descent')
    parser.add_argument('--num_epochs',default=100, help='Saves model parameters in this directory')
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

def ensure_dir(files):
    for f in files:
        d = os.path.dirname(f)
        if not os.path.exists(d):
            os.makedirs(d)
    return 1

def load_model_weights(model, restart=False):
    global filepath_best_weights
    if restart:
        model.load_weights(filepath_best_weights)
    return

def save_model_and_weights(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(experiment_root+"model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(experiment_root+"model.h5")
    print("Saved model to disk")

def plot_group(history):
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

def plot_video(decoded_imgs, x_test):
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
            plt.savefig( visualization_filepath+'reconstruction_vid'+str(video)+'.png' )
        plt.close()

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

    global train_file_name,  dataset_keyword, data_slice_size, train_split, valid_split, test_split
    
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
