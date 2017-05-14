# Evaluaytion pipeline is ready! :)
# Evaluate a saved model
from imports_lib import *
from utils import *

# TODO remove temporary fix
train_file_name,  dataset_keyword = '../data_small_100.h5', 'data_small'


# later...
experiment_num = '6'
experiment_root = './exp'+experiment_num+'/'
visualization_filepath = './exp'+experiment_num+'/visualizations/'
visualization_filepath_test_time = './exp'+experiment_num+'/visualizations/Test_time/'
frames = 100
ensure_dir([visualization_filepath_test_time])

batch_size = 2

# load json and create model
json_file = open(experiment_root+'model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(experiment_root+"model.h5")
print("Loaded model from disk")

# Loading data
train_set_data, train_set_data, valid_set_data, valid_set_data, test_set_data, test_set_data = read_data()
X = test_set_data
Y = X
x_test = test_set_data

# evaluate loaded model on test data
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
score = model.evaluate(X, Y, verbose=0, batch_size=batch_size)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

decoded_imgs = model.predict(x_test, batch_size=batch_size)
# video_index = [1,5,10,50,100,150,200]
video_index = [1,5,10,15,20,25,30]
frame_index = [1,5,10,25,40,50,60,75,90,99]

for video in video_index:
    plt.figure(figsize=(20, 4))
    print('Processing video:',video)
    for i in range(len(frame_index)):
        # display original
        ax = plt.subplot(2, len(frame_index), i + 1)
        # TODO remove hard links
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
        plt.savefig( visualization_filepath_test_time+ 'reconstruction_vid'+str(video)+'.png' )
    plt.close()
