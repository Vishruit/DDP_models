# from utils import ensure_dir, argParser

dataset_path = './data_small_100.h5'
train_file_name,  dataset_keyword = '../data_small_100.h5', 'data_small'
image_height, image_width, image_depth=32,32,3
frames, height, width = 100, 256, 320
dropout_rate = 0.7
data_augmentation=False
append_CSVfile_FLAG = False
#data = (nsamples, 202*100*256*320) float32

# parse the arguments from the command line
args, argDict = argParser()
lr, batch_size, init_Code, num_epochs, model_initializer, save_dir, verbose, debug, restart, machine_code = argAssigner(args)


experiment_num = '8'

experiment_root = './exp'+experiment_num+'/'
visualization_filepath = './exp'+experiment_num+'/visualizations/'
visualization_filepath_test_time = './exp'+experiment_num+'/visualizations/Test_time/'
code_base_file_path = './exp' + experiment_num +'/code/'

filepath_best_weights='./exp'+experiment_num+'/save_dir/weights.best.hdf5'
filepath_chpkt_weights = './exp'+experiment_num+'/save_dir/CheckPoint/'
filepath_csvLogger = './exp'+experiment_num+'/save_dir/CheckPoint/csv_log_file.csv'

ensure_dir([visualization_filepath, filepath_best_weights, filepath_chpkt_weights, experiment_root])
ensure_dir([visualization_filepath_test_time, code_base_file_path])

img_size = 32
num_channels = 3
num_classes = 10

# if not debug:
data_slice_size = 202
train_split, valid_split, test_split = 7, 1.5, 1.5
# else:
# data_slice_size = 50
# train_split, valid_split, test_split = 1.5, 1.5, 7

video_index = [1,5,10,15,20,25,30]
frame_index = [1,5,10,25,40,50,60,75,90,99]
