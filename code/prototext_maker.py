import random
import numpy as np
import os
import shutil
import time
import sys
np.random.seed(1)


def getJPGFilePaths(directory,excludeFiles):
    file_paths = []
    file_name = []
    file_loc = []
    global ext
    for (root, directories, files) in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            fileloc = os.path.join(root)
            if filename.endswith("." + ext) and filename!=excludeFiles:
                file_paths.append(filepath)  # Add it to the list.
                file_name.append(filename)
                file_loc.append(fileloc)
            # break #To do a deep search in all the sub-directories
    return file_paths, file_name, file_loc

def getLabelFilePaths(directory,excludeFiles):
    file_paths = []
    file_name = []
    file_loc = []
    global label_file_name
    for (root, directories, files) in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            fileloc = os.path.join(root)
            if filename.endswith(label_file_name) and filename!=excludeFiles:
                file_paths.append(filepath)  # Add it to the list.
                file_name.append(filename)
                file_loc.append(fileloc)
            # break #To do a deep search in all the sub-directories
    return file_paths, file_name, file_loc

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
        return 0
    return 1

def createPrototextForCaffe(imgfilepaths, imgfilenames, imgfilelocs):
    global dataset_data, ext, dataset_label, prototext_file_location
    labelLocation = dataset_label
    dataLocation = dataset_data
    imgfilelocs = np.unique(imgfilelocs)
    num_vids = len(imgfilelocs)
    files = [open(prototext_file_location[0], 'w'), open(prototext_file_location[1], 'w'), open(prototext_file_location[2], 'w')]
    # sampleSize = 100
    # 'frame_1002.png' format of the filename
    i=1
    for fileloc in imgfilelocs:
        print fileloc
        if i < int(0.7*num_vids):
            prototext = files[0] # train
            print('train', i)
        elif i < int(0.85*num_vids):
            prototext = files[1] # valid
            print('valid', i)
        else:
            prototext = files[2] # test
            print('test', i)
        i+=1
        # sample_fileloc = data_jpg_save + fileloc[len(labelLocation):len(fileloc)]
        relative_folderpath = fileloc[len(labelLocation):len(fileloc)]

        data_fileloc_actual = dataLocation + relative_folderpath
        label_fileloc_actual = labelLocation + relative_folderpath  # = fileloc


        [file_paths, file_name, file_loc] = getJPGFilePaths(data_fileloc_actual,[])
        totalSize = len(file_name)
        framesToLoad = range(1,totalSize+1,1)
        # frameBatch = min(totalSize, sampleSize)
        framesToLoad = np.sort(framesToLoad)
        # print framesToLoad
        # ensure_dir(sample_fileloc)
        for frame in framesToLoad:
            src_file_name_data = data_fileloc_actual + '/frame_' + str(frame) + '.' + ext
            src_file_name_label = label_fileloc_actual + '/label.png'
            # print('TODO write prototext format here')
            prototext.write(src_file_name_data + ' ' + src_file_name_label)
            prototext.write('\n')
            # shutil.copy(src_file_name_data, dst_file_name_data)
    # save the file at the location required
    files[0].close()
    files[1].close()
    files[2].close()
    # prototext.close()
    pass


excludeFiles = []
ext = 'png'
datasetLocation = '/home/prabakaran/Vishruit/DDP/DATA_caffe'
prototext_file_location = [datasetLocation+'/train.txt', datasetLocation+'/valid.txt', datasetLocation+'/test.txt']

label_file_name = 'label.png'

dataset_data = datasetLocation + '/DATA_jpg'
dataset_label = datasetLocation + '/DATA_mapped'

# Actual filepaths and filenames list
[file_paths_label, file_names_label, file_locs_label] = getLabelFilePaths(dataset_label, excludeFiles)
# [file_paths_data, file_names_data, file_locs_data] = getJPGFilePaths(dataset_data, excludeFiles)

# # Actual filepaths and filenames list
# [file_paths, file_names, file_locs] = getJPGFilePaths(dataLocation, excludeFiles)
print file_paths_label[1]
# print file_paths_data[1]

# numInstances = len(file_paths)
print 'Start'
createPrototextForCaffe(file_paths_label, file_names_label, file_locs_label)
print 'All is well !!! Finished.'
