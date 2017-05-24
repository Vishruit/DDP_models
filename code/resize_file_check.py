import random
import numpy as np
import os
import shutil
import time
import sys
np.random.seed(1)
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc

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

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
        return 0
    return 1

def resize_all_files(imgfilepaths, imgfilenames, imgfilelocs):
    global datasetLocation, ext, height, width
    imgfilelocs = np.unique(imgfilelocs)
    num_vids = len(imgfilelocs)
    i = 1
    for fileloc in imgfilelocs:
        # print fileloc
        # sample_fileloc = data_jpg_save + fileloc[len(labelLocation):len(fileloc)]
        relative_folderpath = fileloc[len(datasetLocation):len(fileloc)]
        data_fileloc_actual = datasetLocation + relative_folderpath

        [file_paths, file_name, file_loc] = getJPGFilePaths(data_fileloc_actual,[])
        totalSize = len(file_name)
        framesToLoad = range(1,totalSize+1,1)
        framesToLoad = np.sort(framesToLoad)

        for fpath in file_paths:
            src_file_name_data = fpath
            im = Image.open(src_file_name_data)
            # print 'width: %d - height: %d' % im.size
            # continue
            if im.size != (width, height): # Prints cols * rows
                temp = np.zeros((height,width))
                img = np.array(im)
                temp[ :img.shape[0], :img.shape[1] ] = img + temp[ :img.shape[0], :img.shape[1] ]
                scipy.misc.imsave(src_file_name_data, temp)
                print(i)
                i += 1
    pass


excludeFiles = []
ext = 'png'

height, width = 360, 480
# datasetLocation = '/partition1/vishruit/soft/DATA_caffe'
datasetLocation = '/home/prabakaran/Vishruit/DDP/DATA_caffe/'
# datasetLocation = '/partition1/vishruit/soft/DATA_caffe/DATA_jpg'


# Actual filepaths and filenames list
[file_paths_label, file_names_label, file_locs_label] = getJPGFilePaths(datasetLocation, excludeFiles)

# # Actual filepaths and filenames list
print file_paths_label[1]
# print file_paths_data[1]

# numInstances = len(file_paths)
print 'Start'
resize_all_files(file_paths_label, file_names_label, file_locs_label)
print 'All is well !!! Finished.'
