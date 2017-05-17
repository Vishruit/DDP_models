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

def apply_label_tag(pixel):
    if pixel == 0:
        return 1 # Background pixel
    else:
        tag = int(pixel / 25 ) + 2 # +1 for background and +1 for indexing from 1

    if tag > 11:
        return 11
    else:
        return tag

def create_label_tags(imgfilepaths, imgfilenames, imgfilelocs):
    global datasetLocation, ext, height, width
    imgfilelocs = np.unique(imgfilelocs)
    num_vids = len(imgfilelocs)
    i = 1
    for fileloc in imgfilelocs:
        print fileloc
        relative_folderpath = fileloc[len(datasetLocation):len(fileloc)]
        data_fileloc_actual = datasetLocation + relative_folderpath

        [file_paths, file_name, file_loc] = getJPGFilePaths(data_fileloc_actual,[])
        totalSize = len(file_name)
        print totalSize
        framesToLoad = range(1,totalSize+1,1)
        framesToLoad = np.sort(framesToLoad)

        for fpath in file_paths:
            src_file_name_data = fpath
            im = Image.open(src_file_name_data)
            temp = np.zeros(im.size)
            im_array = np.array(im)
            for i in range(im_array.shape[0]):
                for j in range(im_array.shape[1]):
                    im_array[i,j] = apply_label_tag(im_array[i,j])
            scipy.misc.imsave(src_file_name_data, temp)
            print(i)
            i += 1
    pass


excludeFiles = []
ext = 'png'
# height, width = 256,320
datasetLocation = '/partition1/vishruit/soft/DATA_caffe/DATA_mapped'

# Actual filepaths and filenames list
[file_paths_label, file_names_label, file_locs_label] = getJPGFilePaths(datasetLocation, excludeFiles)

# # Actual filepaths and filenames list
print file_paths_label[1]
# print file_paths_data[1]

# numInstances = len(file_paths)
print 'Start'
create_label_tags(file_paths_label, file_names_label, file_locs_label)
print 'All is well !!! Finished.'
