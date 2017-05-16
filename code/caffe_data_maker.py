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

def createDatasetForCaffe(imgfilepaths, imgfilenames, imgfilelocs):
    global dataLocation, ext, data_jpg_save, data_label_save, labelLocation
    imgfilelocs = np.unique(imgfilelocs)
    # sampleSize = 100
    # 'frame_1002.png' format of the filename
    for fileloc in imgfilelocs:
        print fileloc
        # sample_fileloc = data_jpg_save + fileloc[len(labelLocation):len(fileloc)]
        relative_folderpath = fileloc[len(labelLocation):len(fileloc)]
        data_fileloc_save = data_jpg_save + relative_folderpath
        label_fileloc_save = data_label_save + relative_folderpath

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
            dst_file_name_data = data_fileloc_save + '/frame_' + str(frame) + '.' + ext
            src_file_name_data = data_fileloc_actual + '/frame_' + str(frame) + '.' + ext
            ensure_dir(dst_file_name_data)
            shutil.copy(src_file_name_data, dst_file_name_data)

        # Saves the label files to the label_fileloc_save respectively
        dst_file_name_label = label_fileloc_save + '/label.png'
        src_file_name_label = label_fileloc_actual + '/label.png'
        ensure_dir(dst_file_name_label)
        shutil.copy(src_file_name_label, dst_file_name_label)
    pass

excludeFiles = []
ext = 'png'
dataLocation = '/home/prabakaran/Vishruit/DDP/DATA_jpg'
labelLocation = '/home/prabakaran/Vishruit/DDP/DATA_mapped'
dataLocation_save = '/home/prabakaran/Vishruit/DDP/DATA_caffe'

label_file_name = 'label.png'

data_jpg_save = dataLocation_save + '/DATA_jpg'
data_label_save = dataLocation_save + '/DATA_mapped'

# Actual filepaths and filenames list
[file_paths_label, file_names_label, file_locs_label] = getLabelFilePaths(labelLocation, excludeFiles)

print 'Start'
createDatasetForCaffe(file_paths_label, file_names_label, file_locs_label)
print 'All is well !!! Finished.'
