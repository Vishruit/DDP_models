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

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
        return 0
    return 1

def createSmallJPGData_Randomly(imgfilepaths, imgfilenames, imgfilelocs):
    global data_jpgLocation, dataLocation, ext
    imgfilelocs = np.unique(imgfilelocs)
    sampleSize = 100
    # 'frame_1002.png' format of the filename
    for fileloc in imgfilelocs:
        print fileloc
        sample_fileloc = data_jpgLocation + fileloc[len(dataLocation):len(fileloc)]
        [file_paths, file_name, file_loc] = getJPGFilePaths(fileloc,[])
        totalSize = len(file_name)
        framesToLoad = random.sample(range(1,totalSize+1,1), sampleSize) if (totalSize>=sampleSize) else range(1,totalSize+1,1)
        # frameBatch = min(totalSize, sampleSize)
        framesToLoad = np.sort(framesToLoad)
        print framesToLoad
        # ensure_dir(sample_fileloc)
        for frame in framesToLoad:
            dst_file_name = sample_fileloc + '/frame_' + str(frame) + '.' + ext
            src_file_name = fileloc + '/frame_' + str(frame) + '.' + ext
            ensure_dir(dst_file_name)
            shutil.copy(src_file_name, dst_file_name)


excludeFiles = []
ext = 'png'
# Root Location for the original data
dataLocation = '/home/prabakaran/Vishruit/DDP/DATA_jpg'
# Home location for saving the format file
dataLocation_save = '/home/prabakaran/Vishruit/DDP/DATA_small'

data_jpgLocation = dataLocation_save + '_' + ext

# Actual filepaths and filenames list
[file_paths, file_names, file_locs] = getJPGFilePaths(dataLocation, excludeFiles)
print file_paths[1]

numInstances = len(file_paths)
print 'Start'
createSmallJPGData_Randomly(file_paths, file_names, file_locs)
print 'All is well !!! Finished.'
