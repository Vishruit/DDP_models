# This file automatically finds all the .ptw files in a given directory and
# converts them to a numpy array for reseach purposes. It bypasses the need of
# a proprietary software like ALTAIR to extract and read the data.

'''
Before running this file perform 'pip install pyradi'
Please run the test file before running this file and follow the below
instructions to remove the error in the pyradi package, if any.

Comment out the line 516 in the file 'ryptw.py'
Header.h_Framatone = ord(headerinfo[3076:3077])
This ensures smooth running as required for this program.
'''

import random
import numpy as np
from pandas import HDFStore,DataFrame # create (or open) an hdf5 file and opens in append mode
import h5py

import os
import time
import sys

def getH5FilePaths(directory,excludeFiles):
    file_paths = []
    file_name = []
    file_loc = []
    for (root, directories, files) in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            fileloc = os.path.join(root)
            if filename.endswith(".h5") and filename!=excludeFiles:
                file_paths.append(filepath)  # Add it to the list.
                file_name.append(filename)
                file_loc.append(fileloc)
            # break #To do a deep search in all the sub-directories
    return file_paths, file_name, file_loc

def dynamicPrint(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
        return 0
    return 1

def createHDFFile(filename, numInstances, frameBatch, rows, cols, iter=0):
    # global file_Location_hdf
    global data_hdfLocation
    filename = data_hdfLocation + '/' + filename
    ensure_dir(filename)
    hdf = h5py.File(filename, 'a')
    dataset = hdf.create_dataset("data_small", shape=(numInstances, frameBatch, rows, cols), dtype=np.float32)
    return hdf, dataset

def saveHDF5Array(data, dataset, frameBatch, rows, cols, frame):
    dataset[frame,:frameBatch,:rows,:cols] = data.reshape(1,frameBatch, rows ,cols)
    pass

def autoPTW2NP(ptwfilepath, ptwfilename, iter):
    header = ryptw.readPTWHeader(ptwfilepath)
    header.h_PaletteAGC = 1
    rows = header.h_Rows
    cols = header.h_Cols
    # numFrames = 10
    numFrames = header.h_lastframe # Total frames in the video
    initialFrame = 1
    finalFrame = initialFrame + numFrames
    if numFrames >= initialFrame:
        if numFrames < finalFrame-1:
            finalFrame = numFrames
        framesToLoad = range(initialFrame, finalFrame, 1)
        frames = len(framesToLoad)
        data, fheaders  = ryptw.getPTWFrame (header, framesToLoad[0])
        hdf, dataset = createHDFFile(ptwfilename, iter, numFrames, rows, cols)
        saveHDF5Array(data, dataset,  rows, cols, initialFrame-1)
        # filename = createJPGFile(ptwfilename, iter)
        for frame in framesToLoad[1:]:
            f, fheaders = (ryptw.getPTWFrame (header, frame))
            sys.stdout.write('[Frame: %s of %s] \r' % (frame, numFrames))
            sys.stdout.flush()
            saveHDF5Array(f, dataset,  rows, cols, frame)
            # saveFrame(f, filename, frame) # works
        hdf.close()
        print '\n'
        return data

def createSmallHDFData_Randomly(ptwfilepaths, ptwfilenames):
    startPoint = 0 # for resuming purposes
    for iter in range(startPoint, len(file_paths),1):
        print 'iter', iter
        ptwfilename = ptwfilenames[iter]
        ptwfilepath = ptwfilepaths[iter]
        print 'File: ' + str(iter)
        print 'Filename: ' + ptwfilename
        print 'FilePath: ' + ptwfilepath
        frameBatch = 100

        with h5py.File(ptwfilepath, 'r') as hf:
            data = hf['data_float32'][:]
            numFrames = len(data)
            framesToLoad = random.sample(range(numFrames), frameBatch) if (numFrames>=frameBatch) else range(numFrames)
            frameBatch = min(numFrames, frameBatch)
            framesToLoad = np.sort(framesToLoad)
            print framesToLoad
            data = data[framesToLoad]
            print data.shape
            rows, cols = data.shape[1],data.shape[2]
            print frameBatch,rows,cols

        print iter
        global numInstances
        if iter==startPoint:
            hdf, dataset = createHDFFile('data_small_100.h5', numInstances, frameBatch, 256, 320, iter) # 256 and 320 obtained from observation
        saveHDF5Array(data, dataset, frameBatch, rows, cols, iter)
        sys.stdout.write('[File: %s of %s] \r' % (iter, numInstances))
        sys.stdout.flush()

        # TODO correct this mess
        # for frame in range(len(framesToLoad[1:])):
        #     sys.stdout.write('[Frame: %s of %s] \r' % (frame, numFrames))
        #     sys.stdout.flush()
        #     saveHDF5Array(data, dataset,frameBatch, rows, cols, frame)
    print 'Hi\nBaby'
    hdf.close()

def printFileSizes(file_paths):
    a = []
    for path in file_paths:
        with h5py.File(path, 'r') as hf:
            print hf['data_float32'][:].shape
            a.append(hf['data_float32'][:].shape)
    return a


excludeFiles = []
# Root Location for the original data
dataLocation = '/home/prabakaran/Vishruit/DDP/DATA_hdf'
# Home location for saving the format file
dataLocation1 = '/home/prabakaran/Vishruit/DDP/DATA_small'

data_hdfLocation = dataLocation1 + '_hdf'

# Actual filepaths and filenames list
[file_paths, file_names, file_locs] = getH5FilePaths(dataLocation, excludeFiles)

# Creating filepaths for desired file format
# Also creating fileLocation paths
file_paths_hdf = []
file_Location_hdf = []

for file_path in file_paths:
    file_paths_hdf.append( data_hdfLocation + file_path[len(dataLocation):len(file_path)] )
# Save folder locations
for file_loc in file_locs:
    file_Location_hdf.append( data_hdfLocation + file_loc[len(dataLocation):len(file_loc)] )

print file_paths[1]

numInstances = len(file_paths)

# Use this method to convert files
np.random.seed(1)# Create a smaller dataset

# a = printFileSizes(file_paths)
createSmallHDFData_Randomly(file_paths, file_names)

# for iter in range(len(file_paths)-201): # TODO len(file_paths)
#     print file_paths[iter]
#     data = createSmallHDFData_Randomly(file_paths[iter], file_names[iter], iter)


# TODO call the createSmallHDF function here
# a=1
# for iter in range(10*(a-1),10*a,1): # TODO len(file_paths)-202   20*(a-1)
#
#     autoPTW2NP(file_paths[iter], file_names[iter], iter)

print 'All is well !!! Finished.'
