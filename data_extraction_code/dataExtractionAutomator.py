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

# from IPython.display import display
# from IPython.display import Image
# from IPython.display import HTML

#make pngs at 150 dpi
'''
import matplotlib as mpl
mpl.rc("savefig", dpi=75)
mpl.rc('figure', figsize=(10,8))
'''

import numpy as np
import numpy as np
from pandas import HDFStore,DataFrame # create (or open) an hdf5 file and opens in append mode
import h5py

import os
import time
import sys

import pyradi.ryptw as ryptw
import pyradi.ryplot as ryplot
import pyradi.ryfiles as ryfiles

def getPTWFilePaths(directory,excludeFiles):
    file_paths = []
    file_name = []
    file_loc = []
    for (root, directories, files) in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            fileloc = os.path.join(root)
            if filename.endswith(".ptw") and filename!=excludeFiles:
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

def createHDFFile(filename, iter, numFrames,rows,cols):
    global file_Location_hdf
    filename = file_Location_hdf[iter] + '/' + filename + '.h5'
    ensure_dir(filename)
    hdf = h5py.File(filename, 'a')
    dataset = hdf.create_dataset("data_float32", shape=(numFrames,rows,cols), dtype=np.float32)
    return hdf, dataset

def createJPGFile(filename, iter):
    global file_Location_npy, ext
    filename = file_Location_jpg[iter] + '/' + filename + '/frame_'
    ensure_dir(filename)
    return filename

def saveHDF5Array(data, dataset, rows, cols, frame):

    dataset[frame-1,:,:] = data.reshape(1, rows ,cols)
    # dataset = f.create_dataset("data", data = data)
    # np.save(filename,data)
    pass

def savePandasHDF5Array(data, filename, iter, f=None):
    global file_Location_npy
    filename = file_Location_npy[iter] + '/' + filename
    if not ensure_dir(filename):
        # hdf = HDFStore('storage.h5')
        hdf = HDFStore(filename)
        # f = h5py.File(filename, 'a')
        # dataset = f.create_dataset("data", data = data)
        return f
    df =DataFrame(np.random.rand(5,3), columns=('A','B','C'))# put the dataset in the storage
    hdf.put('d1', df, format='table', data_columns=True)
    # dataset = f.create_dataset("data", data = data)
    # np.save(filename,data)
    pass

def saveNPArray(data, filename, iter):
    global file_Location_npy
    filename = file_Location_npy[iter] + '/' + filename
    ensure_dir(filename)
    np.save(filename,data)
    pass

def saveFrame(f, filename, frame):
    global ext
    ryfiles.rawFrameToImageFile(f, filename + str(frame-1)+ext)
    pass

def saveJPGPics(frames, filename, iter, initialFrame):
    global file_Location_npy
    filename = file_Location_jpg[iter] + '/' + filename + '/frame_'
    ensure_dir(filename)
    global ext
    # filename = file_Location_jpg[iter] + 'frame'
    i=initialFrame
    for frame in frames:
        ryfiles.rawFrameToImageFile(frame, filename+str(i)+ext)
        i+=1
    # np.save(filename,frame)
    pass

def autoPTW2NP(ptwfilepath, ptwfilename, iter):
    # ptwfile  = './PyradiSampleLWIR.ptw'
    # outfilename = 'PyradiSampleLWIR.txt'

    print 'File: ' + str(iter)
    print 'Filename: ' + ptwfilename
    print 'FilePath: ' + ptwfilepath

    # try:
    #     header = ryptw.readPTWHeader(ptwfilepath)
    #     pass
    # except Exception as e:
    #     raise
    # finally:
    #     global expFile, it
    #     it+=1
    #     expFile.append(ptwfilepath)
    #     print 'Check this file'
    #     return
    #     pass

    header = ryptw.readPTWHeader(ptwfilepath)
    header.h_PaletteAGC = 1
    # header.h_PaletteFull
    rows = header.h_Rows
    cols = header.h_Cols
    # ryptw.showHeader(header) # Suppressed Output
    # numFrames = 10
    numFrames = header.h_lastframe # Total frames in the video
    initialFrame = 1
    finalFrame = initialFrame + numFrames

    '''
    # Test time
    numFrames = 100 # Testing time
    initialFrame = 1 + 1*500 + 0*100
    finalFrame = initialFrame + 100
    '''

    if numFrames >= initialFrame:
        if numFrames < finalFrame-1:
            finalFrame = numFrames
        # start
        # framesToLoad = range(1, numFrames+1, 1)
        framesToLoad = range(initialFrame, finalFrame, 1)
        frames = len(framesToLoad)
        data, fheaders  = ryptw.getPTWFrame (header, framesToLoad[0])
        # hdf = saveHDF5Array(data, ptwfilename, iter, hdf, rows,cols)

        # hdf, dataset = createHDFFile(ptwfilename, iter, numFrames, rows, cols)
        # saveHDF5Array(data, dataset,  rows, cols, initialFrame-1)
        filename = createJPGFile(ptwfilename, iter)

        for frame in framesToLoad[1:]:
            f, fheaders = (ryptw.getPTWFrame (header, frame))

            # data = np.concatenate((data, f)) # TODO

            # print str(frame) + '/' + str(numFrames)
            sys.stdout.write('[Frame: %s of %s] \r' % (frame, numFrames))
            sys.stdout.flush()
            # saveNPArray(frame, ptwfilename, iter)
            # saveJPGPics(frame, ptwfilename, iter)
            # saveFrame(f, ptwfilename, iter=iter)
            saveFrame(f, filename, frame) # works
            # saveHDF5Array(f, dataset,  rows, cols, frame)

        print '\n'
        '''
        print data.shape

        img = data.reshape(frames, rows ,cols)
        print(img.shape)
        saveNPArray(img, ptwfilename, iter)
        saveJPGPics(img, ptwfilename, iter, initialFrame)
        '''
        # hdf.close()
        return data

def createSmallHDFData_Randomly(ptwfilepath, ptwfilename, iter):
    print 'File: ' + str(iter)
    print 'Filename: ' + ptwfilename
    print 'FilePath: ' + ptwfilepath

    header = ryptw.readPTWHeader(ptwfilepath)
    header.h_PaletteAGC = 1
    rows = header.h_Rows
    cols = header.h_Cols

    # totalFrames =
    frameBatch = 300
    # numFrames = 10
    numFrames = header.h_lastframe # Total frames in the video
    initialFrame = 100
    finalFrame = numFrames + 1
    # TODO check for errors
    if numFrames >= initialFrame:
        if numFrames < finalFrame-1:
            finalFrame = numFrames

        framesToLoad = np.random.random_integers(initialFrame, finalFrame, numFrames)
        frames = len(framesToLoad)
        data, fheaders  = ryptw.getPTWFrame (header, framesToLoad[0])

        hdf, dataset = createHDFFile(ptwfilename, iter, numFrames, rows, cols)
        saveHDF5Array(data, dataset,  rows, cols, initialFrame-1)

        # TODO to do it all frames at a time and not frame by frame
        for frame in framesToLoad[1:]:
            f, fheaders = (ryptw.getPTWFrame (header, frame))
            sys.stdout.write('[Frame: %s of %s] \r' % (frame, numFrames))
            sys.stdout.flush()
            saveHDF5Array(f, dataset,  rows, cols, frame)

        print '\n'
        hdf.close()
        return data



excludeFiles = '1_0.5ws_4wfr_18lpm.ptw'
# expFile = ['']
# it =0
# saveFolderLocation = './Extracted Data/'

# Save format for Image .jpg or .png
ext = '.png'

# Root Location for the original data
dataLocation = '/media/prabakaran/MYHDD/Vishruit/DATA'
# dataLocation = 'F:\\Vishruit\\DATA'
# Home location for saving the format file
dataLocation1 = '/home/prabakaran/Vishruit/DDP/DATA'
data_npyLocation = dataLocation1 + '_npy'
data_jpgLocation = dataLocation1 + '_jpg'
data_hdfLocation = dataLocation1 + '_hdf'

# Actual filepaths and filenames list
[file_paths, file_names, file_locs] = getPTWFilePaths(dataLocation, excludeFiles)

# Creating filepaths for desired file format
# Also creating fileLocation paths
file_paths_npy = []
file_paths_jpg = []
file_paths_hdf = []
file_Location_npy = []
file_Location_jpg = []
file_Location_hdf = []
for file_path in file_paths:
    file_paths_npy.append( data_npyLocation + file_path[len(dataLocation):len(file_path)] )
    file_paths_jpg.append( data_jpgLocation + file_path[len(dataLocation):len(file_path)] )
    file_paths_hdf.append( data_hdfLocation + file_path[len(dataLocation):len(file_path)] )

# Save folder locations
for file_loc in file_locs:
    file_Location_npy.append( data_npyLocation + file_loc[len(dataLocation):len(file_loc)] )
    file_Location_jpg.append( data_jpgLocation + file_loc[len(dataLocation):len(file_loc)] )
    file_Location_hdf.append( data_hdfLocation + file_loc[len(dataLocation):len(file_loc)] )


print file_paths[1]


# Use this method to convert files
'''
for iter in range(len(file_paths)): # TODO len(file_paths)-202
    autoPTW2NP(file_paths[iter], file_names[iter], iter)
'''

# a=1
for iter in range(10*(a-1),10*a,1): # TODO len(file_paths)-202   20*(a-1)
    autoPTW2NP(file_paths[iter], file_names[iter], iter)

# print expFile
# print it

print 'All is well !!! Finished.'


# TODO call the createSmallHDF function here
# Create a smaller dataset
np.random.seed(1)





'''
filename = "/my/directory/filename.txt"
dir = os.path.dirname(filename)
'''

'''
total = 1000
i = 0
while i < total:
    dynamicPrint(i, total, status='Doing very long job')
    time.sleep(0.01)  # emulating long-playing job
    i += 1
print 'qwe\n'
i=0
while i < total:
    dynamicPrint(i, total, status='Doing very job')
    time.sleep(0.01)  # emulating long-playing job
    i += 1
'''

'''
In [1]: import h5py

In [2]: f = h5py.File('Expt2.ptw.h5', 'r')

In [3]: t = f['data_unlimited'][()]

In [4]: t

'''





# from joblib import Parallel, delayed
# import multiprocessing
# num_cores = multiprocessing.cpu_count()
# Parallel(n_jobs=num_cores)(delayed(autoPTW2NP)(file_paths[iter], file_names[iter]) for iter in range(10))
