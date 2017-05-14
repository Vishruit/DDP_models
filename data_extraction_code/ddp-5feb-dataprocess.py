# IPython log file

run dataExtractionAutomator.py
run dataExtractionAutomator.py
import h5py
ls
cd E
cd Extracted\ Data
ls
np.load(we.npy)
np.load('we.npy')
#[Out]# array(['qwerty', 'asdfgh'], 
#[Out]#       dtype='|S6')
t = np.load('we.npy')
t
#[Out]# array(['qwerty', 'asdfgh'], 
#[Out]#       dtype='|S6')
type(t)
#[Out]# <type 'numpy.ndarray'>
type(t[:])
#[Out]# <type 'numpy.ndarray'>
type(t[:2])
#[Out]# <type 'numpy.ndarray'>
type(t[:3])
#[Out]# <type 'numpy.ndarray'>
 f = h5py.File('we.npy', 'w')
f
#[Out]# <HDF5 file "we.npy" (mode r+)>
f[1]
f[:]
logstart?
logstart?
logstart?
# Sun, 05 Feb 2017 15:30:27
logstart -o -r -t ddp-5feb-dataprocess.py append
# Sun, 05 Feb 2017 15:30:33
ls
# Sun, 05 Feb 2017 15:30:38
ls -l
# Sun, 05 Feb 2017 15:30:47
!head
# Sun, 05 Feb 2017 15:31:09
ls
# Sun, 05 Feb 2017 15:31:17
!head ddp-5feb-dataprocess.py
# Sun, 05 Feb 2017 15:33:09
import h5py
# Sun, 05 Feb 2017 15:33:35
history
# Sun, 05 Feb 2017 15:34:07
f
#[Out]# <HDF5 file "we.npy" (mode r+)>
# Sun, 05 Feb 2017 15:34:12
f.filename
#[Out]# u'we.npy'
# Sun, 05 Feb 2017 15:34:24
f.items
#[Out]# <bound method File.items of <HDF5 file "we.npy" (mode r+)>>
# Sun, 05 Feb 2017 15:34:29
ls
# Sun, 05 Feb 2017 16:13:02
def saveNPArray(data, filename, iter):
        global file_Location_npy
        filename = file_Location_npy[iter] + '\\' + filename
        ensure_dir(filename)
        np.save(filename,data)
        pass

# Sun, 05 Feb 2017 16:13:22
saveNPArray([2 2], 'qw')
# Sun, 05 Feb 2017 16:13:32
t = [2 2]
# Sun, 05 Feb 2017 16:13:35
t = [2 ,2]
# Sun, 05 Feb 2017 16:13:40
saveNPArray(t, 'qw')
# Sun, 05 Feb 2017 16:14:11
def saveHDF5Array(data, filename, iter, f=None):
        global file_Location_npy
        filename = file_Location_npy[iter] + '\\' + filename
        if !ensure_dir(filename):
                f = h5py.File(filename, 'w')
                return f
        dataset = f.create_dataset("data", data = data)
        # np.save(filename,data)
        pass
# Sun, 05 Feb 2017 16:14:15
saveNPArray(t, 'qw')
# Sun, 05 Feb 2017 16:14:46
def saveHDF5Array(data, filename, iter, f=None):
        global file_Location_npy
        filename = file_Location_npy[iter] + '\\' + filename
        if !ensure_dir(filename):
                f = h5py.File(filename, 'w')
                return f
        dataset = f.create_dataset("data", data = data)
        # np.save(filename,data)
        pass
# Sun, 05 Feb 2017 16:15:02
def saveHDF5Array(data, filename, iter, f=None):
        global file_Location_npy
        filename = file_Location_npy[iter] + '\\' + filename
        if not ensure_dir(filename):
                f = h5py.File(filename, 'w')
                return f
        dataset = f.create_dataset("data", data = data)
        # np.save(filename,data)
        pass

# Sun, 05 Feb 2017 16:15:12
saveNPArray(t, 'qw')
# Sun, 05 Feb 2017 16:15:17
saveNPArray(t, 'qw',t)
# Sun, 05 Feb 2017 16:15:37
saveNPArray(t, 'qw',1)
# Sun, 05 Feb 2017 16:15:45
ls
# Sun, 05 Feb 2017 16:16:02
filename
# Sun, 05 Feb 2017 16:17:55
f
#[Out]# <HDF5 file "we.npy" (mode r+)>
# Sun, 05 Feb 2017 16:18:09
f = saveNPArray(t, 'qw',1)
# Sun, 05 Feb 2017 16:18:11
f
# Sun, 05 Feb 2017 16:18:19
type(f)
#[Out]# NoneType
# Sun, 05 Feb 2017 16:25:19
t 
#[Out]# [2, 2]
# Sun, 05 Feb 2017 16:25:26
t = [t 4]
# Sun, 05 Feb 2017 16:25:34
t = t + [3 4]
# Sun, 05 Feb 2017 16:25:37
t = t + [3,4]
# Sun, 05 Feb 2017 16:25:40
t 
#[Out]# [2, 2, 3, 4]
# Sun, 05 Feb 2017 16:25:52
t = t;[34] 
#[Out]# [34]
# Sun, 05 Feb 2017 16:25:54
t
#[Out]# [2, 2, 3, 4]
# Sun, 05 Feb 2017 16:26:00
[3]
#[Out]# [3]
# Sun, 05 Feb 2017 16:26:20
t = [t; 1 ,2 ,3 ,4]
# Sun, 05 Feb 2017 16:26:22
t
#[Out]# [2, 2, 3, 4]
# Sun, 05 Feb 2017 16:26:56
t.append([1 2 3  4])
# Sun, 05 Feb 2017 16:27:02
t.append([1 ,2 ,3,  4])
# Sun, 05 Feb 2017 16:27:03
t
#[Out]# [2, 2, 3, 4, [1, 2, 3, 4]]
# Sun, 05 Feb 2017 16:27:16
t= []
# Sun, 05 Feb 2017 16:27:26
t.append([2,2])
# Sun, 05 Feb 2017 16:27:28
t.append([2,4])
# Sun, 05 Feb 2017 16:27:31
t.append([2,5])
# Sun, 05 Feb 2017 16:27:33
t
#[Out]# [[2, 2], [2, 4], [2, 5]]
# Sun, 05 Feb 2017 16:27:35
t[1]
#[Out]# [2, 4]
# Sun, 05 Feb 2017 16:27:40
t[0]
#[Out]# [2, 2]
# Sun, 05 Feb 2017 16:28:25
ls
# Sun, 05 Feb 2017 16:28:41
f = h5py.File('qwert.h5', 'w')
# Sun, 05 Feb 2017 16:28:43
ls
# Sun, 05 Feb 2017 16:28:51
t
#[Out]# [[2, 2], [2, 4], [2, 5]]
# Sun, 05 Feb 2017 16:28:59
dataset = f.create_dataset("data", data = t)
# Sun, 05 Feb 2017 16:29:04
dataset
#[Out]# <HDF5 dataset "data": shape (3, 2), type "<i4">
# Sun, 05 Feb 2017 16:29:50
f
#[Out]# <HDF5 file "qwert.h5" (mode r+)>
# Sun, 05 Feb 2017 16:29:53
f.
# Sun, 05 Feb 2017 16:30:26
f['data']
#[Out]# <HDF5 dataset "data": shape (3, 2), type "<i4">
# Sun, 05 Feb 2017 16:30:32
f['/data']
#[Out]# <HDF5 dataset "data": shape (3, 2), type "<i4">
# Sun, 05 Feb 2017 16:31:50
hdf5dump
# Sun, 05 Feb 2017 16:32:18
h5py.Dataset
#[Out]# h5py._hl.dataset.Dataset
# Sun, 05 Feb 2017 16:33:06
dataset.value
#[Out]# array([[2, 2],
#[Out]#        [2, 4],
#[Out]#        [2, 5]])
# Sun, 05 Feb 2017 16:42:44
f('data')
# Sun, 05 Feb 2017 16:43:02
f['data']
#[Out]# <HDF5 dataset "data": shape (3, 2), type "<i4">
# Sun, 05 Feb 2017 16:43:31
f['data'].append([4,4])
# Sun, 05 Feb 2017 16:43:44
x = f['data']
# Sun, 05 Feb 2017 16:43:48
type(x)
#[Out]# h5py._hl.dataset.Dataset
# Sun, 05 Feb 2017 16:45:24
f['data'].keys()
# Sun, 05 Feb 2017 16:45:37
f
#[Out]# <HDF5 file "qwert.h5" (mode r+)>
# Sun, 05 Feb 2017 16:45:48
f.keys()
#[Out]# [u'data']
# Sun, 05 Feb 2017 16:47:10
t = [[2,3;2,3],[2,4;4,5]]
# Sun, 05 Feb 2017 16:47:13
t
#[Out]# [[2, 2], [2, 4], [2, 5]]
# Sun, 05 Feb 2017 16:48:14
t = [[[2,3],[2,3]],[[2,4],[4,5]]]
# Sun, 05 Feb 2017 16:48:15
t
#[Out]# [[[2, 3], [2, 3]], [[2, 4], [4, 5]]]
# Sun, 05 Feb 2017 16:48:20
t.shap
# Sun, 05 Feb 2017 16:48:23
t.shape()
# Sun, 05 Feb 2017 16:48:24
t.shape
# Sun, 05 Feb 2017 16:49:06
t = [[[2,3][2,3]],[[2,4][4,5]]]
# Sun, 05 Feb 2017 16:49:11
t = [[[2,3] [2,3]],[[2,4] [4,5]]]
# Sun, 05 Feb 2017 16:52:15
dataset = f.create_dataset("data", data = t)
# Sun, 05 Feb 2017 16:57:57
dataset = f.f.\\\\\\\ ("data", data = t)
# Sun, 05 Feb 2017 16:58:01
f.close
#[Out]# <bound method File.close of <HDF5 file "qwert.h5" (mode r+)>>
# Sun, 05 Feb 2017 16:58:04
f.close()
# Sun, 05 Feb 2017 16:58:06
f
#[Out]# <Closed HDF5 file>
# Sun, 05 Feb 2017 16:58:16
ls
# Sun, 05 Feb 2017 16:58:30
f = h5py.File(filename, 'a')
# Sun, 05 Feb 2017 16:58:38
f = h5py.File('qwert.h5', 'w')
# Sun, 05 Feb 2017 16:58:49
dataset = f.create_dataset("data", data = data)
# Sun, 05 Feb 2017 16:58:54
dataset = f.create_dataset("data", data = t)
# Sun, 05 Feb 2017 16:58:55
dataset = f.create_dataset("data", data = t)
# Sun, 05 Feb 2017 16:59:07
f
#[Out]# <HDF5 file "qwert.h5" (mode r+)>
# Sun, 05 Feb 2017 16:59:13
f.keys
#[Out]# <bound method File.keys of <HDF5 file "qwert.h5" (mode r+)>>
# Sun, 05 Feb 2017 16:59:16
f.keys()
#[Out]# [u'data']
# Sun, 05 Feb 2017 16:59:27
f.close()
# Sun, 05 Feb 2017 17:00:06
!h5dump qwert.h5
# Sun, 05 Feb 2017 17:25:49
a = np.random.random(size=(2,2,2))
# Sun, 05 Feb 2017 17:25:50
a
#[Out]# array([[[ 0.65355133,  0.86820977],
#[Out]#         [ 0.89261818,  0.55109183]],
#[Out]# 
#[Out]#        [[ 0.55809384,  0.06275186],
#[Out]#         [ 0.7018475 ,  0.67516623]]])
# Sun, 05 Feb 2017 17:26:04
a = np.random.random(size=(2,3,4))
# Sun, 05 Feb 2017 17:26:06
a
#[Out]# array([[[ 0.6743129 ,  0.2235317 ,  0.36808721,  0.79703754],
#[Out]#         [ 0.33636327,  0.11517274,  0.45340833,  0.89546714],
#[Out]#         [ 0.2575046 ,  0.04164981,  0.66210853,  0.09263139]],
#[Out]# 
#[Out]#        [[ 0.13964075,  0.89193979,  0.71639801,  0.72148317],
#[Out]#         [ 0.40059789,  0.33611509,  0.86973319,  0.95078016],
#[Out]#         [ 0.32503119,  0.92155489,  0.5659259 ,  0.5286179 ]]])
# Sun, 05 Feb 2017 17:26:32
a[1,:,:]
#[Out]# array([[ 0.13964075,  0.89193979,  0.71639801,  0.72148317],
#[Out]#        [ 0.40059789,  0.33611509,  0.86973319,  0.95078016],
#[Out]#        [ 0.32503119,  0.92155489,  0.5659259 ,  0.5286179 ]])
# Sun, 05 Feb 2017 17:27:02
h5f = h5py.File('data.h5', 'a')
# Sun, 05 Feb 2017 17:27:03
ls
# Sun, 05 Feb 2017 17:27:33
h5f.create_dataset('dataset_1', data=a)
#[Out]# <HDF5 dataset "dataset_1": shape (2, 3, 4), type "<f8">
# Sun, 05 Feb 2017 17:28:20
a
#[Out]# array([[[ 0.6743129 ,  0.2235317 ,  0.36808721,  0.79703754],
#[Out]#         [ 0.33636327,  0.11517274,  0.45340833,  0.89546714],
#[Out]#         [ 0.2575046 ,  0.04164981,  0.66210853,  0.09263139]],
#[Out]# 
#[Out]#        [[ 0.13964075,  0.89193979,  0.71639801,  0.72148317],
#[Out]#         [ 0.40059789,  0.33611509,  0.86973319,  0.95078016],
#[Out]#         [ 0.32503119,  0.92155489,  0.5659259 ,  0.5286179 ]]])
# Sun, 05 Feb 2017 17:29:39
print(h5f.shape)
# Sun, 05 Feb 2017 17:31:13
h = h5f.create_dataset('dataset_1', data=a)
# Sun, 05 Feb 2017 17:31:58
h = h5f['data']
# Sun, 05 Feb 2017 17:32:09
h = h5f['dataset1']
# Sun, 05 Feb 2017 17:32:24
h5f.close()
# Sun, 05 Feb 2017 17:32:54
!h5dump data.h5
# Sun, 05 Feb 2017 17:33:21
h5f = h5py.File('data.h5', 'a')
# Sun, 05 Feb 2017 17:33:27
h = h5f.create_dataset('dataset_1', data=a)
# Sun, 05 Feb 2017 17:33:33
h = h5f.create_dataset('dataset_2', data=a)
# Sun, 05 Feb 2017 17:33:39
h.shape
#[Out]# (2, 3, 4)
# Sun, 05 Feb 2017 17:34:15
h.resize(1,2,12)
# Sun, 05 Feb 2017 17:34:23
h.resize(1,2,2)
# Sun, 05 Feb 2017 17:34:26
h.resize(1,2)
# Sun, 05 Feb 2017 17:42:16
import numpy as np
# Sun, 05 Feb 2017 17:42:17
from pandas importHDFStore,DataFrame# create (or open) an hdf5 file and opens in append mode
# Sun, 05 Feb 2017 17:42:18
hdf =HDFStore('storage.h5')
# Sun, 05 Feb 2017 17:42:29
from pandas importHDFStore,DataFrame
# Sun, 05 Feb 2017 17:42:34
from pandas importHDFStore, DataFrame
# Sun, 05 Feb 2017 17:42:42
from pandas import HDFStore,DataFrame
# Sun, 05 Feb 2017 17:43:13
hdf =HDFStore('storage.h5')
# Sun, 05 Feb 2017 17:43:15
ls
# Sun, 05 Feb 2017 17:48:06
df =DataFrame(np.random.rand(5,3), columns=('A','B','C'))
# Sun, 05 Feb 2017 17:48:08
df
#[Out]#           A         B         C
#[Out]# 0  0.333166  0.092269  0.819371
#[Out]# 1  0.233245  0.649195  0.718987
#[Out]# 2  0.331667  0.237529  0.781798
#[Out]# 3  0.324188  0.452287  0.811880
#[Out]# 4  0.239229  0.153740  0.373840
# Sun, 05 Feb 2017 17:48:14
df =DataFrame(np.random.rand(5,3))
# Sun, 05 Feb 2017 17:48:16
df
#[Out]#           0         1         2
#[Out]# 0  0.612967  0.625579  0.176081
#[Out]# 1  0.795840  0.190902  0.055109
#[Out]# 2  0.307932  0.688888  0.057545
#[Out]# 3  0.823256  0.284031  0.398731
#[Out]# 4  0.449865  0.412283  0.823011
# Sun, 05 Feb 2017 17:48:25
df =DataFrame(np.random.rand(5,3,1))
# Sun, 05 Feb 2017 18:17:02
dset = f.create_dataset("unlimited", (10, 10), maxshape=(None, 10))
# Sun, 05 Feb 2017 18:17:04
f
#[Out]# <Closed HDF5 file>
# Sun, 05 Feb 2017 18:17:38
dset = hf5.create_dataset("unlimited", (10, 10, 10), maxshape=(None, 10.10))
# Sun, 05 Feb 2017 18:17:58
hdf 
#[Out]# <class 'pandas.io.pytables.HDFStore'>
#[Out]# File path: storage.h5
#[Out]# Empty
# Sun, 05 Feb 2017 18:21:49
np.arange(10)
#[Out]# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# Sun, 05 Feb 2017 18:25:16
f = File('foo.h5', 'w')
# Sun, 05 Feb 2017 18:25:22
f['data'] = np.ones((4, 3, 2), 'f')
# Sun, 05 Feb 2017 18:25:25
f
#[Out]# <Closed HDF5 file>
# Sun, 05 Feb 2017 18:34:02
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 18:35:48
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 18:36:23
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 18:38:28
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 18:38:47
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 18:39:47
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 20:20:37
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 20:21:15
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 20:21:23
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 20:21:36
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 20:24:28
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 20:24:33
f
#[Out]# <Closed HDF5 file>
# Sun, 05 Feb 2017 20:24:42
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 20:25:55
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 20:26:13
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 20:26:34
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 20:27:28
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 20:28:04
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 20:30:01
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 20:30:57
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 20:32:37
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 20:33:14
run dataExtractionAutomator.py
# Sun, 05 Feb 2017 20:35:04
# This file automatically finds all the .ptw files in a given directory and
# Sun, 05 Feb 2017 20:35:05
# converts them to a numpy array for reseach purposes. It bypasses the need of
# Sun, 05 Feb 2017 20:35:05
# a proprietary software like ALTAIR to extract and read the data.
# Sun, 05 Feb 2017 20:35:05
'''
Before running this file perform 'pip install pyradi'
Please run the test file before running this file and follow the below
instructions to remove the error in the pyradi package, if any.

Comment out the line 516 in the file 'ryptw.py'
Header.h_Framatone = ord(headerinfo[3076:3077])
This ensures smooth running as required for this program.
'''
#[Out]# "\nBefore running this file perform 'pip install pyradi'\nPlease run the test file before running this file and follow the below\ninstructions to remove the error in the pyradi package, if any.\n\nComment out the line 516 in the file 'ryptw.py'\nHeader.h_Framatone = ord(headerinfo[3076:3077])\nThis ensures smooth running as required for this program.\n"
# Sun, 05 Feb 2017 20:35:05
# from IPython.display import display
# Sun, 05 Feb 2017 20:35:06
# from IPython.display import Image
# Sun, 05 Feb 2017 20:35:06
# from IPython.display import HTML
# Sun, 05 Feb 2017 20:35:06
#make pngs at 150 dpi
# Sun, 05 Feb 2017 20:35:06
'''
import matplotlib as mpl
mpl.rc("savefig", dpi=75)
mpl.rc('figure', figsize=(10,8))
'''
#[Out]# '\nimport matplotlib as mpl\nmpl.rc("savefig", dpi=75)\nmpl.rc(\'figure\', figsize=(10,8))\n'
# Sun, 05 Feb 2017 20:35:06
import numpy as np
# Sun, 05 Feb 2017 20:35:06
import numpy as np
# Sun, 05 Feb 2017 20:35:06
from pandas import HDFStore,DataFrame # create (or open) an hdf5 file and opens in append mode
# Sun, 05 Feb 2017 20:35:06
import os
# Sun, 05 Feb 2017 20:35:07
import pyradi.ryptw as ryptw
# Sun, 05 Feb 2017 20:35:07
import pyradi.ryplot as ryplot
# Sun, 05 Feb 2017 20:35:07
import pyradi.ryfiles as ryfiles
# Sun, 05 Feb 2017 20:35:08
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
            
# Sun, 05 Feb 2017 20:35:08
def ensure_dir(f):
        d = os.path.dirname(f)
        if not os.path.exists(d):
                os.makedirs(d)
                return 0
        return 1

# Sun, 05 Feb 2017 20:35:09
def saveHDF5Array(data, filename, iter, f=None):
        global file_Location_npy
        filename = file_Location_npy[iter] + '\\' + filename
        if not ensure_dir(filename):
                f = h5py.File(filename, 'a')
                dataset = f.create_dataset("data", data = data)
                return f
    
# Sun, 05 Feb 2017 20:35:09
    # dataset = f.create_dataset("data", data = data)
# Sun, 05 Feb 2017 20:35:09
    # np.save(filename,data)
# Sun, 05 Feb 2017 20:35:09
    pass
# Sun, 05 Feb 2017 20:35:10
def savePandasHDF5Array(data, filename, iter, f=None):
        global file_Location_npy
        filename = file_Location_npy[iter] + '\\' + filename
        if not ensure_dir(filename):
                # hdf = HDFStore('storage.h5')
                hdf = HDFStore(filename)
                # f = h5py.File(filename, 'a')
                # dataset = f.create_dataset("data", data = data)
                return f
    
# Sun, 05 Feb 2017 20:35:10
    df =DataFrame(np.random.rand(5,3), columns=('A','B','C'))# put the dataset in the storage
# Sun, 05 Feb 2017 20:35:10
    hdf.put('d1', df, format='table', data_columns=True)
# Sun, 05 Feb 2017 20:35:11
    # dataset = f.create_dataset("data", data = data)
# Sun, 05 Feb 2017 20:35:11
    # np.save(filename,data)
# Sun, 05 Feb 2017 20:35:11
    pass
# Sun, 05 Feb 2017 20:35:11
def saveNPArray(data, filename, iter):
        global file_Location_npy
        filename = file_Location_npy[iter] + '\\' + filename
        ensure_dir(filename)
        np.save(filename,data)
        pass

# Sun, 05 Feb 2017 20:35:12
def saveJPGPics(frames, filename, iter):
        global file_Location_npy
        filename = file_Location_jpg[iter] + '\\' + filename + '\\frame_'
        ensure_dir(filename)
        global ext
        # filename = file_Location_jpg[iter] + 'frame'
        i=0
        for frame in frames:
                i+=1
                ryfiles.rawFrameToImageFile(frame, filename+str(i)+ext)
            # np.save(filename,frame)
            pass
    
# Sun, 05 Feb 2017 20:35:13
def autoPTW2NP(ptwfilepath, ptwfilename, iter):
        # ptwfile  = './PyradiSampleLWIR.ptw'
        # outfilename = 'PyradiSampleLWIR.txt'
        header = ryptw.readPTWHeader(ptwfilepath)
        rows = header.h_Rows
        cols = header.h_Cols
        # ryptw.showHeader(header) # Suppressed Output
        # numFrames = header.h_lastframe # Total frames in the video
        numFrames = 100 # Testing time
        framesToLoad = range(1, numFrames+1, 1)
        frames = len(framesToLoad)
        data, fheaders  = ryptw.getPTWFrame (header, framesToLoad[0])
        #f = saveHDF5Array(frame, ptwfilename, iter)
        for frame in framesToLoad[1:]:
                f, fheaders = (ryptw.getPTWFrame (header, frame))
                data = np.concatenate((data, f))        print frame        # saveNPArray(frame, ptwfilename, iter)        # saveJPGPics(frame, ptwfilename, iter)    print data.shape
        
# Sun, 05 Feb 2017 20:35:13
    img = data.reshape(frames, rows ,cols)
# Sun, 05 Feb 2017 20:35:13
    print(img.shape)
# Sun, 05 Feb 2017 20:35:14
    # saveNPArray(img, ptwfilename, iter)
# Sun, 05 Feb 2017 20:35:14
    saveJPGPics(img, ptwfilename, iter)
# Sun, 05 Feb 2017 20:35:14
    return data
# Sun, 05 Feb 2017 20:35:14
excludeFiles = '1_0.5ws_4wfr_18lpm.ptw'
# Sun, 05 Feb 2017 20:35:14
# saveFolderLocation = './Extracted Data/'
# Sun, 05 Feb 2017 20:35:14
# Save format for Image
# Sun, 05 Feb 2017 20:35:14
ext = '.png'
# Sun, 05 Feb 2017 20:35:15
# Root Location for the original data
# Sun, 05 Feb 2017 20:35:15
dataLocation = 'F:\\Vishruit\\DATA'
# Sun, 05 Feb 2017 20:35:15
# Home location for saving the format file
# Sun, 05 Feb 2017 20:35:15
data_npyLocation = dataLocation + '_npy'
# Sun, 05 Feb 2017 20:35:15
data_jpgLocation = dataLocation + '_jpg'
# Sun, 05 Feb 2017 20:35:15
# Actual filepaths and filenames list
# Sun, 05 Feb 2017 20:35:15
[file_paths, file_names, file_locs] = getPTWFilePaths(dataLocation, excludeFiles)
# Sun, 05 Feb 2017 20:35:15
# Creating filepaths for desired file format
# Sun, 05 Feb 2017 20:35:16
# Also creating fileLocation paths
# Sun, 05 Feb 2017 20:35:16
file_paths_npy = []
# Sun, 05 Feb 2017 20:35:16
file_paths_jpg = []
# Sun, 05 Feb 2017 20:35:16
file_Location_npy = []
# Sun, 05 Feb 2017 20:35:16
file_Location_jpg = []
# Sun, 05 Feb 2017 20:35:17
for file_path in file_paths:
        file_paths_npy.append( data_npyLocation + file_path[len(dataLocation):len(file_path)] )
        file_paths_jpg.append( data_jpgLocation + file_path[len(dataLocation):len(file_path)] )
    
# Sun, 05 Feb 2017 20:35:17
# Save folder locations
# Sun, 05 Feb 2017 20:35:17
for file_loc in file_locs:
        file_Location_npy.append( data_npyLocation + file_loc[len(dataLocation):len(file_loc)] )
        file_Location_jpg.append( data_jpgLocation + file_loc[len(dataLocation):len(file_loc)] )
    
# Sun, 05 Feb 2017 20:35:17
print file_paths[1]
# Sun, 05 Feb 2017 20:35:17
for iter in range(len(file_paths) - 202): # len(file_paths)
        autoPTW2NP(file_paths[iter], file_names[iter], iter)
    
# Sun, 05 Feb 2017 20:35:36
'''
filename = "/my/directory/filename.txt"
dir = os.path.dirname(filename)
'''
#[Out]# '\nfilename = "/my/directory/filename.txt"\ndir = os.path.dirname(filename)\n'
# Sun, 05 Feb 2017 20:35:36
# from joblib import Parallel, delayed
# Sun, 05 Feb 2017 20:35:36
# import multiprocessing
# Sun, 05 Feb 2017 20:35:37
# num_cores = multiprocessing.cpu_count()
# Sun, 05 Feb 2017 20:35:37
# Parallel(n_jobs=num_cores)(delayed(autoPTW2NP)(file_paths[iter], file_names[iter]) for iter in range(10))
# Sun, 05 Feb 2017 20:36:27
run dataExtractionAutomator.py
