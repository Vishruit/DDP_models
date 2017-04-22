import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils.io_utils import HDF5Matrix
import h5py, numpy as np

# train_file_name,  dataset_keyword = '../data_small_100.h5', 'data_small'
train_file_name,  dataset_keyword = '/home/prabakaran/Vishruit/DDP/DATA_small_hdf/data_small_100.h5', 'data_small'


def plot_histos():
    global train_file_name,  dataset_keyword
    train_set_data = HDF5Matrix(train_file_name, dataset_keyword)

    hist, bin_edges = np.histogram(train_set_data[0:202], normed =False, bins=100)

    print(hist, bin_edges)
    # plt.hist(train_set_data[0:202].reshape((202*100*256*320,1)), bins='auto')  # plt.hist passes it's arguments to np.histogram
    plt.hist(hist, bins=bin_edges)  # plt.hist passes it's arguments to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()


def demo1():
    # Otsu's demo
    from skimage import data
    from skimage.filters import threshold_otsu, threshold_local
    
    image = data.page()

    global_thresh = threshold_otsu(image)
    binary_global = image > global_thresh

    block_size = 35
    adaptive_thresh = threshold_local(image, block_size, offset=10)
    binary_adaptive = image > adaptive_thresh

    fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
    ax = axes.ravel()
    plt.gray()

    ax[0].imshow(image)
    ax[0].set_title('Original')

    ax[1].imshow(binary_global)
    ax[1].set_title('Global thresholding')

    ax[2].imshow(binary_adaptive)
    ax[2].set_title('Adaptive thresholding')

    for a in ax:
        a.axis('off')

    plt.show()

def demo2():
    # Another demo
    import matplotlib.pyplot as plt
    from skimage import data
    try:
        from skimage import filters
    except ImportError:
        from skimage import filter as filters
    from skimage import exposure

    camera = data.camera()
    val = filters.threshold_otsu(camera)

    hist, bins_center = exposure.histogram(camera)

    plt.figure(figsize=(9, 4))
    plt.subplot(131)
    plt.imshow(camera, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(camera < val, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.subplot(133)
    plt.plot(bins_center, hist, lw=2)
    plt.axvline(val, color='k', ls='--')

    plt.tight_layout()
    plt.show()

demo1()
demo2()
