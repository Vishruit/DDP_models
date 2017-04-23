import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils.io_utils import HDF5Matrix
import h5py, numpy as np, sys

# train_file_name,  dataset_keyword = '../data_small_100.h5', 'data_small'
# train_file_name,  dataset_keyword = '/home/prabakaran/Vishruit/DDP/DATA_small_hdf/data_small_100.h5', 'data_small'
train_file_name,  dataset_keyword = 'Exp_17.ptw.h5', 'data_float32'
images = HDF5Matrix(train_file_name,  dataset_keyword)

def plot_histos():
    global train_file_name,  dataset_keyword
    train_set_data = HDF5Matrix(train_file_name, dataset_keyword)
    hist, bin_edges = np.histogram(train_set_data[0:202], normed =False, bins=100)
    print(hist, bin_edges)
    # plt.hist(train_set_data[0:202].reshape((202*100*256*320,1)), bins='auto')  # plt.hist passes it's arguments to np.histogram
    plt.hist(hist, bins=bin_edges)  # plt.hist passes it's arguments to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()


def demo1(images=None):
    # Otsu's demo
    from skimage import data
    from skimage.filters import threshold_otsu, threshold_local

    if images is None:
        images = data.page()

    fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
    image = images
    print(image.shape)
    plt.imshow(image)
    sys.exit()
    # for image in images:
    global_thresh = threshold_otsu(image)
    binary_global = image > global_thresh

    block_size = 35
    adaptive_thresh = threshold_local(image, block_size, offset=10)
    binary_adaptive = image > adaptive_thresh

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

def plot_image_simple():
    import matplotlib.pyplot as pyp
    x = [0, 2, 4, 6, 8]
    y = [0, 3, 3, 7, 0]
    pyp.plot(x, y)
    pyp.savefig("MyFirstPlot.png")
    pyp.show()

def plot_image():
    import numpy
    import pylab

    t = numpy.arange(0.0, 1.0+0.01, 0.01)
    s = numpy.cos(numpy.pi*4*t)
    pylab.plot(t, s)

    pylab.xlabel('time (s)')
    pylab.ylabel('cos(4t)')
    pylab.title('Simple cosine')
    pylab.grid(True)
    pylab.savefig('simple_cosine')

    pylab.show()

def demo_animation():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()

    def f(x, y):
        return np.sin(x) + np.cos(y)

    x = np.linspace(0, 2 * np.pi, 120)
    y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

    im = plt.imshow(f(x, y), animated=True)

    def updatefig(*args):
        global x, y
        x += np.pi / 15.
        y += np.pi / 20.
        im.set_array(f(x, y))
        return im,

    ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()

def plot_animation_my():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation\
    # from keras.utils.io_utils import HDF5Matrix
    import h5py, numpy as np, sys

    fig = plt.figure()

    train_file_name,  dataset_keyword = 'Exp_17.ptw.h5', 'data_float32'
    images = HDF5Matrix(train_file_name,  dataset_keyword)

    def f(i):
        return images[i]

    i=1

    im = pylab.imshow(f(i), animated=True)

    def updatefig(*args):
        global i
        i = i + 1
        im.set_array(f(i))
        return im,

    ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)

    # ani.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()

def plot_animation_my_normalized(option =1):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation\
    # from keras.utils.io_utils import HDF5Matrix
    import h5py, numpy as np, sys

    fig = plt.figure()

    train_file_name,  dataset_keyword = 'Exp_17.ptw.h5', 'data_float32'
    images = HDF5Matrix(train_file_name,  dataset_keyword)
    maxVal_im = np.max(images)
    minVal_im = np.min(images)
    meanVal_im = np.mean(images)
    stdVal_im = np.mean(images)

    def f(i, option):
        if option == 1:
            maxVal = np.max(images[i])
            image = images[i] / maxVal
            return image
        elif option == 2:
            maxVal = np.max(images[i])
            minVal = np.min(images[i])
            image = (images[i] - minVal) / (maxVal - minVal)
            return image
        elif option == 3:
            maxVal = maxVal_im
            minVal = 0 # np.min(images[i])
            image = (images[i] - minVal) / (maxVal - minVal)
            return image
        elif option == 4:
            maxVal = maxVal_im
            minVal = minVal_im
            image = (images[i] - minVal) / (maxVal - minVal)
            return image
        elif option == 5:
            meanVal = np.mean(images[i])
            stdVal = np.std(images[i])
            image = (images[i] - meanVal) / (stdVal)
            return image
        elif option == 6:
            meanVal = meanVal_im
            stdVal = stdVal_im
            image = (images[i] - meanVal) / (stdVal)
            return image
        elif option == 0:
            return images[i]

    i=1
    option =1

    im = pylab.imshow(f(i, option), animated=True)

    def updatefig(*args):
        global i, option
        i = i + 1
        im.set_array(f(i, option))
        return im,

    ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)

    # ani.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()


print(images[200].shape)

demo1()
plot_animation_my()
# demo1(images[200])
# demo2()
