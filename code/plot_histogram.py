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
    import pylab

    fig = plt.figure()
    ax1=fig.add_subplot(1,2,1)
    ax2=fig.add_subplot(1,2,2)
    # ims=[]

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

    im = ax1.imshow(f(i, option), animated=True)
    im1 = ax2.imshow(f(i, option+1), animated=True)

    def updatefig(*args):
        global i, option
        i = i + 1
        im.set_array(f(i, option))
        im1.set_array(f(i, option+1))
        return im,im1,

    ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)

    # ani.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()

def anim_subplot():
    """
    =================
    Animated subplots
    =================

    This example uses subclassing, but there is no reason that the proper function
    couldn't be set up and then use FuncAnimation. The code is long, but not
    really complex. The length is due solely to the fact that there are a total of
    9 lines that need to be changed for the animation as well as 3 subplots that
    need initial set up.

    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.animation as animation
    from keras.utils.io_utils import HDF5Matrix
    import h5py, sys, pylab


    class SubplotAnimation(animation.TimedAnimation):
        def __init__(self):
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 4)

            self.t = np.linspace(0, 80, 400)
            self.x = np.cos(2 * np.pi * self.t / 10.)
            self.y = np.sin(2 * np.pi * self.t / 10.)
            self.z = 10 * self.t

            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            self.line1 = Line2D([], [], color='black')
            self.line1a = Line2D([], [], color='red', linewidth=2)
            self.line1e = Line2D(
                [], [], color='red', marker='o', markeredgecolor='r')
            ax1.add_line(self.line1)
            ax1.add_line(self.line1a)
            ax1.add_line(self.line1e)
            ax1.set_xlim(-1, 1)
            ax1.set_ylim(-2, 2)
            ax1.set_aspect('equal', 'datalim')

            ax2.set_xlabel('y')
            ax2.set_ylabel('z')
            self.line2 = Line2D([], [], color='black')
            self.line2a = Line2D([], [], color='red', linewidth=2)
            self.line2e = Line2D(
                [], [], color='red', marker='o', markeredgecolor='r')
            ax2.add_line(self.line2)
            ax2.add_line(self.line2a)
            ax2.add_line(self.line2e)
            ax2.set_xlim(-1, 1)
            ax2.set_ylim(0, 800)

            ax3.set_xlabel('x')
            ax3.set_ylabel('z')
            self.line3 = Line2D([], [], color='black')
            self.line3a = Line2D([], [], color='red', linewidth=2)
            self.line3e = Line2D(
                [], [], color='red', marker='o', markeredgecolor='r')
            ax3.add_line(self.line3)
            ax3.add_line(self.line3a)
            ax3.add_line(self.line3e)
            ax3.set_xlim(-1, 1)
            ax3.set_ylim(0, 800)

            animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)

        def _draw_frame(self, framedata):
            i = framedata
            head = i - 1
            head_slice = (self.t > self.t[i] - 1.0) & (self.t < self.t[i])

            self.line1.set_data(self.x[:i], self.y[:i])
            self.line1a.set_data(self.x[head_slice], self.y[head_slice])
            self.line1e.set_data(self.x[head], self.y[head])

            self.line2.set_data(self.y[:i], self.z[:i])
            self.line2a.set_data(self.y[head_slice], self.z[head_slice])
            self.line2e.set_data(self.y[head], self.z[head])

            self.line3.set_data(self.x[:i], self.z[:i])
            self.line3a.set_data(self.x[head_slice], self.z[head_slice])
            self.line3e.set_data(self.x[head], self.z[head])

            self._drawn_artists = [self.line1, self.line1a, self.line1e,
                                   self.line2, self.line2a, self.line2e,
                                   self.line3, self.line3a, self.line3e]

        def new_frame_seq(self):
            return iter(range(self.t.size))

        def _init_draw(self):
            lines = [self.line1, self.line1a, self.line1e,
                     self.line2, self.line2a, self.line2e,
                     self.line3, self.line3a, self.line3e]
            for l in lines:
                l.set_data([], [])

    ani = SubplotAnimation()
    # ani.save('test_sub.mp4')
    plt.show()


print(images[200].shape)

demo1()
plot_animation_my()
# demo1(images[200])
# demo2()
