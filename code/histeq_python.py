import numpy as np
import h5py, numpy as np, sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from keras.utils.io_utils import HDF5Matrix
from anim_final import auto_plot, option

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf

def read_data():
    train_file_name, dataset_keyword = '../Exp_17.ptw.h5', 'data_float32'
    images = HDF5Matrix(train_file_name,  dataset_keyword)
    return images

def f(i, option):
    global data, data_equalized
    if option == 0:
        return data[i]
    elif option ==1:
        return data_equalized[i]

def make_anim(options, filename):
    fig = plt.figure()
    ax = []
    for j in range(options):
        row, col = auto_plot(options)
        ax.append(fig.add_subplot(row, col, j+1))

    imx = []
    for j in range(options):
        imx.append(ax[j].imshow(f(i, option(j) ), animated=True)) # Colored image
        fig.colorbar(imx[j],ax=ax[j])

    def updatefig(i):
        # Doesn't show the plot
        global options
        for j in range(options):
            imx[j].set_array( f(i, option(j)) )
        return imx
    ani = animation.FuncAnimation(fig, updatefig, interval=10, blit=True, frames=2000)
    ani.save('./animations/'+filename, writer='imagemagick', fps=30)

if __name__ == '__main__':
    # generate some test data with shape 1000, 1, 96, 96
    images = read_data()
    data = images.data
    # data = np.random.rand(1000, 1, 96, 96)
    options = 2
    filename = 'numpy_hist_vid.gif'

    # loop over them
    print('hi1')
    data_equalized = np.zeros(data.shape)
    for i in range(data.shape[0]):
        image = data[i, :, :]
        data_equalized[i, :, :] = image_histogram_equalization(image)[0]
    print('hi2')
    make_anim(options, filename)
