import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py, numpy as np, sys
import pylab
from keras.utils.io_utils import HDF5Matrix
from skimage.filters import threshold_mean
from image_thresholding import thresh_im
from skimage.filters import threshold_mean
from skimage.filters import threshold_minimum
from skimage.filters import threshold_otsu, threshold_local
from skimage.filters import threshold_adaptive

def read_data():
    train_file_name,  dataset_keyword = 'Exp_17.ptw.h5', 'data_float32'
    images = HDF5Matrix(train_file_name,  dataset_keyword)
    # maxVal_im = np.max(images)
    # minVal_im = np.min(images)
    # meanVal_im = np.mean(images)
    # stdVal_im = np.mean(images)
    maxVal_im, minVal_im, meanVal_im, stdVal_im = 1,0,1,1
    return images, maxVal_im, minVal_im, meanVal_im, stdVal_im

images, maxVal_im, minVal_im, meanVal_im, stdVal_im = read_data()

options = 2

fig = plt.figure()
ax = []
for j in range(options):
    # ax.append(fig.add_subplot(3, 3, j+1))
    ax.append(fig.add_subplot(2, 1, j+1))
# ax_0 = fig.add_subplot(3, 3, 1)

def f(i, option):
    if option == 1:
        # Custom '1' for debugging
        image = images[i]
        image_histogram, bins = np.histogram(image.flatten(), 255, normed=True)
        cdf = image_histogram.cumsum() # cumulative distribution function
        cdf = 255 * cdf / cdf[-1] # normalize

        # use linear interpolation of cdf to find new pixel values
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

        return image_equalized.reshape(image.shape)
        # binary_image1 = threshold_adaptive(image, 15, 'mean')
        binary_image2 = threshold_otsu(image) #, param='sigma')
        print(binary_image2.shape)
        return binary_image2
    elif option == 1:
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
        minVal = 0
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
    elif option == 7:
        thresh = threshold_mean(images[i])
        thresh = threshold_minimum(image[i])
        thresh = threshold_otsu(image[i])
        binary = images[i] > thresh
        return binary
    elif option == 8:
        global_thresh = threshold_otsu(image)
        binary_global = image > global_thresh
        block_size = 35
        adaptive_thresh = threshold_local(image, block_size, offset=10)
        binary_adaptive = image > adaptive_thresh
        return binary_adaptive
    elif option == 9:
        # binary = threshold_adaptive(images[1], 1, 'gaussian')
        binary = threshold_local(images[1], 1, 'gaussian')
        return binary
    elif option == 0:
        return images[i]

def option(num):
    return num

i=1
imx = []
for j in range(options):
    # imx.append(ax[j].imshow(f(i, option(j) ), cmap=plt.get_cmap('viridis'), animated=True))
    imx.append(ax[j].imshow(f(i, option(j) ), cmap='gray', animated=True))
    # imx.append(ax[j].imshow(f(i, option(j) ), animated=True))
    fig.colorbar(imx[j],ax=ax[j])
# plt.gray()
# im0 = ax_0.imshow(f(i, option(0) ), animated=True)

def updatefig(*args):
    global i, option, options
    i = i + 1
    for j in range(options):
        imx[j].set_array( f(i, option(j)) )
    # im0.set_array( f(i, option(0)) )
    # return imx[0],imx[1],imx[2],imx[3],imx[4],imx[5],imx[6],
    # return imx[0:options] # the comma was the issue! Now fixed :)
    return imx # the comma was the issue! Now fixed :)

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
# ani.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()
