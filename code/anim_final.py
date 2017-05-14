import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pylab
import numpy as np
import h5py, numpy as np, sys
from keras.utils.io_utils import HDF5Matrix
from skimage.filters import threshold_mean
from image_thresholding import thresh_im
from skimage.filters import threshold_mean
from skimage.filters import threshold_minimum
from skimage.filters import threshold_otsu, threshold_local
from skimage.filters import threshold_adaptive


epsilon = 0.0001
def read_data():
    train_file_name,  dataset_keyword = '../Exp_17.ptw.h5', 'data_float32'
    images = HDF5Matrix(train_file_name,  dataset_keyword)
    # maxVal_im = np.max(images)
    # minVal_im = np.min(images)
    # meanVal_im = np.mean(images)
    # stdVal_im = np.mean(images)
    maxVal_im, minVal_im, meanVal_im, stdVal_im = 1.0,epsilon,1.0,1.0
    return images, maxVal_im, minVal_im, meanVal_im, stdVal_im

def auto_plot(options):
    row, col = 1,1
    flag = 1
    while options > row * col:
        row += (1 + flag)/2
        col += (1 - flag)/2
        flag *= -1
    return row, col

options = 2
filename = 'difference_images_vid.gif'
images, maxVal_im, minVal_im, meanVal_im, stdVal_im = read_data()

fig = plt.figure()
ax = []
for j in range(options):
    row, col = auto_plot(options)
    ax.append(fig.add_subplot(row, col, j+1))

def f(i, option):
    if option == 1:
        return images[i+1]-images[i] # Difference image
        # Custom '1' for debugging
        image = images[i]
        num_points = image.shape[0] * image.shape[1]
        a, b = np.min(image), np.max(image)
        im_flat = image.flatten()
        table =  np.zeros( (int(b-a) + 1, 2) )

        table[:,0] = np.array(range(int(a),int(b) + 1)) # Values
        for i in im_flat:
            table[int(i-a),1] += 1 # Count
        table[:,1] = table[:,1] / num_points # PDF
        prev_entry = 0
        # CDF
        for i in range(int(b-a)):
            temp = table[i,1]
            table[i,1] += prev_entry
            prev_entry = temp
        # Setting range
        table[:,1] *= 255
        # table[:,1] = np.round(table[:,1])
        for i in im_flat:
            i = table[int(i-a),1]
        image_histeq = im_flat.reshape(image.shape)
        return image_histeq -image
        #
        # image_histogram, bins = np.histogram(image.flatten(), 1, normed=True)
        # cdf = image_histogram.cumsum() # cumulative distribution function
        # cdf = 255 * cdf / cdf[-1] # normalize
        #
        # # use linear interpolation of cdf to find new pixel values
        # image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
        #
        # return image_equalized.reshape(image.shape)
        # # binary_image1 = threshold_adaptive(image, 15, 'mean')
        # binary_image2 = threshold_otsu(image) #, param='sigma')
        # print(binary_image2.shape)
        # return binary_image2
    elif option == 100:
        maxVal = np.max(images[i])
        image = images[i] / (maxVal + epsilon)
        return image
    elif option == 2:
        maxVal = np.max(images[i])
        minVal = np.min(images[i]) + epsilon
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
        image = (images[i] - meanVal) / (stdVal + epsilon)
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
    # imx.append(ax[j].imshow(f(i, option(j) ), cmap='gray', animated=True)) # Grayscale images
    imx.append(ax[j].imshow(f(i, option(j) ), animated=True)) # Colored image
    fig.colorbar(imx[j],ax=ax[j])

def updatefig_controlled(*args):
    # Generates the entire video and exits with an error message
    # Shows the plot
    global i, option, options
    i = i + 1
    for j in range(options):
        imx[j].set_array( f(i, option(j)) )
    return imx

def updatefig(i):
    # Doesn't show the plot
    global option, options
    for j in range(options):
        imx[j].set_array( f(i, option(j)) )
    return imx

ani = animation.FuncAnimation(fig, updatefig, interval=10, blit=True, frames=2000)
ani.save('./animations/'+filename, writer='imagemagick', fps=30)
# Uncomment in order to show the animation
# plt.show()
