import numpy as np
import h5py, numpy as np, sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from keras.utils.io_utils import HDF5Matrix

from copy import deepcopy
# image1, image2 = 0

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

def option(num):
    return num

def do_hist_eq(image):
    image1 = deepcopy(image)
    num_points = image.shape[0] * image.shape[1]
    a, b = np.min(image), np.max(image)
    im_flat = image.flatten()
    table =  np.zeros( (int(b-a) + 1, 5) )
    print (b, a)

    table[:,0] = np.array(range(int(a),int(b) + 1)) # Values
    for j in im_flat:
        table[int(j-a),1] += 1 # Count
    table[:,2] = table[:,1] / num_points # PDF
    running_sum = 0
    # CDF
    for j in range(int(b-a)):
        # temp = table[i,2]
        table[j,3] = running_sum + table[j,2]
        running_sum = table[j,3]
    # Thresholding
    lower_level_threshold = 0.015 # 1.5%
    upper_level_threshold = 1 - 0.010 # 1.5%
    lower_level, upper_level = 0, 0
    for val, cdf in zip(table[:,0],table[:,3]):
        if cdf < lower_level_threshold:
            lower_level = val
        if cdf >= upper_level_threshold:
            upper_level = val
            break
    table[:int(lower_level-a), 3] = table[int(lower_level-a),3]
    table[int(upper_level-a):, 3] = table[int(upper_level-a),3]

    # Setting range
    # table[:,4] = table[:,3] * (upper_level - lower_level) + lower_level
    table[:,4] = table[:,3] * 255
    # table[:,1] = np.round(table[:,1])
    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            image[j,k] = table[int(image[j,k] - a), 4]
    # for i in im_flat:
        # i = table[int(i-a),4]
    # image_histeq = im_flat.reshape(image.shape)
    # return images[i] - image
    image2 = image
    return image

def auto_plot(options):
    row, col = 1,1
    flag = 1
    while options > row * col:
        row += (1 + flag)/2
        col += (1 - flag)/2
        flag *= -1
    return row, col

def f(i, option):
    global data, data_equalized
    if option == 0:
        image = data[i]
        a, b = np.min(image), np.max(image)
        image = (image - a) / (b - a + 0.0001)
        return image
    elif option ==1:
        return data_equalized[i]
    elif option == 2:
        # Custom '1' for debugging
        global table, image_histeq, image, image2
        image = images[i]
        global image1
        image1 = deepcopy(image)
        num_points = image.shape[0] * image.shape[1]
        a, b = np.min(image), np.max(image)
        im_flat = image.flatten()
        table =  np.zeros( (int(b-a) + 1, 5) )
        print (b, a)

        table[:,0] = np.array(range(int(a),int(b) + 1)) # Values
        for j in im_flat:
            table[int(j-a),1] += 1 # Count
        table[:,2] = table[:,1] / num_points # PDF
        running_sum = 0
        # CDF
        for j in range(int(b-a)):
            # temp = table[i,2]
            table[j,3] = running_sum + table[j,2]
            running_sum = table[j,3]
        # Thresholding
        lower_level_threshold = 0.015 # 1.5%
        upper_level_threshold = 1 - 0.010 # 1.5%
        lower_level, upper_level = 0, 0
        for val, cdf in zip(table[:,0],table[:,3]):
            if cdf < lower_level_threshold:
                lower_level = val
            if cdf >= upper_level_threshold:
                upper_level = val
                break
        table[:int(lower_level-a), 3] = table[int(lower_level-a),3]
        table[int(upper_level-a):, 3] = table[int(upper_level-a),3]

        # Setting range
        # table[:,4] = table[:,3] * (upper_level - lower_level) + lower_level
        table[:,4] = table[:,3] * 255
        # table[:,1] = np.round(table[:,1])
        for j in range(image.shape[0]):
            for k in range(image.shape[1]):
                image[j,k] = table[int(image[j,k] - a), 4]
        # for i in im_flat:
            # i = table[int(i-a),4]
        # image_histeq = im_flat.reshape(image.shape)
        # return images[i] - image
        image2 = image
        return image

def make_anim(options, filename, frames):
    fig = plt.figure()
    ax = []
    for j in range(options):
        row, col = auto_plot(options)
        ax.append(fig.add_subplot(row, col, j+1))

    imx = []
    for j in range(options):
        imx.append(ax[j].imshow(f(i, option(j) ), cmap='gray', animated=True)) # Colored image
        fig.colorbar(imx[j],ax=ax[j])

    def updatefig(i):
        # Doesn't show the plot
        global options
        for j in range(options):
            imx[j].set_array( f(i, option(j)) )
        return imx
    ani = animation.FuncAnimation(fig, updatefig, interval=10, blit=True, frames=frames)
    ani.save('./animations/'+filename, writer='imagemagick', fps=30)

if __name__ == '__main__':
    # generate some test data with shape 1000, 1, 96, 96
    images = read_data()
    data = images.data
    # data = np.random.rand(1000, 1, 96, 96)
    i=1
    options = 3
    frames = 100
    filename = 'numpy_hist_vid_difference0-1.gif'

    # loop over them
    print('hi1')
    data_equalized = np.zeros(data.shape)
    # for i in range(data.shape[0]):
    #     image = data[i, :, :]
    #     data_equalized[i, :, :] = image_histogram_equalization(image)[0]
    print('hi2')
    make_anim(options, filename, frames)
