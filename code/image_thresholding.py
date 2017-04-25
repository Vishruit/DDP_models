import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from skimage import data
from skimage.filters import try_all_threshold
from skimage.filters import threshold_mean
from skimage.filters import threshold_minimum


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf

def plot_threshold(image):
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
    return image

def thresh_im(image):
    # apply mean threshold
    thresh = threshold_mean(image)
    binary = image > thresh
    # Bimodal histogram
    thresh_min = threshold_minimum(image)
    binary_min = image > thresh_min
    # Otsu's THreshold
    thresh = threshold_otsu(image)
    binary = image > thresh
    print('Hi')
    image = binary_min
    return image

if __name__ == '__main__':

    # generate some test data with shape 1000, 1, 96, 96
    data = np.random.rand(1000, 1, 96, 96)

    # loop over them
    data_equalized = np.zeros(data.shape)
    for i in range(data.shape[0]):
        image = data[i, 0, :, :]
        data_equalized[i, 0, :, :] = image_histogram_equalization(image)[0]

# http://stackoverflow.com/questions/28518684/histogram-equalization-of-grayscale-images-with-numpy
