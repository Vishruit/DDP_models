import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py, numpy as np, sys
import pylab
from keras.utils.io_utils import HDF5Matrix

train_file_name,  dataset_keyword = 'Exp_17.ptw.h5', 'data_float32'
images = HDF5Matrix(train_file_name,  dataset_keyword)
maxVal_im = np.max(images)
minVal_im = np.min(images)
meanVal_im = np.mean(images)
stdVal_im = np.mean(images)

options = 7

fig = plt.figure()
ax = []
for j in range(options):
    ax.append(fig.add_subplot(3, 3, j+1))
# ax_0 = fig.add_subplot(3, 3, 1)

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
        meanVal = meanVal_im
        stdVal = stdVal_im
        image = (images[i] - meanVal) / (stdVal)
        return image
    elif option == 0:
        return images[i]

def option(num):
    return num

i=1
imx = []
for j in range(options):
    imx.append(ax[j].imshow(f(i, option(j) ), cmap=plt.get_cmap('viridis'), animated=True))
    fig.colorbar(imx[j],ax=ax[j])

# im0 = ax_0.imshow(f(i, option(0) ), animated=True)


def updatefig(*args):
    global i, option
    i = i + 1
    for j in range(options):
        imx[j].set_array( f(i, option(j)) )
    # im0.set_array( f(i, option(0)) )
    return imx[0],imx[1],imx[2],imx[3],imx[4],imx[5],imx[6],

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)

# ani.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
