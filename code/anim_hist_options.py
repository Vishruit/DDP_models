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


train_file_name,  dataset_keyword = 'Exp_17.ptw.h5', 'data_float32'
images = HDF5Matrix(train_file_name,  dataset_keyword)
# maxVal_im = np.max(images)
# minVal_im = np.min(images)
# meanVal_im = np.mean(images)
# stdVal_im = np.mean(images)
maxVal_im, minVal_im, meanVal_im, stdVal_im = 1,0,1,1


class SubplotAnimation(animation.TimedAnimation):
    def __init__(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 4)

        fig1 = plt.figure()
        ax_0 = fig1.add_subplot(3, 3, 1)
        ax_1 = fig1.add_subplot(3, 3, 2)
        ax_2 = fig1.add_subplot(3, 3, 3)
        ax_3 = fig1.add_subplot(3, 3, 4)
        ax_4 = fig1.add_subplot(3, 3, 5)
        ax_5 = fig1.add_subplot(3, 3, 6)
        ax_6 = fig1.add_subplot(3, 3, 7)

        self.t = np.linspace(0, 80, 400)
        self.x = np.cos(2 * np.pi * self.t / 10.)
        self.y = np.sin(2 * np.pi * self.t / 10.)
        self.z = 10 * self.t

        self.i =1
        print ('HI')

        self.im0 = ax_0.imshow(self.f(self.i, self.option(0)), animated=True)
        self.im1 = ax_1.imshow(self.f(self.i, self.option(1)), animated=True)
        self.im2 = ax_2.imshow(self.f(self.i, self.option(2)), animated=True)
        self.im3 = ax_3.imshow(self.f(self.i, self.option(3)), animated=True)
        self.im4 = ax_4.imshow(self.f(self.i, self.option(4)), animated=True)
        self.im5 = ax_5.imshow(self.f(self.i, self.option(5)), animated=True)
        self.im6 = ax_6.imshow(self.f(self.i, self.option(6)), animated=True)

        self.imx = [self.im0, self.im1, self.im2, self.im3,
                    self.im4, self.im5, self.im6]

        ax1, self.line1, self.line1a, self.line1e = self.set_ax(ax1, 'x', 'y')
        ax1.set_aspect('equal', 'datalim')
        ax2, self.line2, self.line2a, self.line2e = self.set_ax(ax2, 'y', 'z', ylim=(0,800))
        ax3, self.line3, self.line3a, self.line3e = self.set_ax(ax3, 'x', 'z', ylim=(0,800))

        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)
        print ('HI')
        ani = animation.FuncAnimation(fig1, self.updatefig, interval=50, blit=True)

    def f(self, i, option):
        # return 1
        global maxVal_im, minVal_im, meanVal_im, stdVal_im, images
        if option == 1:
            maxVal = np.max(images[i])
            image = images[i] / maxVal
            print('Hey stalker!')
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

    def option(self, num):
        return num

    def set_ax(self, ax1, xlabel, ylabel, ylim=(-2,2)):
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        line1 = Line2D([], [], color='black')
        line1a = Line2D([], [], color='red', linewidth=2)
        line1e = Line2D(
            [], [], color='red', marker='o', markeredgecolor='r')
        ax1.add_line(line1)
        ax1.add_line(line1a)
        ax1.add_line(line1e)
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(ylim)

        return ax1, line1, line1a, line1e

    def set_ax_im(self, ax_1, im1):
        ax_1.add_image(im1)
        return ax1

    def updatefig(*args):
        self.i = self.i + 1
        # for (im, count) in zip(self.imx, range(len(self.imx))):
        #     self.im.set_array(self.f(self.i, self.option(count)))
        self.im0.set_array(self.f(self.i, self.option(0)))
        self.im1.set_array(self.f(self.i, self.option(1)))
        self.im2.set_array(self.f(self.i, self.option(2)))
        self.im3.set_array(self.f(self.i, self.option(3)))
        self.im4.set_array(self.f(self.i, self.option(4)))
        self.im5.set_array(self.f(self.i, self.option(5)))
        self.im6.set_array(self.f(self.i, self.option(6)))

        return self.im0,self.im1,self.im2,self.im3,self.im4,self.im5,self.im6,

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

    # def _init_draw(self):
    #     lines = [self.line1, self.line1a, self.line1e,
    #              self.line2, self.line2a, self.line2e,
    #              self.line3, self.line3a, self.line3e]
    #
    #     ims = [self.im0, self.im1, self.im2, self.im3,
    #            self.im4, self.im5, self.im6]
    #
    #     for l in lines:
    #         l.set_data([], [])
    #     pass

ani = SubplotAnimation()
# ani.save('test_sub.mp4')
plt.show()
