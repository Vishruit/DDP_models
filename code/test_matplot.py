import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


from matplotlib.cbook import get_sample_data
img = np.load(get_sample_data('axes_grid/bivariate_normal.npy'))

global batch_size, visualization_filepath
video_index = [1,5,10,15,20,25,30]
frame_index = [1,5,10,25,40,50,60,75,90,99]
# decoded_imgs = model.predict(x_test[video_index], batch_size=batch_size)

fig = Figure()
canvas = FigureCanvas(fig)

for (video, vid_it) in zip(video_index, range(len(video_index))):
    # plt.figure(figsize=(20, 4))
    for i in range(len(frame_index)):
        print(video, vid_it, i, len(frame_index))
        # ax = plt.subplot(2, len(frame_index), i + 1)
        ax1 = fig.add_subplot(2,len(frame_index),i+1)
        ax2 = fig.add_subplot(2,len(frame_index),i+len(frame_index)+1)

        ax1.imshow(img, cmap='gray')
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax2.imshow(img, cmap='gray')
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        # print ('this passes the test')
        #plt.imshow(decoded_imgs[vid_it].reshape(frames, 256, 320)[frame_index[i],...])
        # print(ax1, ax2)
    canvas.print_figure( 'test_reconstruction_vid_'+str(video)+'_Epoch_'+'.png' )
    print(fig, ax1)
    # plt.close('all')
    # plt.savefig( visualization_filepath+ 'reconstruction_vid_'+str(video)+'_Epoch_'+str(epoch)+'.png' )


def plot1():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    video_index = [1,5,10,15,20,25,30]
    frame_index = [1,5,10,25,40,50,60,75,90,99]



    # Reference counted: Helps with segmentation fault
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure()
    canvas = FigureCanvas(fig)

    # fig, axi = plt.subplots(2, 10, figsize=(20,4))
    # ax = plt.subplot(2, len(frame_index), i + len(frame_index) + 1)
    for (video, vid_it) in zip(video_index, range(len(video_index))):
        for i in range(len(frame_index)):
            print(video, vid_it, i, len(frame_index))
            ax1, ax2 = axi[:,i]
            print ('this passes the test')
            #plt.imshow(decoded_imgs[vid_it].reshape(frames, 256, 320)[frame_index[i],...])
            # plt.gray().
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
        fig.savefig( 'test_reconstruction_vid_'+str(video)+'_Epoch_'+'.png' )
        # plt.gcf().clear()
        # fig.cla()
        # plt.close('all')
        # plt.close(fig)
