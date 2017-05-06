
def plot_video_plot4(epoch,x_test):
    global batch_size, visualization_filepath
    video_index = [1,5,10,15,20,25,30]
    frame_index = [1,5,10,25,40,50,60,75,90,99]
    decoded_imgs = model.predict(x_test[video_index], batch_size=batch_size/2)
    print (x_test.shape, decoded_imgs.shape)
    fig, ax = plt.subplots(2, len(frame_index), figsize = (20,4))
    line, = ax.plot(np.random.randn(100))

    fig = Figure()
    canvas = FigureCanvas(fig)
    decoded_imgs = model.predict(x_test[video_index], batch_size=batch_size)
    for (video, vid_it) in zip(video_index, range(len(video_index))):
        for i in range(len(frame_index)):
            print(video, vid_it, i, len(frame_index))
            ax1 = fig.add_subplot(2,len(frame_index),i+1)
            ax2 = fig.add_subplot(2,len(frame_index),i+len(frame_index)+1)
            ax1.imshow(x_test[video].reshape(frames, 256, 320)[frame_index[i],...], cmap='gray')
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            print ('this passes the test')
            ax2.imshow(decoded_imgs[vid_it].reshape(frames, 256, 320)[frame_index[i],...], cmap='gray')
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
            # plt.clf()
            canvas.print_figure( visualization_filepath+'reconstruction_vid_'+str(video)+'_Epoch_'+'.png' )


def plot_video_plot(epoch,x_test):
    global batch_size, visualization_filepath
    video_index = [1,5,10,15,20,25,30]
    frame_index = [1,5,10,25,40,50,60,75,90,99]
    decoded_imgs = model.predict(x_test[video_index], batch_size=batch_size/2)
    print (x_test.shape, decoded_imgs.shape)
    fig = Figure()
    canvas = FigureCanvas(fig)
    decoded_imgs = model.predict(x_test[video_index], batch_size=batch_size)

    for (video, vid_it) in zip(video_index, range(len(video_index))):
        for i in range(len(frame_index)):
            print(video, vid_it, i, len(frame_index))
            ax1 = fig.add_subplot(2,len(frame_index),i+1)
            ax2 = fig.add_subplot(2,len(frame_index),i+len(frame_index)+1)
            #ax2 = fig.add_subplot(2,len(frame_index),i+len(frame_index)+1)
            ax1.imshow(x_test[video][frame_index[i],:,:], cmap='gray')
            plt.gray()
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            print ('this passes the test')
            ax2.imshow(decoded_imgs[vid_it][frame_index[i],:,:], cmap='gray')
            plt.gray()
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
            # plt.clf()
            canvas.print_figure( visualization_filepath+'reconstruction_vid_'+str(video)+'_Epoch_'+'.png' )

def plot_video_plot3(epoch,x_test):
    global batch_size, visualization_filepath
    video_index = [1,5,10,15,20,25,30]
    frame_index = [1,5,10,25,40,50,60,75,90,99]
    decoded_imgs = model.predict(x_test[video_index], batch_size=batch_size)

    fig = Figure()
    canvas = FigureCanvas(fig)
    decoded_imgs = autoencoder.predict(x_test)
    # ax = plt.subplot(2, len(frame_index), 1)
    for (video, vid_it) in zip(video_index, range(len(video_index))):
        # plt.figure(figsize=(20, 4))
        for i in range(len(frame_index)):
            print(video, vid_it, i, len(frame_index))
            # ax = plt.subplot(2, len(frame_index), i + 1)
            ax1 = fig.add_subplot(2,len(frame_index),i+1)
            #ax2 = fig.add_subplot(2,len(frame_index),i+len(frame_index)+1)
            # ax1,ax2 = axi[:,i]
            ax1.imshow(x_test[video, frame_index[i],...])
            plt.gray()
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            print ('this passes the test')
            #plt.imshow(decoded_imgs[vid_it].reshape(frames, 256, 320)[frame_index[i],...])
            plt.gray()
            #ax2.get_xaxis().set_visible(False)
            #ax2.get_yaxis().set_visible(False)
            plt.clf()
        canvas.print_figure( visualization_filepath+ 'reconstruction_vid_'+str(video)+'_Epoch_'+str(epoch)+'.png' )
        plt.close()
        # plt.savefig( visualization_filepath+ 'reconstruction_vid_'+str(video)+'_Epoch_'+str(epoch)+'.png' )
    return

def plot_video_plot2(epoch,x_test):
    global batch_size, visualization_filepath
    video_index = [1,5,10,15,20,25,30]
    frame_index = [1,5,10,25,40,50,60,75,90,99]
    decoded_imgs = model.predict(x_test[video_index], batch_size=batch_size)
    plt.figure(figsize=(20, 4))
    # fig, axi = plt.subplots(2, len(frame_index), figsize=(20,4))
    # ax = plt.subplot(2, len(frame_index), 1)
    for (video, vid_it) in zip(video_index, range(len(video_index))):
        # plt.figure(figsize=(20, 4))
        for i in range(len(frame_index)):
            print(video, vid_it, i, len(frame_index))
            ax1 = plt.subplot(2, len(frame_index), i + 1)
            #ax1,ax2 = axi[:,i]
            plt.imshow(x_test[video].reshape(frames, 256, 320)[frame_index[i],...])
            #plt.gray()
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            # ax1 = axi[i+1]
            # ax2 = axi[i+len(frame_index)+1]
            ax2 = plt.subplot(2, len(frame_index), i + len(frame_index) + 1)
            #ax = plt.subplot(2, len(frame_index), 1)
            print ('this passes the test')
            #plt.imshow(decoded_imgs[vid_it].reshape(frames, 256, 320)[frame_index[i],...])
            #plt.gray()
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
        plt.savefig( visualization_filepath+ 'reconstruction_vid_'+str(video)+'_Epoch_'+str(epoch)+'.png' )
        # plt.gcf().clear()
        # plt.clf()
    return

def plot_video_plot1 (epoch,x_test):
    global batch_size, visualization_filepath
    video_index = [1,5,10,15,20,25,30]
    frame_index = [1,5,10,25,40,50,60,75,90,99]
    decoded_imgs = model.predict(x_test[video_index], batch_size=batch_size)
    for (video, vid_it) in zip(video_index, range(len(video_index))):
        plt.figure(figsize=(20, 4))
        for i in range(len(frame_index)):
            ax = plt.subplot(2, len(frame_index), i + 1)
            plt.imshow(x_test[video].reshape(frames, 256, 320)[frame_index[i],...])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, len(frame_index), i + len(frame_index) + 1)
            plt.imshow(decoded_imgs[vid_it].reshape(frames, 256, 320)[frame_index[i],...])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.savefig( visualization_filepath+ 'reconstruction_vid_'+str(video)+'_Epoch_'+str(epoch)+'.png' )
    return
