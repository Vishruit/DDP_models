import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

video_index = [1,5,10,15,20,25,30]
#frame_index = [1,5,10,25,40,50,60,75,90,99]
frame_index = [99,90,80,70,60,50,40,30,20,10]
#decoded_imgs = model.predict(x_test[video_index], batch_size=batch_size)
for (video, vid_it) in zip(video_index, range(len(video_index))):
    plt.figure(figsize=(20, 4))
    for i in range(len(frame_index)):
        '''
        print(video, vid_it, x_test[video].shape, i, len(frame_index))
        ax = plt.subplot(2, len(frame_index), i + 1)
        plt.imshow(x_test[video].reshape(frames, 256, 320)[frame_index[i],...])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        '''
        print(video, vid_it, i, len(frame_index))
        ax = plt.subplot(2, len(frame_index), i + len(frame_index) + 1)
        print ('this passes the test')
        #plt.imshow(decoded_imgs[vid_it].reshape(frames, 256, 320)[frame_index[i],...])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig( 'test_reconstruction_vid_'+str(video)+'_Epoch_'+'.png' )
        #plt.close()
    plt.gcf().clear()
    plt.close('all')
        #plt.close(fig)

