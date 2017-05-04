import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

video_index = [1,5,10,15,20,25,30]
frame_index = [1,5,10,25,40,50,60,75,90,99]
# frame_index = [99,90,80,70,60,50,40,30,20,10]
#decoded_imgs = model.predict(x_test[video_index], batch_size=batch_size)
# plt.figure(figsize=(20, 4))
fig, axi = plt.subplots(2, 10, figsize=(20,4))
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
    #plt.close(fig)
