import caffe
import os
import h5py
import numpy as np
from matplotlib import pyplot as plt


# define a root path..
root_path = '/home/brojackfeely/Research/Video_CTU_withHuQiang/code/ICME_intra_CNN/CNN_network'

net_deploy = root_path +  '/intra_CNN_deploy.prototxt'
model      = root_path + '/model/intra_CNN_iter_60000.caffemodel'

test_filename  = root_path + '/../../../data/ICME_test_64/test_0.h5'

if( os.path.exists(net_deploy) and os.path.exists(model) ):
    print '! check file success: network_deploy and pretrain_model files found...'

# set GPU mode and select '/GPU:0' as the working device
caffe.set_mode_gpu()
caffe.set_device(0)

# load weight and bias parameters into intra_CNN network
net = caffe.Net(net_deploy,
                model,
                caffe.TEST)





# define a data input transformer i.e. preprocess TEST data
# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# transformer.set_raw_scale('data', 255)
# transformer.set_transpose('data', [3,2,1])
# set the data order as [N, C, W, H]...
#               i.e. N- num_of_samples
#                    C- num_of_channels
#                    W- width
#                    H- height



# read .hdf5 data into workspace...
with h5py.File(test_filename, 'r') as hf:
    test_data  = np.array(hf.get('data'))
    test_label = np.array(hf.get('label'))

num_test_samples = len(test_data)






# display layerName and blobSize...
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

print('!Finish LayerName and BlobSize display...')





# pycaffe test sample-by-sample...
counter = 0

for ii in range(num_test_samples):
    # data normalization into [0,1], format: 'uint8'
    sample_data  = test_data[ii]/255
    sample_data  = np.reshape(sample_data, [64, 64, 1, 1])
    sample_label = test_label[ii]

    # caffe test process...
    # check each sample of TEST data...
    # plt.figure(1)
    # plt.imshow(np.squeeze(sample_data))
    # plt.show()


    net.blobs['data'].data[...] = np.transpose(sample_data, [3,2,1,0])

    output = net.forward()
    sample_predict = net.blobs['fc7'].data[0]

    sample_prob = 1/(1+np.exp(-sample_predict))


    print sample_prob, sample_label

    counter = counter + 1

