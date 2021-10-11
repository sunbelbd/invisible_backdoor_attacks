import sys
import os
import matplotlib
import PIL
import six
import numpy as np
import math
import time
import paddle
import paddle.fluid as fluid

matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot(filename, gen_data):
    pad_dim = 1
    paded = pad_dim + img_dim
    gen_data = gen_data.reshape(gen_data.shape[0], img_dim, img_dim)
    n = int(math.ceil(math.sqrt(gen_data.shape[0])))
    gen_data = (np.pad(
        gen_data, [[0, n * n - gen_data.shape[0]], [pad_dim, 0], [pad_dim, 0]],
        'constant').reshape((n, n, paded, paded)).transpose((0, 2, 1, 3))
                .reshape((n * paded, n * paded)))
    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imsave(filename, gen_data, cmap='Greys_r', vmin=-1, vmax=1)
    return fig

gf_dim = 64 # the number of basic channels of the generator's feature map. The number of all the channels of feature maps in the generator is a multiple of the number of basic channels
df_dim = 64 # the number of basic channels of the discriminator's feature map. The number of all the channels of feature maps in the discriminator is a multiple of the number of basic channels
gfc_dim = 1024 * 2 # the dimension of full connection layer of generator
dfc_dim = 1024 # the dimension of full connection layer of discriminator
img_dim = 28  # size of the input picture

NOISE_SIZE = 100  # dimension of input noise
LEARNING_RATE = 2e-4 # learning rate of training

epoch = 20         # epoch number of training
output = "./output_dcgan"   # storage path of model and test results
use_cudnn = True  # use cuDNN or not
use_gpu=True       # use GPU or not

def bn(x, name=None, act='relu'):
    return fluid.layers.batch_norm(
        x,
        param_attr=name + '1',
        bias_attr=name + '2',
        moving_mean_name=name + '3',
        moving_variance_name=name + '4',
        name=name,
        act=act)

def conv(x, num_filters, name=None, act=None):
    return fluid.nets.simple_img_conv_pool(
        input=x,
        filter_size=5,
        num_filters=num_filters,
        pool_size=2,
        pool_stride=2,
        param_attr=name + 'w',
        bias_attr=name + 'b',
        use_cudnn=use_cudnn,
        act=act)

def fc(x, num_filters, name=None, act=None):
    return fluid.layers.fc(input=x,
                           size=num_filters,
                           act=act,
                           param_attr=name + 'w',
                           bias_attr=name + 'b')

def D(x, num_classes):
    x = fluid.layers.reshape(x=x, shape=[-1, 1, 28, 28])
    x = conv(x, df_dim, act='leaky_relu',name='conv1')
    x = bn(conv(x, df_dim * 2,name='conv2'), act='leaky_relu',name='bn1')
    x = bn(fc(x, dfc_dim,name='fc1'), act='leaky_relu',name='bn2')
    x = fc(x, num_classes, act='sigmoid',name='fc2')
    return x


def G(x):
    x = fluid.layers.relu(bn(conv(x, 16, act=None,name='conv1')))
    x = fluid.layers.relu(bn(conv(x, 64, act=None,name='conv1'))
    
    x = bn(conv(x, 128, act=None,name='conv1'))
    x = bn(conv(x, 64, act=None,name='conv1'))
    
    
    
    return x
                          
                          
class AutoEncoder(paddle.nn.Layer):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv0 = paddle.nn.Conv1D(in_channels=1,out_channels=32,kernel_size=7,stride=2)
        self.conv1 = paddle.nn.Conv1D(in_channels=32,out_channels=16,kernel_size=7,stride=2)
        self.convT0 = paddle.nn.Conv1DTranspose(in_channels=16,out_channels=32,kernel_size=7,stride=2)
        self.convT1 = paddle.nn.Conv1DTranspose(in_channels=32,out_channels=1,kernel_size=7,stride=2)

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(x)
        x = F.dropout(x,0.2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.convT0(x)
        x = F.relu(x)
        x = F.dropout(x,0.2)
        x = self.convT1(x)
        return x