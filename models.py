"""
Lasagne implementation of CIFAR-10 examples from "Identity Mappings in Deep Residual Networks" (https://arxiv.org/abs/1603.05027) and "Wide Residual Networks" (https://arxiv.org/abs/1605.07146)
"""
import sys
sys.setrecursionlimit(10000)
import lasagne
from lasagne.nonlinearities import rectify, softmax, sigmoid
from lasagne.layers import InputLayer, SliceLayer, MaxPool2DLayer, DenseLayer, DropoutLayer, helper, batch_norm, BatchNormLayer
# for ResNet
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer, ElemwiseSumLayer, NonlinearityLayer, PadLayer, GlobalPoolLayer, ExpressionLayer
from lasagne.init import Orthogonal, HeNormal, GlorotNormal

PIXELS = 32
imageSize = PIXELS * PIXELS
num_features = imageSize * 3

he_norm = HeNormal(gain='relu')

def ResNet_FullPreActivation(input_var=None, n=18):
    '''
    Adapted from https://github.com/Lasagne/Recipes/tree/master/papers/deep_residual_learning.
    Tweaked to be consistent with 'Identity Mappings in Deep Residual Networks', Kaiming He et al. 2016 (https://arxiv.org/abs/1603.05027)

    Forumala to figure out depth: 6n + 2
    '''
    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, increase_dim=False, projection=True, first=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters*2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters

        if first:
            # hacky solution to keep layers correct
            bn_pre_relu = l
        else:
            # contains the BN -> ReLU portion, steps 1 to 2
            bn_pre_conv = BatchNormLayer(l)
            bn_pre_relu = NonlinearityLayer(bn_pre_conv, rectify)

        # contains the weight -> BN -> ReLU portion, steps 3 to 5
        conv_1 = batch_norm(ConvLayer(bn_pre_relu, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=he_norm))

        # contains the last weight portion, step 6
        conv_2 = ConvLayer(conv_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=he_norm)

        # add shortcut connections
        if increase_dim:
            # projection shortcut, as option B in paper
            projection = ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None)
            block = ElemwiseSumLayer([conv_2, projection])
        else:
            block = ElemwiseSumLayer([conv_2, l])

        return block

    # Building the network
    l_in = InputLayer(shape=(None, 3, PIXELS, PIXELS), input_var=input_var)

    # first layer, output is 16 x 32 x 32
    l = batch_norm(ConvLayer(l_in, num_filters=16, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=he_norm))

    # first stack of residual blocks, output is 16 x 32 x 32
    l = residual_block(l, first=True)
    for _ in range(1,n):
        l = residual_block(l)

    # second stack of residual blocks, output is 32 x 16 x 16
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # third stack of residual blocks, output is 64 x 8 x 8
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    bn_post_conv = BatchNormLayer(l)
    bn_post_relu = NonlinearityLayer(bn_post_conv, rectify)

    # average pooling
    avg_pool = GlobalPoolLayer(bn_post_relu)

    # fully connected layer
    network = DenseLayer(avg_pool, num_units=10, W=HeNormal(), nonlinearity=softmax)

    return network

# ========================================================================================================================

def ResNet_BottleNeck_FullPreActivation(input_var=None, n=18):
    '''
    Adapted from https://github.com/Lasagne/Recipes/tree/master/papers/deep_residual_learning.
    Tweaked to be consistent with 'Identity Mappings in Deep Residual Networks', Kaiming He et al. 2016 (https://arxiv.org/abs/1603.05027)

    Judging from https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua.
    Number of filters go 16 -> 64 -> 128 -> 256

    Forumala to figure out depth: 9n + 2
    '''

    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_bottleneck_block(l, increase_dim=False, first=False):
        input_num_filters = l.output_shape[1]

        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters * 2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters

        if first:
            # hacky solution to keep layers correct
            bn_pre_relu = l
            out_num_filters = out_num_filters * 4
        else:
            # contains the BN -> ReLU portion, steps 1 to 2
            bn_pre_conv = BatchNormLayer(l)
            bn_pre_relu = NonlinearityLayer(bn_pre_conv, rectify)

        bottleneck_filters = out_num_filters / 4

        # contains the weight -> BN -> ReLU portion, steps 3 to 5
        conv_1 = batch_norm(ConvLayer(bn_pre_relu, num_filters=bottleneck_filters, filter_size=(1,1), stride=(1,1), nonlinearity=rectify, pad='same', W=he_norm))

        conv_2 = batch_norm(ConvLayer(conv_1, num_filters=bottleneck_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=he_norm))

        # contains the last weight portion, step 6
        conv_3 = ConvLayer(conv_2, num_filters=out_num_filters, filter_size=(1,1), stride=(1,1), nonlinearity=None, pad='same', W=he_norm)

        if increase_dim:
            # projection shortcut, as option B in paper
            projection = ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None)
            block = ElemwiseSumLayer([conv_3, projection])

        elif first:
            # projection shortcut, as option B in paper
            projection = ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(1,1), nonlinearity=None, pad='same', b=None)
            block = ElemwiseSumLayer([conv_3, projection])

        else:
            block = ElemwiseSumLayer([conv_3, l])

        return block

    # Building the network
    l_in = InputLayer(shape=(None, 3, PIXELS, PIXELS), input_var=input_var)

    # first layer, output is 16x16x16
    l = batch_norm(ConvLayer(l_in, num_filters=16, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=he_norm))

    # first stack of residual blocks, output is 64x16x16
    l = residual_bottleneck_block(l, first=True)
    for _ in range(1,n):
        l = residual_bottleneck_block(l)

    # second stack of residual blocks, output is 128x8x8
    l = residual_bottleneck_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_bottleneck_block(l)

    # third stack of residual blocks, output is 256x4x4
    l = residual_bottleneck_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_bottleneck_block(l)

    bn_post_conv = BatchNormLayer(l)
    bn_post_relu = NonlinearityLayer(bn_post_conv, rectify)

    # average pooling
    avg_pool = GlobalPoolLayer(bn_post_relu)

    # fully connected layer
    network = DenseLayer(avg_pool, num_units=10, W=HeNormal(), nonlinearity=softmax)

    return network

# ========================================================================================================================
#Take k=8 to get 128 channels in output for addition with sift features
def ResNet_FullPre_Wide(input_var=None, n=6, k=1):
    '''
    Adapted from https://github.com/Lasagne/Recipes/tree/master/papers/deep_residual_learning.

    Tweaked to be consistent with 'Identity Mappings in Deep Residual Networks', Kaiming He et al. 2016 (https://arxiv.org/abs/1603.05027)

    And 'Wide Residual Networks', Sergey Zagoruyko, Nikos Komodakis 2016 (http://arxiv.org/pdf/1605.07146v1.pdf)

    Depth = 6n + 2
    '''
    n_filters = {0:16, 1:16*k, 2:32*k, 3:64*k}

    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, l_feat=[], increase_dim=False, projection=True, first=False, filters=16):
        if increase_dim:
            first_stride = (2,2)
        else:
            first_stride = (1,1)

        if first:
            # hacky solution to keep layers correct
            bn_pre_relu = l
        else:
            # contains the BN -> ReLU portion, steps 1 to 2
            bn_pre_conv = BatchNormLayer(l)
            bn_pre_relu = NonlinearityLayer(bn_pre_conv, rectify)

        # contains the weight -> BN -> ReLU portion, steps 3 to 5
        conv_1 = batch_norm(ConvLayer(bn_pre_relu, num_filters=filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=he_norm))

        dropout = DropoutLayer(conv_1, p=0.3)

        # contains the last weight portion, step 6
        conv_2 = ConvLayer(dropout, num_filters=filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=he_norm)

        # add shortcut connections
        if increase_dim:
            # projection shortcut, as option B in paper
            projection = ConvLayer(l, num_filters=filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None)
            block = ElemwiseSumLayer([conv_2, projection])

        elif first:
            # projection shortcut, as option B in paper
            projection = ConvLayer(l, num_filters=filters, filter_size=(1,1), stride=(1,1), nonlinearity=None, pad='same', b=None)

            #Adding sift features along with projection
            #block = ElemwiseSumLayer([conv_2, projection])
            block = ElemwiseSumLayer([conv_2, projection, l_feat])
            #print "conv2 output size:", lasagne.layers.get_output_shape(conv_2)
            #print "projection output size:", lasagne.layers.get_output_shape(projection)
            #print "feat output size:", lasagne.layers.get_output_shape(l_feat)
        else:
            block = ElemwiseSumLayer([conv_2, l])

        return block

    # Building the network, input layer size is
    #Normal 3x32x32
    #l_in = InputLayer(shape=(None, 3, PIXELS, PIXELS), input_var=input_var)
    #SIFT (128+3)x32x32
    #l_in_all = InputLayer(shape=(None, 131, PIXELS, PIXELS), input_var=input_var)
    #HOG (1+3)x32x32
    l_in_all = InputLayer(shape=(None, 4, PIXELS, PIXELS), input_var=input_var)

    #Slicing into rgb and feat channels
    l_in = SliceLayer(l_in_all, indices=slice(0,3), axis=1)
    l_feat = SliceLayer(l_in_all, indices=slice(3,None), axis=1)

    #SIFT channel
    l_bn_feat = batch_norm(ConvLayer(l_feat, num_filters=n_filters[1], filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=he_norm))
    #l_bn_feat = []
    #print "feat output size:", lasagne.layers.get_output_shape(l_bn_feat)

    # first layer, output is 16 x 32 x 32, conv1
    l = batch_norm(ConvLayer(l_in, num_filters=n_filters[0], filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=he_norm))

    # first stack of residual blocks, output is 64 x 32 x 32, conv2
    l = residual_block(l, l_bn_feat, first=True, filters=n_filters[1])
    for _ in range(1,n):
        l = residual_block(l, filters=n_filters[1])
    #print "ResBlock1 output size:", lasagne.layers.get_output_shape(l)

    # second stack of residual blocks, output is 128 x 16 x 16, conv3
    l = residual_block(l, increase_dim=True, filters=n_filters[2])
    for _ in range(1,(n+2)):
        l = residual_block(l, filters=n_filters[2])

    # third stack of residual blocks, output is 256 x 8 x 8, conv4
    l = residual_block(l, increase_dim=True, filters=n_filters[3])
    for _ in range(1,(n+2)):
        l = residual_block(l, filters=n_filters[3])

    bn_post_conv = BatchNormLayer(l)
    bn_post_relu = NonlinearityLayer(bn_post_conv, rectify)

    # average pooling, layer ouput size is 256 x 1
    avg_pool = GlobalPoolLayer(bn_post_relu)
    #print "Global pooling output size:", lasagne.layers.get_output_shape(avg_pool)

    # fully connected layer, layer output size is 10 x 1
    network = DenseLayer(avg_pool, num_units=10, W=HeNormal(), nonlinearity=softmax)
    #print "Fully connected layer output size:", lasagne.layers.get_output_shape(network)

    return network
