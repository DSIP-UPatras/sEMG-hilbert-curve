"""
DenseNet implemented in Keras
This implementation is based on the original paper of Gao Huang, Zhuang Liu, Kilian Q. Weinberger and Laurens van der Maaten.
Besides I took some influences by random implementations, especially of Zhuang Liu's Lua implementation.
# References
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- [DenseNet - Lua implementation](https://github.com/liuzhuang13/DenseNet)
@author: Christopher Masch
"""

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers, initializers


__version__ = '0.0.1'
USE_BIAS = False

def DenseNet(input_shape=None, dense_blocks=3, dense_layers=-1, growth_rate=12, nb_classes=None, dropout_rate=None,
             bottleneck=False, compression=1.0, weight_decay=1e-4, depth=40, seed=0):
    """
    Creating a DenseNet
    
    Arguments:
        input_shape  : shape of the input images. E.g. (28,28,1) for MNIST    
        dense_blocks : amount of dense blocks that will be created (default: 3)    
        dense_layers : number of layers in each dense block. You can also use a list for numbers of layers [2,4,3]
                       or define only 2 to add 2 layers at all dense blocks. -1 means that dense_layers will be calculated
                       by the given depth (default: -1)
        growth_rate  : number of filters to add per dense block (default: 12)
        nb_classes   : number of classes
        dropout_rate : defines the dropout rate that is accomplished after each conv layer (except the first one).
                       In the paper the authors recommend a dropout of 0.2 (default: None)
        bottleneck   : (True / False) if true it will be added in convolution block (default: False)
        compression  : reduce the number of feature-maps at transition layer. In the paper the authors recomment a compression
                       of 0.5 (default: 1.0 - will have no compression effect)
        weight_decay : weight decay of L2 regularization on weights (default: 1e-4)
        depth        : number or layers (default: 40)
        
    Returns:
        Model        : A Keras model instance
    """
    
    if nb_classes==None:
        raise Exception('Please define number of classes (e.g. num_classes=10). This is required for final softmax.')
    
    if compression <=0.0 or compression > 1.0:
        raise Exception('Compression have to be a value between 0.0 and 1.0. If you set compression to 1.0 it will be turn off.')
    
    if type(dense_layers) is list:
        if len(dense_layers) != dense_blocks:
            raise AssertionError('Number of dense blocks have to be same length to specified layers')
    elif dense_layers == -1:
        dense_layers = (depth - 4)/3
        if bottleneck:
            dense_layers = dense_layers / 2
        dense_layers = [dense_layers for _ in range(dense_blocks)]
    else:
        dense_layers = [dense_layers for _ in range(dense_blocks)]
        
    img_input = Input(shape=input_shape)
    nb_channels = growth_rate
    
    print('Creating DenseNet %s' % __version__)
    print('#############################################')
    print('Dense blocks: %s' % dense_blocks)
    print('Layers per dense block: %s' % dense_layers)
    print('#############################################')
    
    kernel_init = initializers.glorot_normal(seed=seed)       
    kernel_regl = regularizers.l2(weight_decay)

    # Initial convolution layer
    x = BatchNormalization()(img_input)
    x = Conv2D(2 * growth_rate, (3,3), padding='same',strides=(1,1),
                      use_bias=USE_BIAS, kernel_regularizer=kernel_regl, kernel_initializer=kernel_init)(x)
    
    # Building dense blocks
    for block in range(dense_blocks - 1):
        
        # Add dense block
        x, nb_channels = dense_block(x, dense_layers[block], nb_channels, [(3,3)]*dense_layers[block], growth_rate, dropout_rate, bottleneck, True, kernel_regl, kernel_init)
        
        # Add transition_block
        x = transition_layer(x, nb_channels, 'average', dropout_rate, True, compression, kernel_regl, kernel_init)
        nb_channels = int(nb_channels * compression)
    
    # Add last dense block without transition but for that with global average pooling
    x, nb_channels = dense_block(x, dense_layers[-1], nb_channels, growth_rate, dropout_rate, kernel_regl)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(nb_classes, activation='softmax')(x)
    
    return Model(img_input, x, name='densenet')


def dense_block(x, nb_layers, nb_channels, filters, growth_rate, dropout_rate=None, bottleneck=False, batch_norm=False, regularizer=None, initializer='glorot_uniform', dense_block_id=1):
    """
    Creates a dense block and concatenates inputs
    """
    x_list = [x]
    for i in range(nb_layers):
        bid = 'b{}_c{}'.format(dense_block_id, i)
        cb = convolution_block(x, growth_rate, filters[i], dropout_rate, bottleneck, batch_norm, regularizer, initializer, bid)
        x_list.append(cb)
        x = Concatenate(axis=-1, name='{}_concatenate'.format(bid))(x_list[:])
        nb_channels += growth_rate
    return x, nb_channels


def convolution_block(x, nb_channels, filter_size, dropout_rate=None, bottleneck=False, batch_norm=False, regularizer=None, initializer='glorot_uniform', conv_block_id=1):
    """
    Creates a convolution block consisting of BN-ReLU-Conv.
    Optional: bottleneck, dropout
    """
    
    # Bottleneck
    if bottleneck:
        bottleneckWidth = 4
        if batch_norm:
            x = BatchNormalization(name='{}_batchnorm_1'.format(conv_block_id))(x)
        x = Activation('relu', name='{}_relu_1'.format(conv_block_id))(x)
        x = Conv2D(nb_channels * bottleneckWidth, (1, 1), use_bias=USE_BIAS, 
            kernel_regularizer=regularizer, kernel_initializer=initializer,
            name='{}_conv2d_{:d}_1x1'.format(conv_block_id, nb_channels * bottleneckWidth))(x)
        # Dropout
        if dropout_rate:
            x = Dropout(dropout_rate, name='{}_dropout_1'.format(conv_block_id))(x)
    
    # Standard (BN-ReLU-Conv)
    if batch_norm:
        x = BatchNormalization(name='{}_batchnorm_2'.format(conv_block_id))(x)
    x = Activation('relu', name='{}_relu_2'.format(conv_block_id))(x)
    x = Conv2D(nb_channels, filter_size, padding='same', use_bias=USE_BIAS,
        kernel_regularizer=regularizer, kernel_initializer=initializer,
        name='{}_conv2d_{:d}_{}x{}'.format(conv_block_id, nb_channels, filter_size[0], filter_size[1]))(x)
    
    # Dropout
    if dropout_rate:
        x = Dropout(dropout_rate, name='{}_dropout_2'.format(conv_block_id))(x)
    
    return x


def transition_layer(x, nb_channels, pool_type='average', pool_size=(2,2), dropout_rate=None, batch_norm = False, compression=1.0, regularizer=None, initializer='glorot_uniform', trans_block_id=1):
    """
    Creates a transition layer between dense blocks as transition, which do convolution and pooling.
    Works as downsampling.
    """
    
    trans_block_id = 't{}'.format(trans_block_id)
    if batch_norm:
        x = BatchNormalization(name='{}_batchnorm_1'.format(trans_block_id))(x)
    x = Activation('relu', name='{}_relu'.format(trans_block_id))(x)
    x = Conv2D(int(nb_channels*compression), (1, 1), padding='same', use_bias=USE_BIAS, 
        kernel_regularizer=regularizer, kernel_initializer=initializer,
        name='{}_conv2d_{:d}_1x1'.format(trans_block_id, int(nb_channels*compression)))(x)
    
    # Adding dropout
    if dropout_rate:
        x = Dropout(dropout_rate, name='{}_dropout'.format(trans_block_id))(x)
    
    if pool_type=='average':
        x = AveragePooling2D(pool_size, strides=pool_size, name='{}_pool'.format(trans_block_id))(x)
    elif pool_type=='max':
        x = MaxPooling2D(pool_size, strides=pool_size, name='{}_pool'.format(trans_block_id))(x)
    return x
