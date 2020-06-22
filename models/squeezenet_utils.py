# SqueezeNet utils
from tensorflow.keras.layers import *
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.models import Model

def fire_module(x, squeeze=16, expand=64, fire_id=1, kernel_regularizer=None, kernel_initializer=None):
    x = Convolution2D(squeeze, (1, 1),
    	padding='valid', kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
    	name='{}_conv2d_sq1x1_{}'.format(fire_id, squeeze))(x)
    x = Activation('relu', name='{}_relu_sq1x1_{}'.format(fire_id, squeeze))(x)

    left = Convolution2D(expand, (1, 1),
    	padding='valid', kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
    	name='{}_conv2d_ex1x1_{}'.format(fire_id, expand))(x)
    left = Activation('relu', name='{}_relu_ex1x1_{}'.format(fire_id, expand))(left)

    right = Convolution2D(expand, (3, 3), 
    	padding='same', kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
    	name='{}_conv2d_ex3x3_{}'.format(fire_id, expand))(x)
    right = Activation('relu', name='{}_relu_ex3x3_{}'.format(fire_id, expand))(right)

    x = Concatenate(axis=-1, name='{}_concatenate'.format(fire_id))([left, right])
    return x


def squeezenet_block(x, n_fm=3, n_sq=16, n_ex=32, n_dropout=0, n_pool='max', n_pool_size=(2,2), block_id=1, kernel_regularizer=None, kernel_initializer='glorot_uniform'):
    for i in range(n_fm):
        last_fire_id = 'b{}_f{}'.format(block_id, i)
        x = fire_module(x, squeeze=n_sq, expand=n_ex, fire_id=last_fire_id, kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer)
    
    x = Dropout(n_dropout, name='b{}_dropout'.format(block_id))(x)
    if n_pool == 'max':
        x = MaxPooling2D(n_pool_size, strides = n_pool_size, name='b{}_pool'.format(block_id))(x)
    else:
        x = AveragePooling2D(n_pool_size, strides = n_pool_size, name='b{}_pool'.format(block_id))(x)

    return x

