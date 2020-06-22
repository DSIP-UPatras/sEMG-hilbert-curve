from tensorflow.keras.layers import *
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.models import Model
import math
import numpy as np


# VGG-like
def VGGNet(input_shape, classes, 
    n_blocks = 3, n_pool='average', n_pool_size=(2,2), n_dropout=0., n_l2=0.0005, 
    n_init='glorot_normal', n_batch_norm=False, 
    n_seed=0, output_layer_name='output'):
    """ 
    
    Arguments:
        input_shape -- tuple, dimensions of the input in the form (height, width, channels)
        classes -- integer, number of classes to be classified, defines the dimension of the softmax unit
        n_pool -- string, pool method to be used {'max', 'average'}
        n_dropout -- float, rate of dropping units
        n_l2 -- float, ampunt of weight decay regularization
        n_init -- string, type of kernel initializer {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'normal', 'uniform'}
        n_seed -- integer, random seed for kernel intiliazer

    Returns:
        model -- keras.models.Model (https://keras.io)

        ## PHCNet - Summary
        # CONV,16,3,3,1,1,same  #16*3*3*10+16   = 1456
        # CONV,16,3,3,1,1,same  #16*3*3*16+16   = 2320
        # CONV,16,3,3,1,1,same  #16*3*3*16+16   = 2320
        # POOL,2,2,2,2
        # CONV,32,3,3,1,1,same  #32*3*3*16+32   = 4640
        # CONV,32,3,3,1,1,same  #32*3*3*32+32   = 9248
        # POOL,2,2,2,2
        # CONV,64,3,3,1,1,same  #64*3*3*32+64   = 18496
        # CONV,64,3,3,1,1,same  #64*3*3*64+64   = 36928
        # POOL,2,2,2,2
        # FC,53                 #53*64+53       = 3445
        ## Params: 78853, RF: 38
    """

    if n_init == 'glorot_normal':
        kernel_init = initializers.glorot_normal(seed=n_seed)
    elif n_init == 'glorot_uniform':
        kernel_init = initializers.glorot_uniform(seed=n_seed)
    elif n_init == 'he_normal':
        kernel_init = initializers.he_normal(seed=n_seed)
    elif n_init == 'he_uniform':
        kernel_init = initializers.he_uniform(seed=n_seed)
    elif n_init == 'normal':
        kernel_init = initializers.normal(seed=n_seed)
    elif n_init == 'uniform':
        kernel_init = initializers.uniform(seed=n_seed)
    kernel_regl = regularizers.l2(n_l2)

    # n_pool_size = (2,2)
    # n_blocks = 3

    ## Block 0 [Input]
    X_input = Input(input_shape, name='b0_input')
    X = X_input
    if n_batch_norm:
        X = BatchNormalization(name='b0_batchnorm')(X)
    
    ## Block 1 [Pad -> Conv -> ReLU -> Dropout]
    X = Conv2D(16, (3, 3), padding='same', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, name='b1_conv2d_32_3x3')(X)
    if n_batch_norm:
        X = BatchNormalization(name='b1_batchnorm')(X)
    X = Activation('relu', name='b1_relu')(X)
    # X = Dropout(n_dropout, name='b1_dropout')(X)
    
    filters = 16
    ins = min(input_shape[0], input_shape[1])
    num_pool_layers = int(min(math.log(ins, n_pool_size[0]), n_blocks))
    pool_layers = np.linspace(n_blocks, 0, num_pool_layers, endpoint=False, dtype=int)-1

    for i in range(n_blocks):
        ## Block 2 [Pad -> Conv -> ReLU -> -> Dropout -> Pool]
        X = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, name='b{}_1_conv2d_32_3x3'.format(i+2))(X)
        X = Activation('relu', name='b{}_1_relu'.format(i+2))(X)
        X = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, name='b{}_2_conv2d_32_3x3'.format(i+2))(X)
        X = Activation('relu', name='b{}_2_relu'.format(i+2))(X)
        X = Dropout(n_dropout, name='b{}_dropout'.format(i+2))(X)
        if i in pool_layers:
            if n_pool == 'max':
                X = MaxPooling2D(n_pool_size, strides = n_pool_size, name='b{}_pool'.format(i+2))(X)
            else:
                X = AveragePooling2D(n_pool_size, strides = n_pool_size, name='b{}_pool'.format(i+2))(X)
        filters = filters*2
    

 
    X = GlobalAveragePooling2D()(X)

    ## Block 5 [FC -> Softmax]
    X = Dense(classes, kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, name='b{}_fc_{}'.format(n_blocks+2, classes))(X)
    X = Activation('softmax', name=output_layer_name)(X)
    
    model = Model(inputs = X_input, outputs = X, name='PHCNet')
    # model.summary()

    return model

## SqueezeNet-like
def SqueezeNet(input_shape, classes, 
    n_blocks = 3, n_pool='average', n_pool_size=(2,2), n_dropout=0., n_l2=0.0005, 
    n_init='glorot_normal', n_batch_norm=False, 
    n_seed=0, output_layer_name='output'):
    """ 
    
    Arguments:
        input_shape -- tuple, dimensions of the input in the form (height, width, channels)
        classes -- integer, number of classes to be classified, defines the dimension of the softmax unit
        n_pool -- string, pool method to be used {'max', 'average'}
        n_dropout -- float, rate of dropping units
        n_l2 -- float, ampunt of weight decay regularization
        n_init -- string, type of kernel initializer {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'normal', 'uniform'}
        n_seed -- integer, random seed for kernel intiliazer
    
    Returns:
        model -- keras.models.Model (https://keras.io)

        ## PHCNet - Summary
        # CONV,16,3,3,1,1,same  #16*3*3*10+16              = 1456
        # SQUEEZE_BLOCK,4,8,16  #136+3*264 +4*144+4*1168   = 6176
        # SQUEEZE_BLOCK,4,8,32  #264+3*520 +4*288+4*2336   = 12320
        # SQUEEZE_BLOCK,4,8,64  #520+3*1032+4*576+4*4672   = 24608
        # CONV,53,1,1,1,1,same  #53*128+53                 = 6837
        ## Params: 51397, RF: ?
    """
    from models.squeezenet_utils import squeezenet_block

    if n_init == 'glorot_normal':
        kernel_init = initializers.glorot_normal(seed=n_seed)
    elif n_init == 'glorot_uniform':
        kernel_init = initializers.glorot_uniform(seed=n_seed)
    elif n_init == 'he_normal':
        kernel_init = initializers.he_normal(seed=n_seed)
    elif n_init == 'he_uniform':
        kernel_init = initializers.he_uniform(seed=n_seed)
    elif n_init == 'normal':
        kernel_init = initializers.normal(seed=n_seed)
    elif n_init == 'uniform':
        kernel_init = initializers.uniform(seed=n_seed)
    kernel_regl = regularizers.l2(n_l2)

    # n_pool_size = (2,2)
    # n_blocks = 3

    ## Block 0 [Input]
    X_input = Input(input_shape, name='b0_input')
    X = X_input
    if n_batch_norm:
        X = BatchNormalization(name='b0_batchnorm')(X)

    ## Block 1 [Pad -> Conv -> ReLU -> Dropout]
    X = Conv2D(16, (3, 3), padding='same', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, name='b1_conv2d_32_3x3')(X)
    if n_batch_norm:
        X = BatchNormalization(name='b1_batchnorm')(X)
    X = Activation('relu', name='b1_relu')(X)
    
    iters = 3 if input_shape[0]>=8 else 2
    ins = min(input_shape[0], input_shape[1])
    iters = int(math.log(ins, n_pool_size[0]))

    for i in range(iters):
    	X = squeezenet_block(X, n_fm=n_blocks, n_sq=16, n_ex=32, n_dropout=n_dropout, n_pool=n_pool, n_pool_size=n_pool_size, block_id=i+2, kernel_regularizer=kernel_regl, kernel_initializer=kernel_init)


    X = Conv2D(classes, (1, 1), padding='same', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, name='b5_conv2d_{}_1x1'.format(classes))(X)
    X = Activation('relu', name='b5_relu')(X)
    X = GlobalAveragePooling2D(name='b5_globalpool')(X)
    X = Activation('softmax', name=output_layer_name)(X)
    
    model = Model(inputs = X_input, outputs = X, name='PHCNet')
    # model.summary()

    return model

def DenseNet(input_shape, classes, 
    n_blocks = 3, n_pool='average', n_pool_size=(2,2), n_dropout=0., n_l2=0.0005, 
    n_init='glorot_normal', n_batch_norm=False, 
    n_seed=0, output_layer_name='output'):
    """ 
    
    Arguments:
        input_shape -- tuple, dimensions of the input in the form (height, width, channels)
        classes -- integer, number of classes to be classified, defines the dimension of the softmax unit
        n_pool -- string, pool method to be used {'max', 'average'}
        n_dropout -- float, rate of dropping units
        n_l2 -- float, ampunt of weight decay regularization
        n_init -- string, type of kernel initializer {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'normal', 'uniform'}
        n_seed -- integer, random seed for kernel intiliazer
    
    Returns:
        model -- keras.models.Model (https://keras.io)

        ## PHCNet - Summary
        # CONV,16,3,3,1,1,same    #16*3*3*10+16          = 1456
        # DENSE_BLOCK,3,16        #768+1536+2304         = 4608
        # TRANS_BLOCK,32/64,2,2   #                      = 2048
        # DENSE_BLOCK,4,16        #4608+6912+9216+11520  = 32256
        # TRANS_BLOCK,64/96,2,2   #                      = 6144
        # DENSE_BLOCK,3,16        #9216+11520+13824      = 34560
        # TRANS_BLOCK,64/112,2,2  #                      = 7168
        # CONV,53,1,1,1,1,same    #53*64+53              = 3445
        ## Params: 91685, RF: ??
    """
    from models.densenet_utils import dense_block, transition_layer

    if n_init == 'glorot_normal':
        kernel_init = initializers.glorot_normal(seed=n_seed)
    elif n_init == 'glorot_uniform':
        kernel_init = initializers.glorot_uniform(seed=n_seed)
    elif n_init == 'he_normal':
        kernel_init = initializers.he_normal(seed=n_seed)
    elif n_init == 'he_uniform':
        kernel_init = initializers.he_uniform(seed=n_seed)
    elif n_init == 'normal':
        kernel_init = initializers.normal(seed=n_seed)
    elif n_init == 'uniform':
        kernel_init = initializers.uniform(seed=n_seed)
    kernel_regl = regularizers.l2(n_l2)

    # n_pool_size = (2,2)
    # n_blocks = 3
    growth = 16
    nb_channels = 16
    compressions = np.arange(n_blocks)+2 #[1/2, 2/3, 3/4]
    compressions = (compressions-1) / compressions
    compressions = compressions[::-1]
    # compressions = [1/2] * n_blocks

    ## Block 0 [Input]
    X_input = Input(input_shape, name='b0_input')
    X = X_input
    if n_batch_norm:
        X = BatchNormalization(name='b0_batchnorm')(X)

    ## Block 1 [Pad -> Conv -> ReLU -> Dropout]
    X = Conv2D(16, (3, 3), padding='same', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, name='b1_conv2d_16_3x3')(X)
    # if n_batch_norm:
    #     X = BatchNormalization(name='b1_batchnorm')(X)
    # X = Activation('relu', name='b1_relu')(X)
    
    # if n_blocks == 3:
    #     X, nb_channels = dense_block(X, 3, nb_channels, [(3,3)]*3, growth, dropout_rate=n_dropout, bottleneck=False, batch_norm=n_batch_norm, regularizer=kernel_regl, initializer=kernel_init, dense_block_id=1)
    #     X = transition_layer(X, nb_channels, pool_type=n_pool, pool_size=n_pool_size, dropout_rate=n_dropout, batch_norm=False, compression=compressions[0], regularizer=kernel_regl, initializer=kernel_init, trans_block_id=1)
    #     nb_channels = int(nb_channels * compressions[0])

    # X, nb_channels = dense_block(X, 4, nb_channels, [(3,3)]*4, growth, dropout_rate=n_dropout, bottleneck=False, batch_norm=n_batch_norm, regularizer=kernel_regl, initializer=kernel_init, dense_block_id=2)
    # X = transition_layer(X, nb_channels, pool_type=n_pool, pool_size=n_pool_size, dropout_rate=n_dropout, batch_norm=False, compression=compressions[1], regularizer=kernel_regl, initializer=kernel_init, trans_block_id=2)
    # nb_channels = int(nb_channels * compressions[1])

    # X, nb_channels = dense_block(X, 3, nb_channels, [(3,3)]*3, growth, dropout_rate=n_dropout, bottleneck=False, batch_norm=n_batch_norm, regularizer=kernel_regl, initializer=kernel_init, dense_block_id=3)
    # X = transition_layer(X, nb_channels, pool_type=n_pool, pool_size=n_pool_size, dropout_rate=n_dropout, batch_norm=False, compression=compressions[2], regularizer=kernel_regl, initializer=kernel_init, trans_block_id=3)
    
    ins = min(input_shape[0], input_shape[1])
    num_pool_layers = int(min(math.log(ins, n_pool_size[0]), n_blocks))
    pool_layers = np.linspace(n_blocks, 0, num_pool_layers, endpoint=False, dtype=int)-1

    for i in range(n_blocks):
        # print('dense_block')
        X, nb_channels = dense_block(X, 2, nb_channels, [(3,3)]*2, growth, dropout_rate=n_dropout, bottleneck=False, batch_norm=n_batch_norm, regularizer=kernel_regl, initializer=kernel_init, dense_block_id=i+1)
        if i in pool_layers:
            # print('transition_block')
            X = transition_layer(X, nb_channels, pool_type=n_pool, pool_size=n_pool_size, dropout_rate=n_dropout, batch_norm=False, compression=compressions[i], regularizer=kernel_regl, initializer=kernel_init, trans_block_id=i+1)
            nb_channels = int(nb_channels * compressions[i])


    X = Activation('relu')(X)

    X = Conv2D(classes, (1, 1), padding='same', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, name='b{}_conv2d_{}_1x1'.format(n_blocks+1, classes))(X)
    X = Activation('relu', name='b{}_relu'.format(n_blocks+1))(X)
    X = GlobalAveragePooling2D(name='b{}_globalpool'.format(n_blocks+1))(X)
    X = Activation('softmax', name=output_layer_name)(X)
    
    model = Model(inputs = X_input, outputs = X, name='PHCNet')
    # model.summary()

    return model


def MSHilbNet(input_shape, classes, 
    n_blocks = 3, n_scales=[2, 2], n_outputs=[1,2,3], n_dropout=0., n_l2=0.0005, 
    n_init='glorot_normal', n_batch_norm=False, n_classifier='average', n_classifier_type='1_conv',
    n_seed=0, output_layer_name='output'):
    """ 
    
    Arguments:
        input_shape -- tuple, dimensions of the input in the form (height, width, channels)
        classes -- integer, number of classes to be classified, defines the dimension of the softmax unit
        n_pool -- string, pool method to be used {'max', 'average'}
        n_dropout -- float, rate of dropping units
        n_l2 -- float, ampunt of weight decay regularization
        n_init -- string, type of kernel initializer {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'normal', 'uniform'}
        n_seed -- integer, random seed for kernel intiliazer
    
    Returns:
        model -- keras.models.Model (https://keras.io)

        ## PHCNet - Summary
    """
    from models.multiscale_utils import downscale_pool, regular_convs, strided_convs
    from models.custom_layers import AttentionWithContext as Attention
    import tensorflow as tf

    if n_init == 'glorot_normal':
        kernel_init = initializers.glorot_normal(seed=n_seed)
    elif n_init == 'glorot_uniform':
        kernel_init = initializers.glorot_uniform(seed=n_seed)
    elif n_init == 'he_normal':
        kernel_init = initializers.he_normal(seed=n_seed)
    elif n_init == 'he_uniform':
        kernel_init = initializers.he_uniform(seed=n_seed)
    elif n_init == 'normal':
        kernel_init = initializers.normal(seed=n_seed)
    elif n_init == 'uniform':
        kernel_init = initializers.uniform(seed=n_seed)
    kernel_regl = regularizers.l2(n_l2)

    
    ## Block 0 [Input]
    X_input = Input(input_shape, name='b0_input')
    X = X_input
    if n_batch_norm:
        X = BatchNormalization(name='b0_batchnorm')(X)
    
    outputs = []
    # outputs_index = [2,4,5] # 1 ... layers

    # Input layer
    x_prev_layer = [X]
    for i in range(len(n_scales)):
        x_prev_layer.append(
            downscale_pool(x_prev_layer[-1], n_scales[i], func=MaxPooling2D, bid=i)
            )

    ## Intermediate layers
    for i in range(1, n_blocks+1):
        xr = regular_convs(x_prev_layer, filters=32, activation='relu', drop=n_dropout, bid=i, kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, batch_norm=n_batch_norm)
        xs = strided_convs(x_prev_layer, n_scales, filters=16, activation='relu', drop=n_dropout, bid=i, kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, batch_norm=n_batch_norm)
        x_prev_layer = [xr[0]]

        for j in range(len(xs)):
            x_prev_layer.append(
                Concatenate(name='b{}_concatenate_{}'.format(i, j))([xr[j+1], xs[j]])
                )
        if i in n_outputs:
            outputs.append([i,x_prev_layer[-1]])

    ## Classifiers
    classifiers = []
    for i in range(len(outputs)):
        if n_classifier_type == '1_conv':
            x = Conv2D(classes, (1, 1), padding='same', name='b{}_classifier_conv2d_{}_1x1'.format(outputs[i][0], classes))(outputs[i][1])
            # x = Activation('relu', name='b{}_classifier_relu'.format(outputs[i][0], i))(x)
        elif n_classifier_type == '2_conv':
            x = Conv2D(128, (1, 1), padding='same', activation='relu', name='b{}_classifier_conv2d_128_1x1'.format(outputs[i][0]))(outputs[i][1])
            x = Conv2D(classes, (1, 1), padding='same', name='b{}_classifier_conv2d_{}_1x1'.format(outputs[i][0], classes))(x)
        elif n_classifier_type == '3_conv':
            x = Conv2D(256, (1, 1), padding='same', activation='relu', name='b{}_classifier_conv2d_256_1x1'.format(outputs[i][0]))(outputs[i][1])
            x = Conv2D(128, (1, 1), padding='same', activation='relu', name='b{}_classifier_conv2d_128_1x1'.format(outputs[i][0]))(x)
            x = Conv2D(classes, (1, 1), padding='same', name='b{}_classifier_conv2d_{}_1x1'.format(outputs[i][0], classes))(x)
        x = GlobalAveragePooling2D(name='b{}_classifier_globalpool'.format(outputs[i][0]))(x)
        x = Activation('softmax', name='b{}_classifier_softmax'.format(outputs[i][0]))(x)
        classifiers.append(x)
    
    if len(classifiers) >= 2:
        if n_classifier == 'average':
            # X = Average(name='b{}_average'.format(n_blocks + 1))(classifiers)
            X = Average(name=output_layer_name)(classifiers)
        elif n_classifier == 'attention':
            for i in range(len(classifiers)):
                classifiers[i] = Reshape((1,classes), name='b{}_classifier_reshape'.format(outputs[i][0]))(classifiers[i])
            X = Concatenate(axis=1, name='output_concatenate')(classifiers)
            X = Attention(name=output_layer_name, return_attention=False)(X)
        else:
            pass

    else:
        X = Lambda(lambda x: x, name=output_layer_name) (classifiers[0])
    # X = GlobalAveragePooling2D(name='b{}_globalpool'.format(n_blocks+1))(X)
    # X = Activation('softmax', name=output_layer_name)(X)
    
    model = Model(inputs = X_input, outputs = X, name='PHCNet')
    # model.summary()

    return model



def crop(dimension, start, end):
    """ Crops (or slices) a Tensor on a given dimension from start to end
        example : to crop tensor x[:, :, 5:10] 
        call slice(2, 5, 10) as you want to crop on the second dimension"""
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)
