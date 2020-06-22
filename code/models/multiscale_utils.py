
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers, initializers


def downscale_pool(x, factor, func=MaxPooling2D, bid=0):
	x = func(pool_size=(factor, factor), strides=(factor, factor), padding='same',
		name='b{}_downscale'.format(bid))(x)
	return x

def downscale_conv(x, factor, func=Conv2D, bid=0, kernel_regularizer=None, kernel_initializer=None):
	x = func(filters, kernel_size=(factor, factor), strides=(factor, factor), padding='same',
		kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
		name='b{}_downscale'.format(bid))(x)
	return x

def regular_convs(x, filters=32, kernel=(3,3), activation='relu', drop=0., bid=0, kernel_regularizer=None, kernel_initializer=None, batch_norm=True, seed=0):
	out = []
	for i in range(len(x)):
		y = Conv2D(filters, kernel, padding='same',
			kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
			name='b{}_regular_conv2d_{}_{}_{}x{}'.format(bid, i, filters, kernel[0], kernel[1]))(x[i])
		if batch_norm:
			y = BatchNormalization(name='b{}_regular_batchnorm_{}'.format(bid, i))(y)
		y = Activation(activation, name='b{}_regular_{}_{}'.format(bid, activation, i))(y)
		y = Dropout(drop, seed=seed, name='b{}_regular_drop_{}'.format(bid, i))(y)
		out.append(y)
	return out

def strided_convs(x, scales, filters=32, activation='relu', drop=0., bid=0, kernel_regularizer=None, kernel_initializer=None, batch_norm=True, seed=0):
	out = []
	for i in range(len(scales)):
		y = Conv2D(filters, (scales[i], scales[i]), strides=(scales[i], scales[i]), 
			kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
			name='b{}_strided_conv2d_{}_{}_{}x{}'.format(bid, i, filters, scales[i], scales[i]))(x[i])
		if batch_norm:
			y = BatchNormalization(name='b{}_strided_batchnorm_{}'.format(bid, i))(y)
		y = Activation(activation, name='b{}_strided_{}_{}'.format(bid, activation, i))(y)
		y = Dropout(drop, seed=seed, name='b{}_strided_drop_{}'.format(bid, i))(y)
		out.append(y)
	return out


if __name__ == '__main__':
	x_input = Input(shape=(8,8,10))
	outputs = []
	outputs_index = [2,4,5] # 1 ... layers
	layers = 5
	classes = 53

	scales = [2, 2, 2]
	x_prev_layer = [x_input]
	for i in range(len(scales)):
		x_prev_layer.append(
			downscale_pool(x_prev_layer[-1], scales[i], bid=i)
			)
	Model(x_input, x_prev_layer).summary()
	## Intermediate layers
	for l in range(1, layers+1):
		xr = regular_convs(x_prev_layer, bid=l)
		xs = strided_convs(x_prev_layer, scales, bid=l)
		x_prev_layer = [xr[0]]

		for i in range(len(xs)):
			x_prev_layer.append(
				Concatenate(name='concatenate_{}{}'.format(l, i))([xr[i+1], xs[i]])
				)
		if l in outputs_index:
			outputs.append(x_prev_layer[-1])

	## Classifiers
	classifiers = []
	for l in range(len(outputs)):
		x = Conv2D(classes, (1, 1), padding='same', name='classifier_conv2d_{}_{}'.format(classes, l))(outputs[i])
		x = Activation('relu', name='classifier_relu_{}'.format(l))(x)
		x = GlobalAveragePooling2D(name='classifier_globalpool_{}'.format(l))(x)
		x = Activation('softmax', name='classifier_softmax_{}'.format(l))(x)
		classifiers.append(x)


	Model(x_input, classifiers).summary()
	print(len(classifiers), [layer.name for layer in classifiers])
