import tensorflow as tf
import sys
sys.path.append('./')
from settings import Settings
CONFIG = Settings()

############################ NETWORK ARCHITECTURE ############################

# USEFUL LAYERS
fc = tf.layers.dense
conv = tf.layers.conv2d
deconv = tf.layers.conv2d_transpose
relu = tf.nn.relu
maxpool = tf.layers.max_pooling2d
dropout_layer = tf.layers.dropout
batchnorm = tf.layers.batch_normalization
winit = tf.contrib.layers.xavier_initializer()
repeat = tf.contrib.layers.repeat
arg_scope = tf.contrib.framework.arg_scope
l2_regularizer = tf.contrib.layers.l2_regularizer


def downsample(input, n_filters_out, is_training, bn=False, use_relu=False, l2=None, name="down"):
	x = tf.layers.separable_conv2d(input, n_filters_out, (3, 3), strides=2, padding='same', activation=None,
								   dilation_rate=1, use_bias=False, depthwise_initializer=winit,
								   pointwise_initializer=winit,
								   pointwise_regularizer=l2_regularizer(0.00004))
	x = tf.layers.batch_normalization(x, training=is_training)
	x = tf.nn.relu(x)
	return x


def upsampling(inputs, scale):
	return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * scale, tf.shape(inputs)[2] * scale],
                                    align_corners=True)


def residual_separable(input, n_filters,  is_training, dropout=0.3, dilation=1, l2=None, name="down"):
	x= tf.layers.separable_conv2d(input, n_filters, (3, 3), strides=1, padding='same', activation=None,
                        dilation_rate=dilation, use_bias=False, depthwise_initializer=winit, pointwise_initializer=winit,
                        pointwise_regularizer=l2_regularizer(0.00004))
	x = tf.layers.batch_normalization(x, training=is_training)
	x = dropout_layer(x, rate=dropout)
	if input.shape[3] == x.shape[3]:
		x = tf.add(x, input)
	x = tf.nn.relu(x)
	return x

def residual_separable_multi(input, n_filters,  is_training, dropout=0.3, dilation=1, l2=None, name="down"):
	input_b = tf.identity(input)
	d = tf.keras.layers.DepthwiseConv2D(3, strides=(1, 1), depth_multiplier=1, padding='same', use_bias=False)
	x = d(input)
	x = tf.layers.batch_normalization(x, training=is_training)
	x = tf.nn.relu(x)

	d2= tf.keras.layers.DepthwiseConv2D(3, strides=(1, 1), depth_multiplier=1, padding='same', use_bias=False)
	d2.dilation_rate = (dilation, dilation)
	x2 = d2(input)
	x2 = tf.layers.batch_normalization(x2, training=is_training)
	x2 = tf.nn.relu(x2)

	x +=x2

	x = tf.layers.conv2d(x, n_filters, 1, strides=1, padding='same', activation=None, kernel_initializer=winit,
						 dilation_rate=1, use_bias=False, kernel_regularizer=l2_regularizer(0.00004))
	x = tf.layers.batch_normalization(x, training=is_training)

	x = dropout_layer(x, rate=dropout)

	if input.shape[3] == x.shape[3]:
		x = tf.add(x, input_b)

	x = tf.nn.relu(x)
	return x


def encoder_module(input, n_filters,  is_training, dropout=0.3, dilation=[1,1], l2=None, name="down"):
	x = residual_separable(input, n_filters,  is_training, dropout=dropout, dilation=dilation[0], l2=l2, name=name)
	x = residual_separable(x, n_filters,  is_training, dropout=dropout, dilation=dilation[1], l2=l2, name=name)
	return x

def encoder_module_multi(input, n_filters,  is_training, dropout=0.3, dilation=[1,1], l2=None, name="down"):
	x = residual_separable_multi(input, n_filters,  is_training, dropout=dropout, dilation=dilation[0], l2=l2, name=name)
	x = residual_separable_multi(x, n_filters,  is_training, dropout=dropout, dilation=dilation[1], l2=l2, name=name)
	return x

# Components
def maxpool_layer(x,size,stride,name):
	with tf.name_scope(name):
		x = tf.layers.max_pooling2d(x, size, stride, padding='VALID')

	return x

def conv_layer(x, kernel, depth, train_logical, name):
	with tf.variable_scope(name):
		x = tf.layers.conv2d(x, depth, kernel, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), bias_initializer=tf.zeros_initializer())
		x = tf.layers.batch_normalization(x, training=train_logical, momentum=0.99, epsilon=0.001, center=True, scale=True)
		x = tf.nn.relu(x)

	return x

def conv_transpose_layer(x, kernel, depth, stride, name):
	with tf.variable_scope(name):
		x = tf.contrib.layers.conv2d_transpose(x, depth, kernel, stride=stride, scope=name)

	return x

 

def mininet3d(points, neighbors, train_logical, mask):


	with tf.variable_scope('net'):

		'''
		Calculate Relative features
		'''

		# neighbours: (?, 32768, 8, 5):  (b, points, neigbours, features)
		# points: (?, 32768, 1, 5):  (b, points, neigbours, features)
		# mask: (?, 64, 512, 1):  (b, h, w, 1)
		input_image = tf.reshape(points, (-1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, 5))
		# input_image: (?, 64, 512, 5):  (b, h, w, 5)


		point_groups = tf.image.extract_image_patches(input_image, ksizes=[1, 4, 4, 1], strides=[1, 4, 4, 1], rates=[1, 1, 1, 1], padding='SAME')
		point_groups = tf.reshape(point_groups, (-1, 16, 128, 16, 5)) # (b , h, w, , neighb, feat)
		mask_groups = tf.image.extract_image_patches(mask, ksizes=[1, 4, 4, 1], strides=[1, 4, 4, 1], rates=[1, 1, 1, 1], padding='SAME')
		mask_groups = tf.reshape(mask_groups, (-1, 16, 128, 16, 1)) # (b , h, w,  neighb, feat)
		# point_groups: (?, 16, 128, 16, 5):  (b , h, w, neigh, feat)



		# Get the mean point (taking apart non-valid points
		point_groups_sumF =tf.reduce_sum(point_groups, axis=3) #(b , h, w, feat)
		mask_groups_sumF = tf.reduce_sum(mask_groups, axis=3)
		point_groups_mean = point_groups_sumF / mask_groups_sumF
		# point_groups_mean: (?, 16, 128, 5):  (b , h, w, feat)

		point_groups_mean = tf.expand_dims(point_groups_mean, 3)
		point_groups_mean = tf.tile(point_groups_mean, [1, 1, 1, 16, 1])
		is_nan = tf.math.is_nan(point_groups_mean)
		point_groups_mean = tf.where(is_nan, x=tf.zeros_like(point_groups_mean), y=point_groups_mean)


		# substract mean point to points
		relative_points = point_groups - point_groups_mean

		mask_groups_tile = tf.tile(mask_groups, [1, 1, 1, 1, 5])
		relative_points = tf.where(tf.cast(mask_groups_tile, dtype=tf.bool), x=relative_points, y=tf.zeros_like(relative_points))



		# relative_points: (?, 16, 128, 16, 5):  (b , h, w,  neighb, feat)
		xyz_rel = relative_points[:, :, :, :, 0:3]
		relative_distance = tf.expand_dims(tf.norm(xyz_rel, ord='euclidean', axis=-1), axis=-1)
		input_final = tf.concat([point_groups, relative_points, relative_distance], axis=-1)
		# input_final: (?, 16, 128, 16, 11):  (b , h, w, neighb, feat)
		# neighbours: (?, 32768, 8, 5):  (b, points, neigbours, features)

		input_final = tf.reshape(input_final, (-1, 16*128, 16, 11)) # (?, 16*128, 16, 11)


		x_conv = tf.layers.conv2d(input_final, 192, (1, 16), padding='VALID', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), bias_initializer=tf.zeros_initializer())
		x_conv = tf.layers.batch_normalization(x_conv, training=train_logical, momentum=0.99, epsilon=0.001, center=True, scale=True)
		x_conv = tf.nn.relu(x_conv)


		x = conv_layer(input_final, (1, 1), 24, train_logical, name="point-based_local_1")
		x = conv_layer(x, (1, 1), 48, train_logical, name="point-based_local_2")
		x_context = maxpool_layer(x, (1, 16), (1, 1), name="point-maxpool1")

		x = conv_layer(x, (1, 1), 96, train_logical, name="point-based_local_3")
		x = conv_layer(x, (1, 1), 192, train_logical, name="point-based_local_4")

		x_local = maxpool_layer(x, (1, 16), (1, 1), name="point-maxpool1")




		x_context_image = tf.reshape(x_context, (-1, 16, 128, 48))
		x_context_1 = tf.image.extract_image_patches(x_context_image, ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
		x_context_2 = tf.image.extract_image_patches(x_context_image, ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 2, 2, 1], padding='SAME')
		x_context_3 = tf.image.extract_image_patches(x_context_image, ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 3, 3, 1], padding='SAME')
		x_context_1 = tf.reshape(x_context_1, (-1, 16, 128, 9, 48))
		x_context_2 = tf.reshape(x_context_2, (-1, 16, 128, 9, 48))
		x_context_3 = tf.reshape(x_context_3, (-1, 16, 128, 9, 48))
		x_context_1 = tf.reshape(x_context_1, (-1, 16*128, 9, 48))
		x_context_2 = tf.reshape(x_context_2, (-1, 16*128, 9, 48))
		x_context_3 = tf.reshape(x_context_3, (-1, 16*128, 9, 48))
		x_context_1 = conv_layer(x_context_1, (1, 1), 96, train_logical, name="point-based_context_1")
		x_context_2 = conv_layer(x_context_2, (1, 1), 48, train_logical, name="point-based_context_2")
		x_context_3 = conv_layer(x_context_3, (1, 1), 48, train_logical, name="point-based_context_3")
		x_context_1 = maxpool_layer(x_context_1, (1, 9), (1, 1), name="point-maxpool1_cont")
		x_context_2 = maxpool_layer(x_context_2, (1, 9), (1, 1), name="point-maxpool2_cont")
		x_context_3 = maxpool_layer(x_context_3, (1, 9), (1, 1), name="point-maxpool3_cont")
		x_context_1 = tf.reshape(x_context_1, (-1, 16, 128, 96))
		x_context_2 = tf.reshape(x_context_2, (-1, 16, 128, 48))
		x_context_3 = tf.reshape(x_context_3, (-1, 16, 128, 48))

		x_context = tf.concat([x_context_3, x_context_2, x_context_1], -1)

		x_local = tf.reshape(x_local, (-1, 16, 128, 192))
		x_conv = tf.reshape(x_conv, (-1, 16, 128, 192))

		x = tf.concat([x_conv, x_local, x_context], -1)

		atten = tf.reduce_mean(x, [1, 2], keep_dims=True)
		atten = tf.layers.conv2d(atten, 576, (1, 1), padding='VALID', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), bias_initializer=tf.zeros_initializer())
		atten = tf.sigmoid(atten)
		x = tf.multiply(x, atten)
		x = conv_layer(x, (1, 1), 192, train_logical, name="last")


		x_branch = downsample(input_image, n_filters_out=192, is_training=train_logical)

		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x2 = encoder_module(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = downsample(x2, n_filters_out=192, is_training=train_logical)
		x = encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[2, 2], dropout=0.25)
		x = encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[1, 4], dropout=0.25)
		x = encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[2, 8], dropout=0.25)
		x = encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[2, 2], dropout=0.25)
		x = encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[1, 4], dropout=0.25)
		x = encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[2, 8], dropout=0.25)
		x = encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[2, 2], dropout=0.25)
		x = encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[1, 4], dropout=0.25)
		x = encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[2, 8], dropout=0.25)
		x = encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[2, 2], dropout=0.25)
		x = encoder_module_multi(x, n_filters=192, is_training=train_logical, dilation=[1, 4], dropout=0.25)

		x = upsampling(x, 2)
		x = x + x2
		x = residual_separable(x, 192, train_logical, dropout=0, dilation=1)
		x = residual_separable(x, 192, train_logical, dropout=0, dilation=1)
		x = residual_separable(x, 192, train_logical, dropout=0, dilation=1)
		x = residual_separable(x, 192, train_logical, dropout=0, dilation=1)
		x = upsampling(x, 2)
		x = x + x_branch

		x = residual_separable(x, 96, train_logical, dropout=0, dilation=1)
		x = residual_separable(x, 96, train_logical, dropout=0, dilation=1)
		x = upsampling(x, 2)

		x = conv_layer(x, (3, 3), CONFIG.N_CLASSES, train_logical, 'fully_connected1')

		y = tf.identity(x, name='y')

		return y






def mininet3dsmall(points, neighbors, train_logical, mask):

	with tf.variable_scope('net'):
		'''
		Calculate Relative features
		n'''

		# neighbours: (?, 32768, 8, 5):  (b, points, neigbours, features)
		# points: (?, 32768, 1, 5):  (b, points, neigbours, features)
		# mask: (?, 64, 512, 1):  (b, h, w, 1)
		input_image = tf.reshape(points, (-1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, 5))
		# input_image: (?, 64, 512, 5):  (b, h, w, 5)


		point_groups = tf.image.extract_image_patches(input_image, ksizes=[1, 4, 4, 1], strides=[1, 4, 4, 1], rates=[1, 1, 1, 1], padding='SAME')
		point_groups = tf.reshape(point_groups, (-1, 16, 128, 16, 5)) # (b , h, w, , neighb, feat)
		mask_groups = tf.image.extract_image_patches(mask, ksizes=[1, 4, 4, 1], strides=[1, 4, 4, 1], rates=[1, 1, 1, 1], padding='SAME')
		mask_groups = tf.reshape(mask_groups, (-1, 16, 128, 16, 1)) # (b , h, w,  neighb, feat)
		# point_groups: (?, 16, 128, 16, 5):  (b , h, w, neigh, feat)



		# Get the mean point (taking apart non-valid points
		point_groups_sumF =tf.reduce_sum(point_groups, axis=3) #(b , h, w, feat)
		mask_groups_sumF = tf.reduce_sum(mask_groups, axis=3)
		point_groups_mean = point_groups_sumF / mask_groups_sumF
		# point_groups_mean: (?, 16, 128, 5):  (b , h, w, feat)

		point_groups_mean = tf.expand_dims(point_groups_mean, 3)
		point_groups_mean = tf.tile(point_groups_mean, [1, 1, 1, 16, 1])
		is_nan = tf.math.is_nan(point_groups_mean)
		point_groups_mean = tf.where(is_nan, x=tf.zeros_like(point_groups_mean), y=point_groups_mean)


		# substract mean point to points
		relative_points = point_groups - point_groups_mean

		mask_groups_tile = tf.tile(mask_groups, [1, 1, 1, 1, 5])
		relative_points = tf.where(tf.cast(mask_groups_tile, dtype=tf.bool), x=relative_points, y=tf.zeros_like(relative_points))



		# relative_points: (?, 16, 128, 16, 5):  (b , h, w,  neighb, feat)
		xyz_rel = relative_points[:, :, :, :, 0:3]
		relative_distance = tf.expand_dims(tf.norm(xyz_rel, ord='euclidean', axis=-1), axis=-1)
		input_final = tf.concat([point_groups, relative_points, relative_distance], axis=-1)
		# input_final: (?, 16, 128, 16, 11):  (b , h, w, neighb, feat)
		# neighbours: (?, 32768, 8, 5):  (b, points, neigbours, features)

		input_final = tf.reshape(input_final, (-1, 16*128, 16, 11)) # (?, 16*128, 16, 11)


		x_conv = tf.layers.conv2d(input_final, 128, (1, 16), padding='VALID', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), bias_initializer=tf.zeros_initializer())
		x_conv = tf.layers.batch_normalization(x_conv, training=train_logical, momentum=0.99, epsilon=0.001, center=True, scale=True)
		x_conv = tf.nn.relu(x_conv)


		x = conv_layer(input_final, (1, 1), 16, train_logical, name="point-based_local_1")
		x = conv_layer(x, (1, 1), 32, train_logical, name="point-based_local_2")
		x_context = maxpool_layer(x, (1, 16), (1, 1), name="point-maxpool1")

		x = conv_layer(x, (1, 1), 64, train_logical, name="point-based_local_3")
		x = conv_layer(x, (1, 1), 128, train_logical, name="point-based_local_4")

		x_local = maxpool_layer(x, (1, 16), (1, 1), name="point-maxpool1")




		x_context_image = tf.reshape(x_context, (-1, 16, 128, 32))
		x_context_1 = tf.image.extract_image_patches(x_context_image, ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
		x_context_2 = tf.image.extract_image_patches(x_context_image, ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 2, 2, 1], padding='SAME')
		x_context_3 = tf.image.extract_image_patches(x_context_image, ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 3, 3, 1], padding='SAME')
		x_context_1 = tf.reshape(x_context_1, (-1, 16, 128, 9, 32))
		x_context_2 = tf.reshape(x_context_2, (-1, 16, 128, 9, 32))
		x_context_3 = tf.reshape(x_context_3, (-1, 16, 128, 9, 32))
		x_context_1 = tf.reshape(x_context_1, (-1, 16*128, 9, 32))
		x_context_2 = tf.reshape(x_context_2, (-1, 16*128, 9, 32))
		x_context_3 = tf.reshape(x_context_3, (-1, 16*128, 9, 32))
		x_context_1 = conv_layer(x_context_1, (1, 1), 64, train_logical, name="point-based_context_1")
		x_context_2 = conv_layer(x_context_2, (1, 1), 32, train_logical, name="point-based_context_2")
		x_context_3 = conv_layer(x_context_3, (1, 1), 32, train_logical, name="point-based_context_3")
		x_context_1 = maxpool_layer(x_context_1, (1, 9), (1, 1), name="point-maxpool1_cont")
		x_context_2 = maxpool_layer(x_context_2, (1, 9), (1, 1), name="point-maxpool2_cont")
		x_context_3 = maxpool_layer(x_context_3, (1, 9), (1, 1), name="point-maxpool3_cont")
		x_context_1 = tf.reshape(x_context_1, (-1, 16, 128, 64))
		x_context_2 = tf.reshape(x_context_2, (-1, 16, 128, 32))
		x_context_3 = tf.reshape(x_context_3, (-1, 16, 128, 32))

		x_context = tf.concat([x_context_3, x_context_2, x_context_1], -1)

		x_local = tf.reshape(x_local, (-1, 16, 128, 128))
		x_conv = tf.reshape(x_conv, (-1, 16, 128, 128))

		x = tf.concat([x_conv, x_local, x_context], -1)

		atten = tf.reduce_mean(x, [1, 2], keep_dims=True)
		atten = tf.layers.conv2d(atten, 128*3, (1, 1), padding='VALID', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), bias_initializer=tf.zeros_initializer())
		atten = tf.sigmoid(atten)
		x = tf.multiply(x, atten)
		x = conv_layer(x, (1, 1), 128, train_logical, name="last")


		x_branch = downsample(input_image, n_filters_out=128, is_training=train_logical)

		x = encoder_module(x, n_filters=128, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=128, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=128, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=128, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=128, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=128, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=128, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=128, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=128, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=128, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=128, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x2 = encoder_module(x, n_filters=128, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = downsample(x2, n_filters_out=128, is_training=train_logical)

		x = encoder_module_multi(x, n_filters=128, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module_multi(x, n_filters=128, is_training=train_logical, dilation=[2, 2], dropout=0.25)
		x = encoder_module_multi(x, n_filters=128, is_training=train_logical, dilation=[1, 4], dropout=0.25)
		x = encoder_module_multi(x, n_filters=128, is_training=train_logical, dilation=[2, 8], dropout=0.25)
		x = encoder_module_multi(x, n_filters=128, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module_multi(x, n_filters=128, is_training=train_logical, dilation=[2, 2], dropout=0.25)
		x = encoder_module_multi(x, n_filters=128, is_training=train_logical, dilation=[1, 4], dropout=0.25)
		x = encoder_module_multi(x, n_filters=128, is_training=train_logical, dilation=[2, 8], dropout=0.25)
		x = encoder_module_multi(x, n_filters=128, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module_multi(x, n_filters=128, is_training=train_logical, dilation=[2, 2], dropout=0.25)

		x = upsampling(x, 2)
		x = x + x2
		x = residual_separable(x, 128, train_logical, dropout=0, dilation=1)
		x = residual_separable(x, 128, train_logical, dropout=0, dilation=1) 
		x = upsampling(x, 2)
		x = x + x_branch

		x = residual_separable(x, 64, train_logical, dropout=0, dilation=1)
		x = upsampling(x, 2)
		x = conv_layer(x, (3, 3), CONFIG.N_CLASSES, train_logical, 'fully_connected1')

		y = tf.identity(x, name='y')

		return y



def mininet3dtiny(points, neighbors, train_logical, mask):

	with tf.variable_scope('net'):
		'''
		Calculate Relative features
		'''


		# neighbours: (?, 32768, 8, 5):  (b, points, neigbours, features)
		# points: (?, 32768, 1, 5):  (b, points, neigbours, features)
		# mask: (?, 64, 512, 1):  (b, h, w, 1)
		input_image = tf.reshape(points, (-1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, 5))
		# input_image: (?, 64, 512, 5):  (b, h, w, 5)


		point_groups = tf.image.extract_image_patches(input_image, ksizes=[1, 4, 4, 1], strides=[1, 4, 4, 1], rates=[1, 1, 1, 1], padding='SAME')
		point_groups = tf.reshape(point_groups, (-1, 16, 128, 16, 5)) # (b , h, w, , neighb, feat)
		mask_groups = tf.image.extract_image_patches(mask, ksizes=[1, 4, 4, 1], strides=[1, 4, 4, 1], rates=[1, 1, 1, 1], padding='SAME')
		mask_groups = tf.reshape(mask_groups, (-1, 16, 128, 16, 1)) # (b , h, w,  neighb, feat)
		# point_groups: (?, 16, 128, 16, 5):  (b , h, w, neigh, feat)



		# Get the mean point (taking apart non-valid points
		point_groups_sumF =tf.reduce_sum(point_groups, axis=3) #(b , h, w, feat)
		mask_groups_sumF = tf.reduce_sum(mask_groups, axis=3)
		point_groups_mean = point_groups_sumF / mask_groups_sumF
		# point_groups_mean: (?, 16, 128, 5):  (b , h, w, feat)

		point_groups_mean = tf.expand_dims(point_groups_mean, 3)
		point_groups_mean = tf.tile(point_groups_mean, [1, 1, 1, 16, 1])
		is_nan = tf.math.is_nan(point_groups_mean)
		point_groups_mean = tf.where(is_nan, x=tf.zeros_like(point_groups_mean), y=point_groups_mean)


		# substract mean point to points
		relative_points = point_groups - point_groups_mean

		mask_groups_tile = tf.tile(mask_groups, [1, 1, 1, 1, 5])
		relative_points = tf.where(tf.cast(mask_groups_tile, dtype=tf.bool), x=relative_points, y=tf.zeros_like(relative_points))



		# relative_points: (?, 16, 128, 16, 5):  (b , h, w,  neighb, feat)
		xyz_rel = relative_points[:, :, :, :, 0:3]
		relative_distance = tf.expand_dims(tf.norm(xyz_rel, ord='euclidean', axis=-1), axis=-1)
		input_final = tf.concat([point_groups, relative_points, relative_distance], axis=-1)
		# input_final: (?, 16, 128, 16, 11):  (b , h, w, neighb, feat)
		# neighbours: (?, 32768, 8, 5):  (b, points, neigbours, features)

		input_final = tf.reshape(input_final, (-1, 16*128, 16, 11)) # (?, 16*128, 16, 11)


		x_conv = tf.layers.conv2d(input_final, 96, (1, 16), padding='VALID', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), bias_initializer=tf.zeros_initializer())
		x_conv = tf.layers.batch_normalization(x_conv, training=train_logical, momentum=0.99, epsilon=0.001, center=True, scale=True)
		x_conv = tf.nn.relu(x_conv)


		x = conv_layer(input_final, (1, 1), 12, train_logical, name="point-based_local_1")
		x = conv_layer(x, (1, 1), 24, train_logical, name="point-based_local_2")
		x_context = maxpool_layer(x, (1, 16), (1, 1), name="point-maxpool1")

		x = conv_layer(x, (1, 1), 48, train_logical, name="point-based_local_3")
		x = conv_layer(x, (1, 1), 96, train_logical, name="point-based_local_4")

		x_local = maxpool_layer(x, (1, 16), (1, 1), name="point-maxpool1")




		x_context_image = tf.reshape(x_context, (-1, 16, 128, 24))
		x_context_1 = tf.image.extract_image_patches(x_context_image, ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
		x_context_2 = tf.image.extract_image_patches(x_context_image, ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 2, 2, 1], padding='SAME')
		x_context_3 = tf.image.extract_image_patches(x_context_image, ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 3, 3, 1], padding='SAME')
		x_context_1 = tf.reshape(x_context_1, (-1, 16, 128, 9, 24))
		x_context_2 = tf.reshape(x_context_2, (-1, 16, 128, 9, 24))
		x_context_3 = tf.reshape(x_context_3, (-1, 16, 128, 9, 24))
		x_context_1 = tf.reshape(x_context_1, (-1, 16*128, 9, 24))
		x_context_2 = tf.reshape(x_context_2, (-1, 16*128, 9, 24))
		x_context_3 = tf.reshape(x_context_3, (-1, 16*128, 9, 24))
		x_context_1 = conv_layer(x_context_1, (1, 1), 48, train_logical, name="point-based_context_1")
		x_context_2 = conv_layer(x_context_2, (1, 1), 24, train_logical, name="point-based_context_2")
		x_context_3 = conv_layer(x_context_3, (1, 1), 24, train_logical, name="point-based_context_3")
		x_context_1 = maxpool_layer(x_context_1, (1, 9), (1, 1), name="point-maxpool1_cont")
		x_context_2 = maxpool_layer(x_context_2, (1, 9), (1, 1), name="point-maxpool2_cont")
		x_context_3 = maxpool_layer(x_context_3, (1, 9), (1, 1), name="point-maxpool3_cont")
		x_context_1 = tf.reshape(x_context_1, (-1, 16, 128, 48))
		x_context_2 = tf.reshape(x_context_2, (-1, 16, 128, 24))
		x_context_3 = tf.reshape(x_context_3, (-1, 16, 128, 24))

		x_context = tf.concat([x_context_3, x_context_2, x_context_1], -1)

		x_local = tf.reshape(x_local, (-1, 16, 128, 96))
		x_conv = tf.reshape(x_conv, (-1, 16, 128, 96))

		x = tf.concat([x_conv, x_local, x_context], -1)

		atten = tf.reduce_mean(x, [1, 2], keep_dims=True)
		atten = tf.layers.conv2d(atten, 96*3, (1, 1), padding='VALID', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), bias_initializer=tf.zeros_initializer())
		atten = tf.sigmoid(atten)
		x = tf.multiply(x, atten)
		x = conv_layer(x, (1, 1), 96, train_logical, name="last")


		x_branch = downsample(input_image, n_filters_out=96, is_training=train_logical)

		x = encoder_module(x, n_filters=96, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=96, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=96, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=96, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=96, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module(x, n_filters=96, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x2 = encoder_module(x, n_filters=96, is_training=train_logical, dilation=[1, 1], dropout=0.25)

		x = downsample(x2, n_filters_out=96, is_training=train_logical)
		x = encoder_module_multi(x, n_filters=96, is_training=train_logical, dilation=[1, 1], dropout=0.25)
		x = encoder_module_multi(x, n_filters=96, is_training=train_logical, dilation=[2, 2], dropout=0.25)
		x = encoder_module_multi(x, n_filters=96, is_training=train_logical, dilation=[1, 4], dropout=0.25)
		x = encoder_module_multi(x, n_filters=96, is_training=train_logical, dilation=[2, 8], dropout=0.25)
		x = encoder_module_multi(x, n_filters=96, is_training=train_logical, dilation=[1, 1], dropout=0.25) 

		x = upsampling(x, 2)
		x = x + x2
		x = residual_separable(x, 96, train_logical, dropout=0, dilation=1)
		x = residual_separable(x, 96, train_logical, dropout=0, dilation=1) 
		x = upsampling(x, 2)
		x = x + x_branch

		x = residual_separable(x, 48, train_logical, dropout=0, dilation=1)
		x = upsampling(x, 2)

		x = conv_layer(x, (3, 3), CONFIG.N_CLASSES, train_logical, 'fully_connected1')

		y = tf.identity(x, name='y')

		return y



