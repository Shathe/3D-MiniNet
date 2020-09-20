import os
import tensorflow as tf
import numpy as np
import cv2
import time
import glob
import math
from augmentation import augmentation
from mininet3d import mininet3d
import sys
sys.path.append('./')
from settings import Settings
CONFIG = Settings()


############################## TFRECORD READER ##############################

# Takes a sequence of channels and returns the corresponding indices in the rangeimage
def seq_to_idx(seq):
	idx = []
	if "x" in seq:
		idx.append(0)
	if "y" in seq:
		idx.append(1)
	if "z" in seq:
		idx.append(2)
	if "r" in seq:
		idx.append(3)
	if "d" in seq:
		idx.append(4)

	return np.array(idx, dtype=np.intp)


# Loads a queue of random examples, and returns a batch iterator for each input
# and output
def read_example(filename, batch_size):

	# Open tfrecord
	reader = tf.TFRecordReader()
	filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
	_, serialized_example = reader.read(filename_queue)

	# Create random queue
	min_queue_examples = 500
	batch = tf.train.shuffle_batch([serialized_example], batch_size=batch_size, capacity=min_queue_examples+100*batch_size, min_after_dequeue=min_queue_examples, num_threads=2)

	# Read a batch
	parsed_example = tf.parse_example(batch,features={'neighbors': tf.FixedLenFeature([], tf.string),'points': tf.FixedLenFeature([], tf.string),'label': tf.FixedLenFeature([], tf.string)})

	# Decode point cloud
	idx = seq_to_idx(CONFIG.CHANNELS)

	points_raw = tf.decode_raw(parsed_example['points'], tf.float32)
	points = tf.reshape(points_raw, [batch_size, CONFIG.IMAGE_HEIGHT * CONFIG.IMAGE_WIDTH, 1 , 5])
	points = tf.gather(points, seq_to_idx(CONFIG.CHANNELS), axis=3)

	neighbors_raw = tf.decode_raw(parsed_example['neighbors'], tf.float32)
	neighbors = tf.reshape(neighbors_raw, [batch_size, CONFIG.IMAGE_HEIGHT * CONFIG.IMAGE_WIDTH, CONFIG.N_LEN, 5])
	neighbors = tf.gather(neighbors, seq_to_idx(CONFIG.CHANNELS), axis=3)

	# Decode label
	label_raw = tf.decode_raw(parsed_example['label'], tf.float32)
	label = tf.reshape(label_raw, [batch_size, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.N_CLASSES + 2])

	return points, neighbors, label





############################ NETWORK LOSS ############################

# Returns slices of a tensor
def slice_tensor(x, start, end=None):
	if end < 0:
		y = x[...,start:]
	else:
		if end is None:
			end = start
		y = x[...,start:end + 1]

	return y


def u_net_loss(pred, label):

	with tf.variable_scope('loss'):
		# Retrieve mask on last channel of the label
		mask = slice_tensor(label, CONFIG.N_CLASSES + 1, -1)
		dist = slice_tensor(label, CONFIG.N_CLASSES, CONFIG.N_CLASSES)
		label  = slice_tensor(label, 0, CONFIG.N_CLASSES - 1)

		weight_norm = 2.0 * 3.0 ** 2.0

		weights_ce = 0.1 + 1.0 * tf.exp(- dist / weight_norm)
		weights_ce = weights_ce * mask

		# Compute the cross entropy
		if CONFIG.FOCAL_LOSS:
			with tf.name_scope('focal_loss'):
				epsilon = 1.e-9
				gamma   = 2.
				pred_softmax  = tf.nn.softmax(pred)
				cross_entropy = tf.multiply(label, -tf.log(pred_softmax))
				weights_fl    = tf.multiply(label, tf.pow(tf.subtract(1., pred_softmax), gamma))
				weigths_total = weights_fl * weights_ce
				loss          = tf.reduce_sum(weights_ce * weights_fl * cross_entropy) / tf.reduce_sum(weights_ce)

				tf.summary.scalar("global_loss", loss)
		else:
			with tf.name_scope('loss'):

				cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=pred, name="cross_entropy")
				loss          = tf.reduce_sum(tf.reshape(cross_entropy, [CONFIG.BATCH_SIZE, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, 1]) * weights_ce) / tf.reduce_sum(weights_ce)

				tf.summary.scalar("global_loss", loss)

		# Compute average precision
		with tf.name_scope('average_precision'):
			softmax_pred = tf.nn.softmax(pred)

			argmax_pred = tf.math.argmax(softmax_pred, axis=3)
			mask_bin = tf.squeeze(tf.math.greater(mask, 0))
			for c in range(1, CONFIG.N_CLASSES):
				p = tf.math.equal(argmax_pred, c)
				l = tf.squeeze(tf.math.equal(slice_tensor(label, c, c), 1.0))

				intersection = tf.logical_and(p, l)
				union        = tf.logical_or(p, l)

				iou = tf.reduce_sum(tf.cast(tf.logical_and(intersection, mask_bin), tf.float32)) / (tf.reduce_sum(tf.cast(tf.logical_and(union, mask_bin), tf.float32)) + 0.00000001)
				tf.summary.scalar("iou_class_0" + str(c), iou)


	# Display prediction and groundtruth for object of class 1
	with tf.variable_scope('predictions'):
		for i in range(0, CONFIG.N_CLASSES):
			tf.summary.image('pred_class_0' + str(i),
						 	tf.reshape(tf.transpose(slice_tensor(pred, i, i) * mask, perm=[0,3,1,2]), (CONFIG.BATCH_SIZE, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, 1)),
						 	max_outputs=CONFIG.BATCH_SIZE)


	with tf.variable_scope('labels'):
		for i in range(0, CONFIG.N_CLASSES):
			tf.summary.image('label_class_0' + str(i),
				 			tf.reshape(tf.transpose(slice_tensor(label, i, i) * mask, perm=[0,3,1,2]), (CONFIG.BATCH_SIZE, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, 1)),
				 			max_outputs=CONFIG.BATCH_SIZE)


	return loss




############################ TRAINING MANAGER ############################

# Displays configuration
def print_config():
	print("\n----------- RIU-NET CONFIGURATION -----------")
	print("input channels     : {}".format(CONFIG.CHANNELS.upper()))
	print("input dims         : {}x{}x{}".format(CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.IMAGE_DEPTH))
	print("pointnet embeddings: {}".format("yes" if CONFIG.POINTNET == True else "no"))
	print("focal loss         : {}".format("yes" if CONFIG.FOCAL_LOSS == True else "no"))
	print("# of parameters    : {}".format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
	print("---------------------------------------------\n")
# Compute the average example processing time
def time_to_speed(batch_time, batch_size):
	return round(float(batch_size) / batch_time, 2)

# Pretty obvious
def make_dir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

# Returns last saved checkpoint index
def get_last_checkpoint(output_model):
	files = glob.glob(output_model + "-*.index")
	checkpoints = [-1]
	for checkpoint in files:
		checkpoint = checkpoint.replace(output_model + "-", "")
		checkpoint = checkpoint.replace(".index", "")
		checkpoints.append(int(checkpoint))

	return max(checkpoints)

# Computes current learning rate given the decay settings
def get_learning_rate(iteration, start_rate, decay_interval, decay_value):
	rate = start_rate * (decay_value ** math.floor(iteration / decay_interval))
	return rate


# Training routine
def train():
	# Remove deprecated messages
	tf.logging.set_verbosity(tf.logging.ERROR)

	# Reading TFRecords
	with tf.name_scope('train_batch'):
		batch_points, batch_neighbors, batch_label = read_example(CONFIG.TFRECORD_TRAIN, CONFIG.BATCH_SIZE)
	with tf.name_scope('val_batch'):
		val_points, val_neighbors, val_label = read_example(CONFIG.TFRECORD_VAL, CONFIG.BATCH_SIZE)
	with tf.name_scope('learning_rate'):
		learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate_placeholder")

	# Save learning rate
	tf.summary.scalar("learning_rate", learning_rate, family="learning_rate")


	# Creating input placeholder
	points    = tf.placeholder(shape = [None, CONFIG.IMAGE_HEIGHT * CONFIG.IMAGE_WIDTH, 1, CONFIG.IMAGE_DEPTH], dtype=tf.float32, name='points_placeholder')
	neighbors = tf.placeholder(shape = [None, CONFIG.IMAGE_HEIGHT * CONFIG.IMAGE_WIDTH, CONFIG.N_LEN, CONFIG.IMAGE_DEPTH], dtype=tf.float32, name='neighbors_placeholder')
	label     = tf.placeholder(shape = [None, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.N_CLASSES+2], dtype=tf.float32, name='label_placeholder')

	train_flag = tf.placeholder(dtype=tf.bool, name='flag_placeholder')


	# Create Network and Loss operator
	mask = slice_tensor(label, CONFIG.N_CLASSES + 1, -1)

	y    = mininet3d(points, neighbors, train_flag, mask)
	loss = u_net_loss(y, label)


	# Creating optimizer
	opt = tf.train.AdamOptimizer(learning_rate=CONFIG.LEARNING_RATE)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_step = opt.minimize(loss)

	# Merging summaries
	summary_op = tf.summary.merge_all()

	# Saver for checkpoints
	saver = tf.train.Saver(max_to_keep=1000)

	# Starting session
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	# Check if a checkpoint already exists and load it
	start_step = 0
	if get_last_checkpoint(CONFIG.OUTPUT_MODEL) >= 0:
		start_step = get_last_checkpoint(CONFIG.OUTPUT_MODEL)
		print("[From checkpoint] Restoring checkpoint {}".format(start_step))
		saver.restore(sess, CONFIG.OUTPUT_MODEL + "-" + str(start_step))


	# Creating threads for batch sampling
	coord  = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	# Creating output logs
	train_summary_writer = tf.summary.FileWriter(CONFIG.OUTPUT_LOGS + "train/", tf.get_default_graph())
	val_summary_writer   = tf.summary.FileWriter(CONFIG.OUTPUT_LOGS + "val/", tf.get_default_graph())

	# Display config
	print_config()

	# Starting iterations
	for i in range(start_step, CONFIG.NUM_ITERS):

		# Retrieving current batch
		points_data, neighbors_data, label_data = sess.run([batch_points, batch_neighbors, batch_label])

		# points_data, label_data = augmentation(points_data, label_data)

		# Computing learning rate
		rate = get_learning_rate(i, CONFIG.LEARNING_RATE, CONFIG.LR_DECAY_INTERVAL, CONFIG.LR_DECAY_VALUE)

		# Training Network on the current batch
		t = time.time()
		_, loss_data, data = sess.run([train_step, loss, y], feed_dict={train_flag: True, points: points_data, neighbors: neighbors_data, label: label_data, learning_rate: rate})

		print('[Training] Iteration: {}, loss: {}, {} e/s, {}lr '.format(int(i), loss_data, time_to_speed(time.time() - t, CONFIG.BATCH_SIZE), rate))

		# If summary has to be saved
		if i % 100 == 0:
			summary_str = sess.run(summary_op, feed_dict={train_flag: True, points: points_data, neighbors: neighbors_data, label: label_data, learning_rate: rate})
			train_summary_writer.add_summary(summary_str, i)
		# If checkpoint has to be saved
		if (i + 1) % CONFIG.SAVE_INTERVAL == 0:
			saver.save(sess, CONFIG.OUTPUT_MODEL, global_step=i+1)
		# If validation should be done
		if i % CONFIG.VAL_INTERVAL == 0:
			# Retrieving validation batch
			points_data, neighbors_data, label_data = sess.run([val_points, val_neighbors, val_label])

			# Running summaries and saving
			summary_str = sess.run(summary_op, feed_dict={train_flag: False, points: points_data, neighbors: neighbors_data, label: label_data, learning_rate: rate})
			val_summary_writer.add_summary(summary_str, i)

			print("[Validation] Done on one batch")

	# Saving final network configuration
	saver.save(sess, CONFIG.OUTPUT_MODEL, global_step=i+1)

if __name__ == "__main__":
	train()
