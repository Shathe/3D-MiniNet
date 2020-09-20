import sys
import tensorflow as tf
import numpy as np
import cv2
import csv
import os
import glob
import math
import matplotlib.pyplot as plt


# SQUEEZESEG DATASET
# Channels description:
# 0: X
# 1: Y
# 2: Z
# 3: REFLECTANCE
# 4: DEPTH
# 5: LABEL


# imports Settings manager
sys.path.append('./')
import data_loader
from settings import Settings
CONFIG = Settings(required_args=["config"])

semantic_base = "lidar_2d/"


# Generates tfrecords for training
def make_tfrecord():

	# Creates each tfrecord (train and val)
	for dataset in ["train", "val"]:

		# Get path
		dataset_output = CONFIG.TFRECORD_TRAIN if dataset == "train" else CONFIG.TFRECORD_VAL

		with tf.python_io.TFRecordWriter(dataset_output) as writer:

			file_list_name = open(dataset_output + ".txt", "w")

			if dataset == "val":
				file_list = open("./data/semantic_val.txt","r")
			else:
				file_list = open("./data/semantic_train.txt", "r")

			# Going through each example
			line_num = 1
			for file in file_list:

				augmentation_list = ["normal"] if dataset is "val" else CONFIG.AUGMENTATION

				# Augmentation settings
				for aug_type in augmentation_list:

					print("[{}] >> Processing file \"{}\" ({}), with augmentation : {}".format(dataset, file[:-1], line_num, aug_type))

					# Load labels
					data = np.load(semantic_base + file[:-1] + ".npy")

					mask = data[:,:,0] != 0

					#data = data_loader.interp_data(data[:,:,0:5], mask)

					p, n = data_loader.pointnetize(data[:,:,0:5], n_size=CONFIG.N_SIZE)
					groundtruth = data_loader.apply_mask(data[:,:,5], mask)

					# Compute weigthed mask
					contours = np.zeros((mask.shape[0], mask.shape[1]), dtype=bool)

					if np.amax(groundtruth) > CONFIG.N_CLASSES-1:
						print("[WARNING] There are more classes than expected !")

					for c in range(1, int(np.amax(groundtruth))+1):
						channel = (groundtruth == c).astype(np.float32)
						gt_dilate = cv2.dilate(channel, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
						gt_dilate = gt_dilate - channel
						contours = np.logical_or(contours, gt_dilate == 1.0)

					contours = contours.astype(np.float32) * mask

					dist = cv2.distanceTransform((1 - contours).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

					# Create output label for training
					label = np.zeros((groundtruth.shape[0], groundtruth.shape[1], CONFIG.N_CLASSES + 2))
					for y in range(groundtruth.shape[0]):
						for x in range(groundtruth.shape[1]):
							label[y, x, int(groundtruth[y, x])] = 1.0

					label[:,:,CONFIG.N_CLASSES]   = dist
					label[:,:,CONFIG.N_CLASSES+1] = mask

					# Serialize example
					n_raw = n.astype(np.float32).tostring()
					p_raw = p.astype(np.float32).tostring()
					label_raw = label.astype(np.float32).tostring()

					# Create tf.Example
					example = tf.train.Example(features=tf.train.Features(feature={
							'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
							'neighbors': tf.train.Feature(bytes_list=tf.train.BytesList(value=[n_raw])),
							'points': tf.train.Feature(bytes_list=tf.train.BytesList(value=[p_raw]))}))

					# Adding Example to tfrecord
					writer.write(example.SerializeToString())

					file_list_name.write(semantic_base + file[:-1] + ".npy\n")

				line_num += 1

			print("Process finished, stored {} entries in \"{}\"".format(line_num-1, dataset_output))

			file_list_name.close()

	print("All files created.")

if __name__ == "__main__":
	make_tfrecord()
