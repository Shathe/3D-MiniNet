import sys
import numpy as np
import cv2
import os
import math
from scipy import interpolate

def lindepth_to_mask(depth_linear, img_height, img_width):
	return np.reshape(depth_linear, (img_height, img_width, 1)) > 0


def clip_normalize(data, interval, log_transformed=False):
	data_clip = np.clip(data, interval[0], interval[1])
	if log_transformed:
		#return (np.log(data_clip) - np.log(interval[0])) / (np.log(interval[1]) - np.log(interval[0]))
		return data_clip
	else:
		return (data_clip - interval[0]) / (interval[1] - interval[0])


def clip_mask_normalize(data, mask, interval, log_transformed=False):
	outerval = np.logical_and(data < interval[1], data > interval[0])
	mask = np.logical_and(mask, outerval)
	data_clip = np.clip(data, interval[0], interval[1])

	if log_transformed:
		#return (np.log(data_clip) - np.log(interval[0])) / (np.log(interval[1]) - np.log(interval[0])), mask
		return np.log(data_clip), mask
	else:
		return (data_clip - interval[0]) / (interval[1] - interval[0]), mask


def fill_sky(data, mask, new_val):
	ret, labels = cv2.connectedComponents(np.asarray(mask == 0).astype(np.uint8))
	sky_label = labels[0, math.floor(mask.shape[1] / 2)]

	cv2.imwrite("./validation/test.png", labels)
	for c in range(data.shape[2]):
		data[:,:,c] = np.where(labels == sky_label, new_val, data[:,:,c])

	return data



def apply_mask(data, mask):
	tmp = np.zeros((data.shape[0], data.shape[1]))

	if len(data.shape) == 2:
		data[np.squeeze(mask)] == 0
	else:
		for c in range(data.shape[2]):
			data[:,:,c] = np.where(np.squeeze(mask) == 1, data[:,:,c], tmp)

	return data


def ri_to_depth_height_mask(ri, depth_clip, height_clip):
	mask = ri[:,:,0] > 0

	depth, mask = clip_mask_normalize(np.sqrt(ri[:,:,0]**2 + ri[:,:,1]**2), mask, depth_clip, log_transformed = True)

	height, mask = clip_mask_normalize(ri[:,:,2], mask, height_clip)

	img = apply_mask(np.dstack((depth, height)).astype(np.float32), mask)

	mask = mask

	return img, mask

def ri_to_depth_height_intensity_mask(ri, depth_clip, height_clip):
	mask = ri[:,:,0] > 0

	depth, mask = clip_mask_normalize(np.sqrt(ri[:,:,0]**2 + ri[:,:,1]**2), mask, depth_clip, log_transformed = True)

	height, mask = clip_mask_normalize(ri[:,:,2], mask, height_clip)

	ref = ri[:,:,3]

	img = apply_mask(np.dstack((depth, height, ref)).astype(np.float32), mask)

	mask = mask

	return img, mask

def ri_to_depth_height_intensity_mask_noclip(ri, depth_clip, height_clip):
	mask = ri[:,:,0] > 0

	depth = np.sqrt(ri[:,:,0]**2 + ri[:,:,1]**2)

	height = ri[:,:,2]

	ref = ri[:,:,3]

	img = apply_mask(np.dstack((depth, height, ref)).astype(np.float32), mask)

	mask = mask

	return img, mask

def ri_to_depth_height_mask_noclip(ri):
	mask = ri[:,:,0] > 0

	depth = np.sqrt(ri[:,:,0]**2 + ri[:,:,1]**2)

	height = ri[:,:,2]

	img = apply_mask(np.dstack((depth, height)).astype(np.float32), mask)

	mask = mask

	return img, mask

def ri_to_xyz_mask(ri):
	mask = ri[:,:,0] > 0

	img = ri[:,:,0:3]

	mask = mask

	return img, mask

def ri_to_xyz_intensity_depth_mask(ri):
	mask = ri[:,:,0] > 0

	img = ri[:,:,0:5]

	mask = mask

	return img, mask


def interp_data(d, mask):
	interp_output = np.zeros(d.shape)
	x = np.arange(0, d.shape[1])
	y = np.arange(0, d.shape[0])

	xx, yy = np.meshgrid(x, y)

	x1 = xx[mask]
	y1 = yy[mask]
	for c in range(d.shape[2]):
		newarr = d[:,:,c]
		newarr = newarr[mask]
		interp_output[:,:,c] = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='nearest')

	return interp_output


def pointnetize(groundtruth, n_size=[3, 3]):
	y_offset    = int(math.floor(n_size[0] / 2))
	x_offset    = int(math.floor(n_size[1] / 2))
	n_len       = n_size[0] * n_size[1]
	mid_offset  = int(math.floor(n_len / 2))
	n_indices   = np.delete(np.arange(n_len), mid_offset) 

	groundtruth_pad = np.pad(groundtruth, ((y_offset, y_offset),(x_offset, x_offset), (0, 0)), "symmetric")

	n_output = np.zeros((groundtruth.shape[0], groundtruth.shape[1], n_len - 1, groundtruth.shape[2]))
	p_output = np.zeros((groundtruth.shape[0], groundtruth.shape[1], 1, groundtruth.shape[2]))

	for y in range(0, groundtruth.shape[0]):
		for x in range(0, groundtruth.shape[1]):
			patch = groundtruth_pad[y:y+n_size[0], x:x+n_size[1],:]
			lin_patch = np.reshape(patch, (n_len, -1))

			if lin_patch[mid_offset,0] != 0: # If center pixel is not empty
				p = lin_patch[mid_offset, :]
				n = lin_patch[n_indices, :]

				mask_filled = n[:,0] != 0

				n[mask_filled, 0:3] = n[mask_filled, 0:3]# - p[0:3] # Defined points in local coordinates

				n_output[y,x,:,:] = n
				p_output[y,x,:,:] = p

	return p_output, n_output


def gt_to_label(groundtruth, mask, n_classes):

	# Compute weigthed mask
	contours = np.zeros((mask.shape[0], mask.shape[1]), dtype=bool)

	if np.amax(groundtruth) > n_classes-1:
		print("[WARNING] There are more classes than expected !")

	for c in range(1, int(np.amax(groundtruth))+1):
		channel = (groundtruth == c).astype(np.float32)
		gt_dilate = cv2.dilate(channel, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
		gt_dilate = gt_dilate - channel
		contours = np.logical_or(contours, gt_dilate == 1.0)

	contours = contours.astype(np.float32) * mask

	dist = cv2.distanceTransform((1 - contours).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

	weight_map = 0.1 + 1.0 * np.exp(- dist / (2.0 * 3.0**2.0))
	weight_map = weight_map * mask

	# Create output label for training
	label = np.zeros((groundtruth.shape[0], groundtruth.shape[1], n_classes + 1))
	for y in range(groundtruth.shape[0]):
		for x in range(groundtruth.shape[1]):
			label[y, x, int(groundtruth[y, x])] = 1.0

	label[:,:,n_classes] = weight_map

	return label
