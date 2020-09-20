
import random
from scipy import misc,ndimage
import cv2
import scipy.misc
import numpy as np

def crop_center(img_np, cropx, cropy):
  y, x = img_np.shape
  startx = x // 2 - (cropx // 2)
  starty = y // 2 - (cropy // 2)
  return img_np[starty:starty + cropy, startx:startx + cropx]



def augmentation(points_data, label_data):
	"""
	All of this augmentation is done in the 2D space, not on the 3D space.
	"""

	for b in range(points_data.shape[0]):
		# Read
		point_i = points_data[b, ...]
		label_i = label_data[b, ...]
		x = point_i[:, :, 0]
		y = point_i[:, :, 1]
		z = point_i[:, :, 2]
		remission = point_i[:, :, 3]
		depth = point_i[:, :, 4]

		x = np.reshape(x, (64, 512))
		y = np.reshape(y, (64, 512))
		z = np.reshape(z, (64, 512))
		remission = np.reshape(remission, (64, 512))
		depth = np.reshape(depth, (64, 512))

		label1 = label_i[:, :, 0]
		label2 = label_i[:, :, 1]
		label3 = label_i[:, :, 2]
		label4 = label_i[:, :, 3]
		label5 = label_i[:, :, 4]
		label6 = label_i[:, :, 5]


		# Augmentation 50% times
		aug = random.random() > 0.5

		if random.random() > 0.5 and aug: # Random flipping
			x = np.fliplr(x)
			y = np.fliplr(y)
			# when you flip left_to_right, change the sign of the axis
			y = - y
			z = np.fliplr(z)
			depth = np.fliplr(depth)
			remission = np.fliplr(remission)
			label1 = np.fliplr(label1)
			label2= np.fliplr(label2)
			label3 = np.fliplr(label3)
			label4= np.fliplr(label4)
			label5= np.fliplr(label5)
			label6 = np.fliplr(label6)


		if random.random() > 0.5 and aug: # Random shifts
			x_shift= random.randint(0, 512) - 256
			y_shift = random.randint(0, 24) - 12
			x = ndimage.shift(x, (y_shift, x_shift), None, order=0, mode='constant', cval=0)
			y = ndimage.shift(y, (y_shift, x_shift), None, order=0, mode='constant', cval=0)
			z = ndimage.shift(z, (y_shift, x_shift), None, order=0, mode='constant', cval=0)
			depth = ndimage.shift(depth, (y_shift, x_shift), None, order=0, mode='constant', cval=0)
			remission = ndimage.shift(remission, (y_shift, x_shift), None, order=0, mode='constant', cval=0)
			label1 = ndimage.shift(label1, (y_shift, x_shift), None, order=0, mode='constant', cval=0)
			label2 = ndimage.shift(label2, (y_shift, x_shift), None, order=0, mode='constant', cval=0)
			label3 = ndimage.shift(label3, (y_shift, x_shift), None, order=0, mode='constant', cval=0)
			label5 = ndimage.shift(label5, (y_shift, x_shift), None, order=0, mode='constant', cval=0)
			label4 = ndimage.shift(label4, (y_shift, x_shift), None, order=0, mode='constant', cval=0)
			label6 = ndimage.shift(label6, (y_shift, x_shift), None, order=0, mode='constant', cval=0)

		if random.random() > 0.5 and aug: # Random zooms (in/out)
			zoom = random.random() + 1
			if random.random() > 0.5:
				zoom = 1. / zoom

			x = ndimage.zoom(x, zoom, output=None, order=0, mode='constant', cval=0.0)
			y = ndimage.zoom(y, zoom, output=None, order=0, mode='constant', cval=0.0)
			z = ndimage.zoom(z, zoom, output=None, order=0, mode='constant', cval=0.0)
			depth = ndimage.zoom(depth, zoom, output=None, order=0, mode='constant', cval=0.0)
			remission = ndimage.zoom(remission, zoom, output=None, order=0, mode='constant', cval=0.0)
			label1 = ndimage.zoom(label1, zoom, output=None, order=0, mode='constant', cval=0.0)
			label2 = ndimage.zoom(label2, zoom, output=None, order=0, mode='constant', cval=0.0)
			label3 = ndimage.zoom(label3, zoom, output=None, order=0, mode='constant', cval=0.0)
			label4 = ndimage.zoom(label4, zoom, output=None, order=0, mode='constant', cval=0.0)
			label5 = ndimage.zoom(label5, zoom, output=None, order=0, mode='constant', cval=0.0)
			label6 = ndimage.zoom(label6, zoom, output=None, order=0, mode='constant', cval=0.0)

			if zoom > 1:

				x = crop_center(x, 512, 64)
				y = crop_center(y, 512, 64)
				z = crop_center(z, 512, 64)
				depth = crop_center(depth, 512, 64)
				remission = crop_center(remission, 512, 64)
				label1 = crop_center(label1, 512, 64)
				label2 = crop_center(label2, 512, 64)
				label3 = crop_center(label3, 512, 64)
				label4 = crop_center(label4, 512, 64)
				label5 = crop_center(label5, 512, 64)
				label6 = crop_center(label6, 512, 64)

			else:
				x = cv2.resize(x, dsize=(512, 64), interpolation=cv2.INTER_NEAREST)
				y = cv2.resize(y, dsize=(512, 64), interpolation=cv2.INTER_NEAREST)
				z = cv2.resize(z, dsize=(512, 64), interpolation=cv2.INTER_NEAREST)
				depth = cv2.resize(depth, dsize=(512, 64), interpolation=cv2.INTER_NEAREST)
				remission = cv2.resize(remission, dsize=(512, 64), interpolation=cv2.INTER_NEAREST)
				label1 = cv2.resize(label1, dsize=(512, 64), interpolation=cv2.INTER_NEAREST)
				label2 = cv2.resize(label2, dsize=(512, 64), interpolation=cv2.INTER_NEAREST)
				label3 = cv2.resize(label3, dsize=(512, 64), interpolation=cv2.INTER_NEAREST)
				label4 = cv2.resize(label4, dsize=(512, 64), interpolation=cv2.INTER_NEAREST)
				label5 = cv2.resize(label5, dsize=(512, 64), interpolation=cv2.INTER_NEAREST)
				label6 = cv2.resize(label6, dsize=(512, 64), interpolation=cv2.INTER_NEAREST)



		label_data[b, :,:,0]= label1
		label_data[b, :,:,1] = label2
		label_data[b, :,:,2] = label3
		label_data[b, :,:,3]= label4
		label_data[b, :,:,4] = label5
		label_data[b, :,:,5] = label6


		points_data[b, :,:, 4] =  np.reshape(depth, (32768, 1))
		points_data[b, :,:, 3] =np.reshape(remission, (32768, 1))
		points_data[b, :,:, 2] =  np.reshape(z, (32768, 1))
		points_data[b, :,:,1] = np.reshape(y, (32768, 1))
		points_data[b, :,:, 0] =  np.reshape(x, (32768, 1))

	return points_data, label_data