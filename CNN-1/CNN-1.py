import glob
import math
import os
import random
import time
from datetime import timedelta

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle


def load_train(train_path, image_size, classes):
	images = []
	labels = []
	ids = []
	cls = []

	print('Reading training images')
	for fld in classes:  # assuming data directory has a separate folder for each class, and that each folder is named after the class
		index = classes.index(fld)
		print('Loading {} files (Index: {})'.format(fld, index))
		path = os.path.join(train_path, fld, '*g')
		files = glob.glob(path)
		for fl in files:
			image = cv2.imread(fl)
			image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
			images.append(image)
			label = np.zeros(len(classes))
			label[index] = 1.0
			labels.append(label)
			flbase = os.path.basename(fl)
			ids.append(flbase)
			cls.append(fld)
	images = np.array(images)
	labels = np.array(labels)
	ids = np.array(ids)
	cls = np.array(cls)

	return images, labels, ids, cls


def load_test(test_path, image_size):
	path = os.path.join(test_path, '*g')
	files = sorted(glob.glob(path))

	X_test = []
	X_test_id = []
	print("Reading test images")
	for fl in files:
		flbase = os.path.basename(fl)
		img = cv2.imread(fl)
		img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
		X_test.append(img)
		X_test_id.append(flbase)

	### because we're not creating a DataSet object for the test images, normalization happens here
	X_test = np.array(X_test, dtype=np.uint8)
	X_test = X_test.astype('float32')
	X_test = X_test / 255

	return X_test, X_test_id


class DataSet(object):

	def __init__(self, images, labels, ids, cls):
		"""Construct a DataSet. one_hot arg is used only if fake_data is true."""

		self._num_examples = images.shape[0]

		# Convert shape from [num examples, rows, columns, depth]
		# to [num examples, rows*columns] (assuming depth == 1)
		# Convert from [0, 255] -> [0.0, 1.0].

		images = images.astype(np.float32)
		images = np.multiply(images, 1.0 / 255.0)

		self._images = images
		self._labels = labels
		self._ids = ids
		self._cls = cls
		self._epochs_completed = 0
		self._index_in_epoch = 0

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def ids(self):
		return self._ids

	@property
	def cls(self):
		return self._cls

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size):
		"""Return the next `batch_size` examples from this data set."""
		start = self._index_in_epoch
		self._index_in_epoch += batch_size


if self._index_in_epoch > self._num_examples:
	# Finished epoch
	self._epochs_completed += 1

	# # Shuffle the data (maybe)
	# perm = np.arange(self._num_examples)
	# np.random.shuffle(perm)
	# self._images = self._images[perm]
	# self._labels = self._labels[perm]
	# Start next epoch

	start = 0
	self._index_in_epoch = batch_size
	assert batch_size <= self._num_examples
end = self._index_in_epoch

return self._images[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size=0):
	class DataSets(object):
		pass

	data_sets = DataSets()

	images, labels, ids, cls = load_train(train_path, image_size, classes)
	images, labels, ids, cls = shuffle(images, labels, ids, cls)  # shuffle the data

	if isinstance(validation_size, float):
		validation_size = int(validation_size * images.shape[0])

		validation_images = images[:validation_size]
		validation_labels = labels[:validation_size]
		validation_ids = ids[:validation_size]
		validation_cls = cls[:validation_size]

		train_images = images[validation_size:]
		train_labels = labels[validation_size:]
		train_ids = ids[validation_size:]
		train_cls = cls[validation_size:]

		data_sets.train = DataSet(train_images, train_labels, train_ids, train_cls)
		data_sets.valid = DataSet(validation_images, validation_labels, validation_ids, validation_cls)

	return data_sets

	def read_test_set(test_path, image_size):
		images, ids = load_test(test_path, image_size)
		return images, ids

	# Convolutional Layer 1.
	filter_size1 = 5
	num_filters1 = 64

	# Convolutional Layer 2.
	filter_size2 = 3
	num_filters2 = 64

	# # Convolutional Layer 3.
	# filter_size3 = 5
	# num_filters3 = 128

	# Fully-connected layer 1.
	fc1_size = 128  # Number of neurons in fully-connected layer.

	# Fully-connected layer 2.
	fc2_size = 128  # Number of neurons in fully-connected layer.

	# Number of color channels for the images: 1 channel for gray-scale.
	num_channels = 3

	# image dimensions (only squares for now)
	img_size = 64

	# Size of image when flattened to a single dimension
	img_size_flat = img_size * img_size * num_channels

	# Tuple with height and width of images used to reshape arrays.
	img_shape = (img_size, img_size)

	# class info
	classes = ['Sphynx', 'Siamese', 'Ragdoll',
	           'Persian', 'Maine_Coon', 'British_shorthair', 'Bombay', 'Birman', 'Bengal', 'Abyssinian']
	# classes = ['Sphynx','Siamese',
	#            'Persian','Maine_Coon','British_shorthair']

	num_classes = len(classes)

	# batch size
	batch_size = 32

	# validation split
	validation_size = .2

	# how long to wait after validation loss stops improving before terminating training
	early_stopping = None  # use None if you don't want to implement early stoping

	train_path = 'dataset'
	# test_path = 'test'
	checkpoint_dir = "ckpoint"
	# load training dataset
	data = read_train_sets(train_path, img_size, classes, validation_size=validation_size)
	# test_images, test_ids = read_test_set(test_path, img_size)
	print("Size of:")
	print("- Training-set:\t\t{}".format(len(data.train.labels)))
	# print("- Test-set:\t\t{}".format(len(test_images)))
	print("- Validation:\t{}".format(len(data.valid.labels)))

	# print(images)
	def plot_images(images, cls_true, cls_pred=None):

		if len(images) == 0:
			print("no images to show")
			return
		else:
			random_indices = random.sample(range(len(images)), min(len(images), 9))

		images, cls_true = zip(*[(images[i], cls_true[i]) for i in random_indices])

		# Create figure with 3x3 sub-plots.
		fig, axes = plt.subplots(3, 3)
		fig.subplots_adjust(hspace=0.3, wspace=0.3)

		for i, ax in enumerate(axes.flat):
			# Plot image.
			ax.imshow(images[i].reshape(img_size, img_size, num_channels))

			# Show true and predicted classes.
			if cls_pred is None:
				xlabel = "True: {0}".format(cls_true[i])
			else:
				xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

			# Show the classes as the label on the x-axis.
			ax.set_xlabel(xlabel)
			# Remove ticks from the plot.
			ax.set_xticks([])
			ax.set_yticks([])

			# Ensure the plot is shown correctly with multiple plots
			# in a single Notebook cell.
		plt.show()
		images, cls_true = data.train.images, data.train.cls

		# Plot the images and labels using our helper-function above.
		plot_images(images=images, cls_true=cls_true)

		def new_weights(shape):
			return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

		def new_biases(length):
			return tf.Variable(tf.constant(0.05, shape=[length]))

		def new_conv_layer(input,  # The previous layer.
		                   num_input_channels,  # Num. channels in prev. layer.
		                   filter_size,  # Width and height of each filter.
		                   num_filters,  # Number of filters.
		                   use_pooling=True):  # Use 2x2 max-pooling.

			# Shape of the filter-weights for the convolution.
			# This format is determined by the TensorFlow API.
			shape = [filter_size, filter_size, num_input_channels, num_filters]

			# Create new weights aka. filters with the given shape.
			weights = new_weights(shape=shape)

			# Create new biases, one for each filter.
			biases = new_biases(length=num_filters)

			# Create the TensorFlow operation for convolution.
			# Note the strides are set to 1 in all dimensions.
			# The first and last stride must always be 1,
			# because the first is for the image-number and
			# the last is for the input-channel.
			# But e.g. strides=[1, 2, 2, 1] would mean that the filter
			# is moved 2 pixels across the x- and y-axis of the image.
			# The padding is set to 'SAME' which means the input image
			# is padded with zeroes so the size of the output is the same.
			layer = tf.nn.conv2d(input=input,
			                     filter=weights,
			                     strides=[1, 1, 1, 1],
			                     padding='SAME')
			# Add the biases to the results of the convolution.
			# A bias-value is added to each filter-channel.
			layer += biases

			# Use pooling to down-sample the image resolution?
			if use_pooling:
				# This is 2x2 max-pooling, which means that we
				# consider 2x2 windows and select the largest value
				# in each window. Then we move 2 pixels to the next window.
				layer = tf.nn.max_pool(value=layer,
				                       ksize=[1, 2, 2, 1],
				                       strides=[1, 2, 2, 1],
				                       padding='SAME')

			# Rectified Linear Unit (ReLU).
			# It calculates max(x, 0) for each input pixel x.
			# This adds some non-linearity to the formula and allows us
			# to learn more complicated functions.
			layer = tf.nn.relu(layer)

			# Note that ReLU is normally executed before the pooling,
			# but since relu(max_pool(x)) == max_pool(relu(x)) we can
			# save 75% of the relu-operations by max-pooling first.

			# We return both the resulting layer and the filter-weights
			# because we will plot the weights later.
			return layer, weights

		def flatten_layer(layer):
			# Get the shape of the input layer.
			layer_shape = layer.get_shape()

			# The shape of the input layer is assumed to be:
			# layer_shape == [num_images, img_height, img_width, num_channels]

			# The number of features is: img_height * img_width * num_channels
			# We can use a function from TensorFlow to calculate this.
			num_features = layer_shape[1:4].num_elements()

			# Reshape the layer to [num_images, num_features].
			# Note that we just set the size of the second dimension
			# to num_features and the size of the first dimension to -1
			# which means the size in that dimension is calculated
			# so the total size of the tensor is unchanged from the reshaping.
			layer_flat = tf.reshape(layer, [-1, num_features])

			# The shape of the flattened layer is now:
			# [num_images, img_height * img_width * num_channels]

			# Return both the flattened layer and the number of features.
			return layer_flat, num_features

		def new_fc_layer(input,  # The previous layer.
		                 num_inputs,  # Num. inputs from prev. layer.
		                 num_outputs,  # Num. outputs.
		                 use_relu=True):  # Use Rectified Linear Unit (ReLU)?

			# Create new weights and biases.
			weights = new_weights(shape=[num_inputs, num_outputs])
			biases = new_biases(length=num_outputs)

			# Calculate the layer as the matrix multiplication of
			# the input and weights, and then add the bias-values.
			layer = tf.matmul(input, weights) + biases

			# Use ReLU?
			if use_relu:
				layer = tf.nn.relu(layer)

			return layer

		x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
		x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
		y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
		y_true_cls = tf.argmax(y_true, dimension=1)
		layer_conv1, weights_conv1 = \
			new_conv_layer(input=x_image,
			               num_input_channels=num_channels,
			               filter_size=filter_size1,
			               num_filters=num_filters1,
			               use_pooling=True)
		layer_conv1
		layer_conv2, weights_conv2 = \
			new_conv_layer(input=layer_conv1,
			               num_input_channels=num_filters1,
			               filter_size=filter_size2,
			               num_filters=num_filters2,
			               use_pooling=True)
