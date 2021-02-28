import numpy as np
import tensorflow as tf
import gzip
import pickle
import sys
sys.path.extend(['alg/'])
import vcl_mixture_notmnist_factorize as vcl
import coreset
import utils
from copy import deepcopy

class SplitMnistGenerator():
	def __init__(self):
		path_train_test_set = "data/not_mnist/train_test_small_notmnist"
		data_dict = pickle.load(open(path_train_test_set , "rb"))

		self.X_train = data_dict['image_train']
		self.X_test = data_dict['image_test']
		self.train_label = data_dict['label_train']
		self.test_label = data_dict['label_test']

		self.sets_0 = [0 , 1 ,2 ,3 ,4]
		self.sets_1 = [5 ,6 ,7 ,8 ,9]
		# self.sets_0  = [0, 1 ,2 ,3, 4, 5, 6 ,7 ,8]
		# self.sets_1 = [1 ,2 ,3 ,4, 5, 6, 7 ,8 ,9]
		self.max_iter = len(self.sets_0)
		self.cur_iter = 0

	def get_dims(self):
		# Get data input and output dimensions
		return self.X_train.shape[1], 2

	def next_task(self):
		if self.cur_iter >= self.max_iter:
			raise Exception('Number of tasks exceeded!')
		else:
			# Retrieve train data
			train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
			train_1_id = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
			next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))

			next_y_train = np.vstack((np.ones((train_0_id.shape[0], 1)), np.zeros((train_1_id.shape[0], 1))))
			next_y_train = np.hstack((next_y_train, 1-next_y_train))

			# Retrieve test data
			test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
			test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
			next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

			next_y_test = np.vstack((np.ones((test_0_id.shape[0], 1)), np.zeros((test_1_id.shape[0], 1))))
			next_y_test = np.hstack((next_y_test, 1-next_y_test))

			self.cur_iter += 1

			return next_x_train, next_y_train, next_x_test, next_y_test

if len(sys.argv) != 7:
	print("run_split_gauss_mixture.py num_gauss num_train no_epochs tau seed dim_factorize")
else:
	data_gen = SplitMnistGenerator()
	list_train, list_test = [], []
	for i in range(5):
		next_x_train, next_y_train, next_x_test, next_y_test = data_gen.next_task()
		list_train.append(next_x_train)
		list_test.append(next_x_test)
	train , test = np.concatenate(list_train,axis = 0) , np.concatenate(list_test ,axis = 0)
	print(len(train), len(test))
	# hidden_size = [50 , 50]
	# batch_size = None
	# no_epochs = int(sys.argv[3])
	# single_head = False
	# num_train = int(sys.argv[2])
	# tau = float(sys.argv[4])
	# i = int(sys.argv[5])
	# dim_factorize = int(sys.argv[6])
	# num_gauss = int(sys.argv[1])
	# coreset_size = 0
	# np.random.seed(1)
	# tf.compat.v1.set_random_seed(i)
	# vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, coreset.rand_from_batch, coreset_size, batch_size, single_head,sd = i, tau = tau , num_train = num_train, num_gauss = num_gauss,dim_factorize = dim_factorize)
	# print(vcl_result)

