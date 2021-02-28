import numpy as np
import tensorflow as tf
import gzip
import sys
sys.path.extend(['alg/'])
# import vcl
import coreset
import utils
from copy import deepcopy
import pickle
import vcl_factorize as vcl

class PermutedMnistGenerator():
	def __init__(self, max_iter=10):
		f = gzip.open('data/mnist.pkl.gz', 'rb')
		train_set, valid_set, test_set = pickle.load(f , encoding='latin1')
		f.close()

		self.X_train = np.vstack((train_set[0], valid_set[0]))
		self.Y_train = np.hstack((train_set[1], valid_set[1]))
		self.X_test = test_set[0]
		self.Y_test = test_set[1]
		self.max_iter = max_iter
		self.cur_iter = 0

	def get_dims(self):
		# Get data input and output dimensions
		return self.X_train.shape[1], 10

	def next_task(self):
		if self.cur_iter >= self.max_iter:
			raise Exception('Number of tasks exceeded!')
		else:
			np.random.seed(self.cur_iter)
			perm_inds = range(self.X_train.shape[1])
			perm_inds = np.random.permutation(perm_inds)

			# Retrieve train data
			next_x_train = deepcopy(self.X_train)
			next_x_train = next_x_train[:,perm_inds]
			next_y_train = np.eye(10)[self.Y_train]

			# Retrieve test data
			next_x_test = deepcopy(self.X_test)
			next_x_test = next_x_test[:,perm_inds]
			next_y_test = np.eye(10)[self.Y_test]

			self.cur_iter += 1

			return next_x_train, next_y_train, next_x_test, next_y_test

hidden_size = [100, 100]
batch_size = 256
no_epochs = int(sys.argv[2])
single_head = True
num_tasks = 5
sd = int(sys.argv[1])
# Run vanilla VCL

coreset_size = 0
np.random.seed(1)
data_gen = PermutedMnistGenerator(num_tasks)

print("seed ",sd)
tf.set_random_seed(sd)
vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, coreset.rand_from_batch, coreset_size, batch_size, single_head,sd = sd)
print (vcl_result)

# Run random coreset VCL
# tf.reset_default_graph()
# tf.set_random_seed(12)
# np.random.seed(1)

# coreset_size = 200
# data_gen = PermutedMnistGenerator(num_tasks)
# rand_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
#     coreset.rand_from_batch, coreset_size, batch_size, single_head)
# print rand_vcl_result

# # Run k-center coreset VCL
# tf.reset_default_graph()
# tf.set_random_seed(12)
# np.random.seed(1)

# data_gen = PermutedMnistGenerator(num_tasks)
# kcen_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
#     coreset.k_center, coreset_size, batch_size, single_head)
# print kcen_vcl_result

# data_gen = pickle.load(open("data_gen", "rb"))
# in_dim , out_dim = data_gen.get_dims()
# x_train , y_train , x_test , y_test = data_gen.next_task()
# print('in dim' , in_dim , 'out dim ', out_dim)
# print(x_train.shape , y_train.shape , x_test.shape , y_test.shape)
# print(y_test)


# f = gzip.open('data/mnist.pkl.gz', 'rb')
# train_set, valid_set, test_set = pickle.load(f , encoding='latin1')
# f.close()

# X_train = np.vstack((train_set[0], valid_set[0]))
# Y_train = np.hstack((train_set[1], valid_set[1]))
# X_test = test_set[0]
# Y_test = test_set[1]

# a = [1 , 0 , 1]
# print(np.eye(2)[a])
