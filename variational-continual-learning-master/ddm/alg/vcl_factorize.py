import numpy as np
import tensorflow as tf
import utils_factorize as utils
from clas_model_multihead_factorize_matrix import Vanilla_NN, MFVI_NN,Vanilla_NN_Factorize
import time

def run_vcl(hidden_size, no_epochs, data_gen, coreset_method, coreset_size=0, batch_size=None, single_head=True,sd = 0):
	print("VCL factorize")
	in_dim, out_dim = data_gen.get_dims()
	x_coresets, y_coresets = [], []
	x_testsets, y_testsets = [], []

	all_acc = np.array([])
	print("max iter ", data_gen.max_iter)

	for task_id in range(data_gen.max_iter):
		x_train, y_train, x_test, y_test = data_gen.next_task()
		x_testsets.append(x_test)
		y_testsets.append(y_test)

		# Set the readout head to train
		head = 0 if single_head else task_id
		bsize = x_train.shape[0] if (batch_size is None) else batch_size

		# Train network with maximum likelihood to initialize first model
		if task_id == 0:
			ml_model = Vanilla_NN_Factorize(in_dim, hidden_size, out_dim, x_train.shape[0],task_id)
			ml_model.train_factorize(x_train, y_train, 200, bsize)
			mf_weights = ml_model.get_weights()
			mf_variances = None
			ml_model.close_session()
		# 	print(mf_weights[0][1][2])

		# Select coreset if needed
		# if coreset_size > 0:
		#     x_coresets, y_coresets, x_train, y_train = coreset_method(x_coresets, y_coresets, x_train, y_train, coreset_size)

		# # Train on non-coreset data

		# if task_id ==0:
		# 	mf_variances, mf_variances_mlp =None, None
		# s_time = time.time()
		# mf_model = MFVI_NN(in_dim, hidden_size, out_dim, x_train.shape[0], prev_means=mf_weights, prev_log_variances=mf_variances, prev_variance_mlp = mf_variances_mlp)
		# mf_model.train(x_train, y_train,head, no_epochs, bsize)
		# e_time = time.time()
		# print("time train ",e_time - s_time)
		# mf_weights, mf_variances, mf_variances_mlp = mf_model.get_weights()
		# print("len ", len(mf_variances_mlp[0]))

		# # # # Incorporate coreset data and make prediction
		# acc = utils.get_scores(mf_model, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size, no_epochs, single_head, batch_size)
		# all_acc = utils.concatenate_results(acc, all_acc)
		# print(all_acc)
		# write_data_to_file(all_acc , "result_vcl_split_seed"+str(sd)+".csv")

		# mf_model.close_session()

	return all_acc

def write_data_to_file(numpy_result , file_name):
	with open(file_name , "w") as f:
		result = "\n".join([",".join([str(t) for t in row]) for row in numpy_result])
		f.write(result)
		print("Wrote data to file .....")
