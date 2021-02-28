import numpy as np
import tensorflow as tf
import utils_mixture_split as utils
import time
from cla_mixture_factorize import Vanilla_NN , MFVI_NN
import os , sys

def run_vcl(hidden_size, no_epochs, data_gen, coreset_method, coreset_size=0, batch_size=None, single_head=True,num_gauss= 1, num_train = 10, tau = 1.0,sd = 1,dim_factorize= 2):
	
	#make dir result
	path_folder_result = create_path_file_result(num_gauss , num_train , tau , sd,dim_factorize)
	in_dim, out_dim = data_gen.get_dims()
	x_coresets, y_coresets = [], []
	x_testsets, y_testsets = [], []
	all_acc = np.array([])
	numtask = 0
	print("num task ", data_gen.max_iter)
	print("num seed ", sd)
	print("in dim , out dim ", in_dim , out_dim)
	lr_list = [0.001 , 0.01 , 0.001 , 0.001, 0.001]

	for task_id in range(data_gen.max_iter):
		numtask += 1

		x_train, y_train, x_test, y_test = data_gen.next_task()
		x_testsets.append(x_test)
		y_testsets.append(y_test)

		# Set the readout head to train
		head = 0 if single_head else task_id
		bsize = x_train.shape[0] if (batch_size is None) else batch_size

		print("start pretraining ........")
		s_time = time.time()
		if task_id == 0:
			mf_weights = []
			mf_weight_share = []
			mf_weight_head = [None , None]
			for mixture in range(num_gauss):
				if mixture == 0:
					tf.set_random_seed(0)
				else:
					tf.set_random_seed(sd * mixture)
				ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0])
				ml_model.train(x_train, y_train, 0, 120 , bsize)
				mf_w = ml_model.get_weights()
				mf_weight_share.append([mf_w[0],mf_w[1]])
				ml_model.close_session()
			mf_variances = None
			mf_coffs = None
			mf_weights=[mf_weight_share , mf_weight_head]
			mf_variance_mlp = None
		e_time = time.time()
		print("time pretraining ",e_time - s_time)


		# Train on non-coreset data
		if task_id == 0:
			start_time = time.time()
			lr = 0.001
			mf_model = MFVI_NN(in_dim, hidden_size, out_dim, x_train.shape[0],no_train_samples=num_train ,prev_means=mf_weights, prev_log_variances=mf_variances,prev_coffs = mf_coffs,pre_variance_mlp = mf_variance_mlp,gauss_mixture = num_gauss, tau = tau , learning_rate= lr,dim_factorize=dim_factorize)
			print("train task ", numtask )
			mf_model.train(x_train, y_train, 0, no_epochs,  bsize)
			end_time = time.time()
			print("total time trainning ", str(end_time - start_time))			
		else:
			start_time = time.time()
			print("seed ",sd)
			lr = lr_list[task_id]
			mf_model = MFVI_NN(in_dim, hidden_size, out_dim, x_train.shape[0],no_train_samples=num_train ,prev_means=mf_weights, prev_log_variances=mf_variances,prev_coffs = mf_coffs,pre_variance_mlp = mf_variance_mlp,gauss_mixture = num_gauss, tau = tau , learning_rate= lr,dim_factorize=dim_factorize)
			print("train task ", numtask )
			print("batch size .... ", bsize)
			mf_model.train(x_train, y_train, head, no_epochs, bsize)
			end_time = time.time()
			print("total time trainning ", str(end_time - start_time))			
		
		#get weight cho model
		model_weights = mf_model.get_weights()
		mf_weights , mf_variances , mf_coffs, mf_variance_mlp = [] , [] , [] , []
		mf_weight_share , mf_weight_head , mf_variance_share , mf_variances_head  = [] , [] , [] , [] 
		#get share
		for mixture in range(num_gauss):
			mf_weight_share.append(model_weights[0][mixture][0])
			mf_variance_share.append(model_weights[0][mixture][1])
			mf_coffs.append(model_weights[0][mixture][2])
		#get head
		mf_weight_head = model_weights[1][0]
		mf_variances_head = model_weights[1][1]
		mf_variance_mlp = model_weights[1][2]
		#combine to total weight
		mf_weights = [mf_weight_share , mf_weight_head]
		mf_variances = [mf_variance_share , mf_variances_head]
			
		print("testing ..........")
		start_time_test = time.time()
		acc = utils.get_scores(mf_model, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size, no_epochs, single_head, batch_size)
		end_test_time = time.time()
		print("total time test ", end_test_time - start_time_test)
		all_acc = utils.concatenate_results(acc, all_acc)
		print(all_acc)
		write_data_to_file(all_acc , path_folder_result +"/gmm_vcl_split_k"+str(num_gauss)+"tau"+str(tau)+"init1.0"+"seed"+str(sd)+"no_train"+str(num_train)+".csv")
		mf_model.close_session()
	return all_acc

def write_data_to_file(numpy_result , file_name):
	with open(file_name , "w") as f:
		result = "\n".join([",".join([str(t) for t in row]) for row in numpy_result])
		f.write(result)
		print("Wrote data to file .....")

def create_path_file_result(num_gauss , num_train , tau , sd, dim_factorize):
	root = "result_split_notmnist_mixture_factorize_dimfactorize"
	if not os.path.exists(root):
		os.mkdir(root)
	path_result_gauss = "/".join([root , "num_gauss_"+str(num_gauss)])
	if not os.path.exists(path_result_gauss):
		os.mkdir(path_result_gauss)
	path_result_factorize = "/".join([path_result_gauss , "dim_factorize"+str(dim_factorize)])
	if not os.path.exists(path_result_factorize):
		os.mkdir(path_result_factorize)
	path_result_num_train = "/".join([path_result_factorize , "num_train_"+str(num_train)])
	if not os.path.exists(path_result_num_train):
		os.mkdir(path_result_num_train)
	path_result_tau = "/".join([path_result_num_train , "tau_"+ str(tau)])
	if not os.path.exists(path_result_tau):
		os.mkdir(path_result_tau)
	path_result_sd = "/".join([path_result_tau , "sd_"+str(sd)])
	if not os.path.exists(path_result_sd):
		os.mkdir(path_result_sd)
	return path_result_sd

def compute_learning_rate(current_lr , num_gauss, num_task):
	return current_lr / num_gauss