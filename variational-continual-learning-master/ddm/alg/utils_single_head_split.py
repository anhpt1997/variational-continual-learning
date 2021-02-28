import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from cla_single_mixture_split import MFVI_NN

def merge_coresets(x_coresets, y_coresets):
	merged_x, merged_y = x_coresets[0], y_coresets[0]
	for i in range(1, len(x_coresets)):
		merged_x = np.vstack((merged_x, x_coresets[i]))
		merged_y = np.vstack((merged_y, y_coresets[i]))
	return merged_x, merged_y

def get_scores(model, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size, no_epochs, single_head, batch_size=None):
	# mf_weights, mf_variances = model.get_weights()
	acc = []
	numclass = 2
	if single_head:
		if len(x_coresets) > 0:
			x_train, y_train = merge_coresets(x_coresets, y_coresets)
			bsize = x_train.shape[0] if (batch_size is None) else batch_size
			final_model = MFVI_NN(x_train.shape[1], hidden_size, y_train.shape[1], x_train.shape[0], prev_means=mf_weights, prev_log_variances=mf_variances)
			final_model.train(x_train, y_train, 0, no_epochs, bsize)
		else:
			final_model = model

	for i in range(len(x_testsets)):
		print(i)
		start, end = compute_offset(numclass , i)
		print("start " ,start , "end ",end)
		if not single_head:
			if len(x_coresets) > 0:
				x_train, y_train = x_coresets[i], y_coresets[i]
				bsize = x_train.shape[0] if (batch_size is None) else batch_size
				final_model = MFVI_NN(x_train.shape[1], hidden_size, y_train.shape[1], x_train.shape[0], prev_means=mf_weights, prev_log_variances=mf_variances)
				final_model.train(x_train, y_train, i, no_epochs, bsize)
			else:
				final_model = model

		head = 0 if single_head else i
		x_test, y_test = x_testsets[i], y_testsets[i]

		cur_acc =0.0
		batch_size = 5000
		N = y_test.shape[0]
		total_batch = int(np.ceil(N * 1.0 / batch_size))
		print("total batch  ", total_batch)
		for i in range(total_batch):
			list_pred = []
			start_ind = i* batch_size
			end_ind = np.min([(i + 1) * batch_size, N])
			batch_x = x_test[start_ind:end_ind, :]
			for j in range(2):
				pred = final_model.prediction_prob(batch_x, head)
				print(j , pred.shape)
				pred_mean = np.mean(pred, axis=0)
				list_pred.append(pred_mean)
			pred_overall = np.mean(np.asarray(list_pred) , axis = 0)
			pred_y = np.argmax(pred_overall[:,start:end], axis=1)
			y = np.argmax(y_test[start_ind : end_ind, :][:,start:end], axis=1)
			cur_acc += len(np.where((pred_y - y) == 0)[0]) * 1.0
			print('cur_acc / size ' , cur_acc , '/' , y_test[start_ind : end_ind].shape[0])
		cur_acc = cur_acc / y_test.shape[0]
		acc.append(cur_acc)

		if len(x_coresets) > 0 and not single_head:
			final_model.close_session()

	if len(x_coresets) > 0 and single_head:
		final_model.close_session()

	return acc

def concatenate_results(score, all_score):
	if all_score.size == 0:
		all_score = np.reshape(score, (1,-1))
	else:
		new_arr = np.empty((all_score.shape[0], all_score.shape[1]+1))
		new_arr[:] = np.nan
		new_arr[:,:-1] = all_score
		all_score = np.vstack((new_arr, score))
	return all_score

def compute_offset(numclass,task):
	end = min(10, numclass *(task+1))
	return numclass * task , end 

numclass , task = 2 ,2
a = np.arange(10)
s ,e = compute_offset(numclass , task)
print(a[s:e])