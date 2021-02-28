import tensorflow as tf
import numpy as np
from copy import deepcopy

np.random.seed(0)
tf.set_random_seed(0)

# variable initialization functions
def weight_variable(shape, init_weights=None):
	if init_weights is not None:
		initial = tf.constant(init_weights)
	else:
		initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def small_variable(shape):
	initial = tf.constant(-6.0, shape=shape)
	return tf.Variable(initial)

def zero_variable(shape):
	initial = tf.zeros(shape=shape)
	return tf.Variable(initial)

def _create_weights_mf(in_dim, hidden_size, out_dim, init_weights=None, init_variances=None):
	size = deepcopy(hidden_size)
	size.append(out_dim)
	size.insert(0, in_dim)
	no_params = 0
	for i in range(len(size) - 1):
		no_weights = size[i] * size[i+1]
		no_biases = size[i+1]
		no_params += (no_weights + no_biases)
	m_weights = weight_variable([no_params], init_weights)
	if init_variances is None:
		v_weights = small_variable([no_params])
	else:
		v_weights = tf.Variable(tf.constant(init_variances, dtype=tf.float32))
	return no_params, m_weights, v_weights, size

class Cla_NN(object):
	def __init__(self, input_size, hidden_size, output_size, training_size):
		# input and output placeholders
		self.x = tf.placeholder(tf.float32, [None, input_size])
		self.y = tf.placeholder(tf.float32, [None, output_size])
		self.task_idx = tf.placeholder(tf.int32)
		self.abc = tf.Variable(0.)
		self.sess = tf.Session()
		
	def assign_optimizer(self, learning_rate=0.001):
		self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

	def assign_session(self):
		# Initializing the variables
		init = tf.global_variables_initializer()

		# launch a session

		self.sess.run(init)

	def train_factorize(self, x_train, y_train, no_epochs=1000, batch_size=100, display_epoch=10):
		N = x_train.shape[0]
		if batch_size > N:
			batch_size = N

		sess = self.sess
		costs = []
		# Training cycle
		for epoch in range(no_epochs):
			perm_inds = range(x_train.shape[0])
			perm_inds = np.random.permutation(perm_inds)
			cur_x_train = x_train[perm_inds]
			cur_y_train = y_train[perm_inds]

			avg_cost = 0.
			total_batch = int(np.ceil(N * 1.0 / batch_size))
			# Loop over all batches
			for i in range(total_batch):
				start_ind = i*batch_size
				end_ind = np.min([(i+1)*batch_size, N])
				batch_x = cur_x_train[start_ind:end_ind, :]
				batch_y = cur_y_train[start_ind:end_ind, :]
				# Run optimization op (backprop) and cost op (to get loss value)
				_, c = sess.run(
					[self.train_step, self.cost], 
					feed_dict={self.x: batch_x, self.y: batch_y})
				# Compute average loss
				avg_cost += c / total_batch
			# Display logs per epoch step
			if epoch % display_epoch == 0:
				print("Epoch:", '%04d' % (epoch+1), "cost=", \
					"{:.9f}".format(avg_cost))
			costs.append(avg_cost)
		print("Optimization Finished!")
		return costs
	
	def train(self, x_train, y_train,task_idx, no_epochs=1000, batch_size=100, display_epoch=10):
		N = x_train.shape[0]
		if batch_size > N:
			batch_size = N

		sess = self.sess
		costs = []
		# Training cycle
		for epoch in range(no_epochs):
			if epoch % display_epoch == 0:
				print("abc ", sess.run(self.abc))
			perm_inds = range(x_train.shape[0])
			perm_inds = np.random.permutation(perm_inds)
			cur_x_train = x_train[perm_inds]
			cur_y_train = y_train[perm_inds]

			avg_cost = 0.
			total_batch = int(np.ceil(N * 1.0 / batch_size))
			# Loop over all batches
			for i in range(total_batch):
				start_ind = i*batch_size
				end_ind = np.min([(i+1)*batch_size, N])
				batch_x = cur_x_train[start_ind:end_ind, :]
				batch_y = cur_y_train[start_ind:end_ind, :]
				# Run optimization op (backprop) and cost op (to get loss value)
				_, c = sess.run(
					[self.train_step, self.cost], 
					feed_dict={self.x: batch_x, self.y: batch_y, self.task_idx: task_idx})
				# Compute average loss
				avg_cost += c / total_batch
			# Display logs per epoch step
			if epoch % display_epoch == 0:
				print("Epoch:", '%04d' % (epoch+1), "cost=", \
					"{:.9f}".format(avg_cost))
			costs.append(avg_cost)
		print("Optimization Finished!")
		return costs

	def prediction(self, x_test, task_idx):
		# Test model
		prediction = self.sess.run([self.pred], feed_dict={self.x: x_test, self.task_idx: task_idx})[0]
		return prediction

	def prediction_prob(self, x_test, task_idx):
		prob = self.sess.run([tf.nn.softmax(self.pred)], feed_dict={self.x: x_test, self.task_idx : task_idx})[0]
		return prob

	def get_weights(self):
		weights = self.sess.run([self.weights])[0]
		return weights

	def close_session(self):
		self.sess.close()


""" Neural Network Model """
class Vanilla_NN(Cla_NN):
	def __init__(self, input_size, hidden_size, output_size, training_size, prev_weights=None, learning_rate=0.001):

		super(Vanilla_NN, self).__init__(input_size, hidden_size, output_size, training_size)
		# init weights and biases
		self.W, self.b, self.W_last, self.b_last, self.size = self.create_weights(
				input_size, hidden_size, output_size, prev_weights)
		self.no_layers = len(hidden_size) + 1
		self.pred = self._prediction(self.x, self.task_idx)
		self.cost = - self._logpred(self.x, self.y, self.task_idx)
		self.weights = [self.W, self.b, self.W_last, self.b_last]

		self.assign_optimizer(learning_rate)
		self.assign_session()

	def _prediction(self, inputs, task_idx):
		act = inputs
		for i in range(self.no_layers-1):
			pre = tf.add(tf.matmul(act, self.W[i]), self.b[i])
			act = tf.nn.relu(pre)
		pre = tf.add(tf.matmul(act, tf.gather(self.W_last, task_idx)), tf.gather(self.b_last, task_idx))
		return pre

	def _logpred(self, inputs, targets, task_idx):
		pred = self._prediction(inputs, task_idx)
		log_lik = - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=targets))
		return log_lik

	def create_weights(self, in_dim, hidden_size, out_dim, prev_weights):
		hidden_size = deepcopy(hidden_size)
		hidden_size.append(out_dim)
		hidden_size.insert(0, in_dim)
		no_params = 0
		no_layers = len(hidden_size) - 1
		W = []
		b = []
		W_last = []
		b_last = []
		for i in range(no_layers-1):
			din = hidden_size[i]
			dout = hidden_size[i+1]
			if prev_weights is None:
				Wi_val = tf.truncated_normal([din, dout], stddev=0.1)
				bi_val = tf.truncated_normal([dout], stddev=0.1)
			else:
				Wi_val = tf.constant(prev_weights[0][i])
				bi_val = tf.constant(prev_weights[1][i])
			Wi = tf.Variable(Wi_val)
			bi = tf.Variable(bi_val)
			W.append(Wi)
			b.append(bi)

		if prev_weights is not None:
			prev_Wlast = prev_weights[2]
			prev_blast = prev_weights[3]
			no_prev_tasks = len(prev_Wlast)
			for j in range(no_prev_tasks):
				W_j = prev_Wlast[j]
				b_j = prev_blast[j]
				Wi = tf.Variable(W_j)
				bi = tf.Variable(b_j)
				W_last.append(Wi)
				b_last.append(bi)

		din = hidden_size[-2]
		dout = hidden_size[-1]
		Wi_val = tf.truncated_normal([din, dout], stddev=0.1)
		bi_val = tf.truncated_normal([dout], stddev=0.1)
		Wi = tf.Variable(Wi_val)
		bi = tf.Variable(bi_val)
		W_last.append(Wi)
		b_last.append(bi)
			
		return W, b, W_last, b_last, hidden_size

class Vanilla_NN_Factorize(Cla_NN):
	def __init__(self, input_size, hidden_size, output_size, training_size,task_idx, prev_weights=None, learning_rate=0.001):

		super(Vanilla_NN_Factorize, self).__init__(input_size, hidden_size, output_size, training_size)
		# init weights and biases
		self.dim_factorize = 2
		self.task_idx = task_idx
		self.W, self.b, self.W_last, self.b_last, self.size = self.create_weights(input_size, hidden_size, output_size, prev_weights )
		self.no_layers = len(hidden_size) + 1
		self.pred = self._prediction(self.x)
		self.cost = - self._logpred(self.x, self.y)
		self.weights = [self.W, self.b, self.W_last, self.b_last]
		print("task id ",self.task_idx)
		self.assign_optimizer(learning_rate)
		self.assign_session()

	def _prediction(self, inputs):
		act = inputs
		for i in range(self.no_layers-1):
			# pre = tf.add(tf.matmul(act, self.W[i]), self.b[i])
			Wi = self.compute_weightfromfactorize(self.W[i])
			pre = tf.add(tf.matmul(act , Wi) , self.b[i])
			act = tf.nn.relu(pre)
		# wlast = self.compute_weightfromfactorize(tf.gather(self.W_last , task_idx))
		wlast = self.compute_weightfromfactorize(self.W_last[self.task_idx])
		pre = tf.add(tf.matmul(act, wlast), tf.gather(self.b_last, self.task_idx))
		return pre

	def _logpred(self, inputs, targets):
		pred = self._prediction(inputs)
		log_lik = - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=targets))
		return log_lik
	
	def merge_UV(self, U,V):
		#get shape 
		shapeU, shapeV= U.shape, V.shape 
		M,N,K = shapeU[0], shapeV[0], shapeU[1]
		Uextand  = tf.repeat(U, [N]*M , axis = 0 )
		Vextand = tf.tile(V, [M, 1])
		UVconcat = tf.concat( (Uextand , Vextand) ,axis = 1)
		return UVconcat , M , N, K

	# def feed_forward_neural_factorize(self, merge_UV, model_factorize):
	# 	uvconcat, m,n,k = merge_UV
	# 	w1, b1, w2, b2, w3 ,b3 = model_factorize
	# 	h1 = tf.matmul(uvconcat , w1) + b1
	# 	a1 = tf.nn.relu(h1)
	# 	h2 = tf.matmul(a1, w2) + b2
	# 	a2 = tf.nn.relu(h2)
	# 	h3 = tf.matmul(a2,w3) + b3
	# 	return tf.reshape(h3 , shape = [m,n])

	def feed_forward_neural_factorize(self, merge_UV, model_factorize):
		uvconcat, m,n,k = merge_UV
		w1, b1 = model_factorize
		h1 = tf.matmul(uvconcat , w1) + b1
		# a1 = tf.nn.relu(h1)
		# h2 = tf.matmul(a1, w2) + b2
		return tf.reshape(h1 , shape = [m,n])

	# def init_weights_mlp_factorize(self, k):
	# 	# 3 layer shape [2k, k , k , 1]
	# 	w1 = tf.truncated_normal([2*k , k] , stddev = 0.1)
	# 	b1 = tf.truncated_normal([k] , stddev = 0.1)
	# 	w2 = tf.truncated_normal([k , k] , stddev=0.1)
	# 	b2 = tf.truncated_normal([k], stddev = 0.1)
	# 	w3 = tf.truncated_normal([k , 1], stddev = 0.1)
	# 	b3 = tf.truncated_normal([1], stddev = 0.1)
	# 	return tf.Variable(w1) , tf.Variable(b1) , tf.Variable(w2) , tf.Variable(b2), tf.Variable(w3), tf.Variable(b3)

	def init_weights_mlp_factorize(self, k):
		# 3 layer shape [2k, k , k , 1]
		w1 = tf.truncated_normal([2*k , 1] , stddev = 0.1)
		b1 = tf.truncated_normal([1], stddev = 0.1)
		return tf.Variable(w1) , tf.Variable(b1)

	def init_weight_parameter_factorize(self, din , dout, k):
		Wiu_val = tf.truncated_normal([din,k] , stddev = 0.1)
		Wiv_val = tf.truncated_normal([dout, k] , stddev = 0.1)
		Wi_weightmlp = self.init_weights_mlp_factorize(k)
		return (tf.Variable(Wiu_val), tf.Variable(Wiv_val), Wi_weightmlp)

	def compute_weightfromfactorize(self, parameter):
		Wiu, Wiv, Wimlp = parameter[0] , parameter[1] , parameter[2]
		mergeUV = self.merge_UV(Wiu , Wiv)
		Wi = self.feed_forward_neural_factorize(mergeUV , Wimlp)
		return Wi

	def create_weights(self, in_dim, hidden_size, out_dim, prev_weights):
		hidden_size = deepcopy(hidden_size)
		hidden_size.append(out_dim)
		hidden_size.insert(0, in_dim)
		no_layers = len(hidden_size) - 1
		W = []
		b = []
		W_last = []
		b_last = []
		#lam cho share network
		for i in range(no_layers-1):
			din = hidden_size[i]
			dout = hidden_size[i+1]
			if prev_weights is None:
				# Wi_val = tf.truncated_normal([din, dout], stddev=0.1)
				#thay Wi_val bang 3 phan tu (U,V,(weight_mlp))
				parameter = self.init_weight_parameter_factorize(din, dout ,self.dim_factorize)				
				bi_val = tf.truncated_normal([dout], stddev=0.1)

				W.append(parameter)
				b.append(tf.Variable(bi_val))
			else:
				pass
				# Wi_val = tf.constant(prev_weights[0][i])
				# bi_val = tf.constant(prev_weights[1][i])
			# Wi = tf.Variable(Wi_val)
			# bi = tf.Variable(bi_val)
			# W.append(Wi)
			# b.append(bi)

		if prev_weights is not None:
			# prev_Wlast = prev_weights[2]
			# prev_blast = prev_weights[3]
			# no_prev_tasks = len(prev_Wlast)
			# for j in range(no_prev_tasks):
			# 	W_j = prev_Wlast[j]
			# 	b_j = prev_blast[j]
			# 	Wi = tf.Variable(W_j)
			# 	bi = tf.Variable(b_j)
			# 	W_last.append(Wi)
			# 	b_last.append(bi)
			pass

		din = hidden_size[-2]
		dout = hidden_size[-1]
		parameter = self.init_weight_parameter_factorize(din , dout , self.dim_factorize)
		# Wiu, Wiv, Wimlp = parameter
		# mergeUV = self.merge_UV(Wiu , Wiv)
		# Wi = self.feed_forward_neural_factorize(mergeUV , Wimlp)

		bi_val = tf.truncated_normal([dout], stddev=0.1)
		bi = tf.Variable(bi_val)
		W_last.append(parameter)
		b_last.append(bi)
			
		return W, b, W_last, b_last, hidden_size

""" Bayesian Neural Network with Mean field VI approximation """
class MFVI_NN(Cla_NN):
	def __init__(self, input_size, hidden_size, output_size, training_size,
		no_train_samples=10, no_pred_samples=50, prev_means=None, prev_log_variances=None, prev_variance_mlp = None, learning_rate=0.001, 
		prior_mean=0., prior_var=1.):

		super(MFVI_NN, self).__init__(input_size, hidden_size, output_size, training_size)

		self.dim_factorize = 5
		m, v, v_factorize, self.size = self.create_weights(
			input_size, hidden_size, output_size, prev_means, prev_log_variances,prev_variance_mlp)
		
		self.W_m, self.b_m, self.W_last_m, self.b_last_m = m[0], m[1], m[2], m[3]   
		self.W_v, self.b_v, self.b_last_v = v[0], v[1], v[2]
		self.Ulast, self.Vlast , self.W1last, self.b1last , self.W2last, self.b2last, self.W3last, self.b3last = v_factorize[0] , v_factorize[1] , v_factorize[2] , v_factorize[3] , v_factorize[4], v_factorize[5] , v_factorize[6] ,v_factorize[7]

		#compute Wlastv
		self.Wlast_v = []
		no_task = len(self.b_last_v)
		for i in range(no_task):
			self.Wlast_v.append(self.compute_weightfromfactorize_variance( (self.Ulast[i] , self.Vlast[i] , (self.W1last[i], self.b1last[i] , self.W2last[i], self.b2last[i], self.W3last[i], self.b3last[i]))))
		self.weights = [m, v , v_factorize]
		# self.task_idx = task_idx
		m, v  = self.create_prior(input_size, hidden_size, output_size, prev_means, prev_log_variances, prev_variance_mlp, prior_mean, prior_var)
		self.prior_W_m, self.prior_b_m, self.prior_W_last_m, self.prior_b_last_m = m[0], m[1], m[2], m[3]
		self.prior_W_v, self.prior_b_v, self.prior_W_last_v, self.prior_b_last_v = v[0], v[1], v[2], v[3]

		# self.Wlast_v_prior = []
		# no_task = len(self.b_last_v)
		# for i in range(no_task):
		# 	self.Wlast_v_prior.append(self.compute_weightfromfactorize_variance( (self.Ulast[i] , self.Vlast[i] , (self.W1last[i], self.b1last[i] , self.W2last[i], self.b2last[i], self.W3last[i], self.b3last[i]))))


		self.no_layers = len(self.size) - 1
		self.no_train_samples = no_train_samples
		self.no_pred_samples = no_pred_samples
		self.pred = self._prediction(self.x,  self.task_idx,self.no_pred_samples)
		self.abc = tf.div(self._KL_term(), training_size)
		self.cost  = tf.div(self._KL_term(), training_size)- self._logpred(self.x, self.y, self.task_idx)
		# self.cost = -self._logpred(self.x , self.y, self.task_idx)
		self.assign_optimizer(learning_rate=0.001)
		self.assign_session()
		print("num train ",self.no_train_samples)
		print("num pred ",self.no_pred_samples)
		print("learning  rate ",learning_rate)
		print("hidden size ", hidden_size)
		print("input size ",input_size)
		print("out put size ", output_size)

	def _prediction(self, inputs, task_idx, no_samples):
		return self._prediction_layer(inputs, task_idx, no_samples)
	# this samples a layer at a time
	def _prediction_layer(self, inputs, task_idx, no_samples):
		#ham nay phai sua lai cho lay mau epsilon
		K = no_samples
		act = tf.tile(tf.expand_dims(inputs, 0), [K, 1, 1])        
		for i in range(self.no_layers-1):
			din = self.size[i]
			dout = self.size[i+1]
			eps_w = tf.random_normal((K, din, dout), 0, 1, dtype=tf.float32)
			eps_b = tf.random_normal((K, 1, dout), 0, 1, dtype=tf.float32)
			
			# weights = tf.add(tf.multiply(eps_w, tf.exp(0.5*self.W_v[i])), self.W_m[i])
			W_v  = self.compute_weightfromfactorize_variance(self.W_v[i])
			W_m = self.W_m[i]
			weights = tf.add(tf.multiply(eps_w, tf.exp(0.5*W_v)), W_m)
			biases = tf.add(tf.multiply(eps_b, tf.exp(0.5*self.b_v[i])), self.b_m[i])
			pre = tf.add(tf.einsum('mni,mio->mno', act, weights), biases)
			act = tf.nn.relu(pre)
		din = self.size[-2]
		dout = self.size[-1]
		eps_w = tf.random_normal((K, din, dout), 0, 1, dtype=tf.float32)
		eps_b = tf.random_normal((K, 1, dout), 0, 1, dtype=tf.float32)


		Wtask_m = tf.gather(self.W_last_m , task_idx)
		# Wtask_v = self.compute_weightfromfactorize_variance(self.W_last_v[task_idx])
		#compute Wtask_v
		# Ulast, Vlast = tf.gather(self.Ulast , self.task_idx) , tf.gather(self.Vlast , self.task_idx)
		# W1last, b1last = tf.gather(self.W1last , self.task_idx) , tf.gather(self.b1last , self.task_idx)
		# W2last, b2last = tf.gather(self.W2last , self.task_idx) , tf.gather(self.b2last , self.task_idx)
		# W3last , b3last = tf.gather(self.W3last , self.task_idx) , tf.gather(self.b3last , self.task_idx)
		# mergeUV = self.merge_variance(Ulast , Vlast, din , dout , self.dim_factorize)
		# Wtask_v = self.feed_forward_neural_factorize_variance(mergeUV , (W1last , b1last , W2last , b2last , W3last , b3last))
		Wtask_v = tf.gather(self.Wlast_v , task_idx)

		btask_m = tf.gather(self.b_last_m, task_idx)
		btask_v = tf.gather(self.b_last_v, task_idx)
		weights = tf.add(tf.multiply(eps_w, tf.exp(0.5*Wtask_v)), Wtask_m)
		biases = tf.add(tf.multiply(eps_b, tf.exp(0.5*btask_v)), btask_m)
		act = tf.expand_dims(act, 3)
		weights = tf.expand_dims(weights, 1)
		pre = tf.add(tf.reduce_sum(act * weights, 2), biases)
		return pre

	def _logpred(self, inputs, targets, task_idx):
		pred = self._prediction(inputs,task_idx, self.no_train_samples)
		targets = tf.tile(tf.expand_dims(targets, 0), [self.no_train_samples, 1, 1])
		log_lik = - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=targets))
		return log_lik

	def _KL_term(self):
		kl = 0
		for i in range(self.no_layers-1):
			din = self.size[i]
			dout = self.size[i+1]
			# m, v = self.W_m[i], self.W_v[i]
			m,v = self.W_m[i] , self.compute_weightfromfactorize_variance(self.W_v[i])
			m0, v0 = self.prior_W_m[i], self.prior_W_v[i]
			const_term = -0.5 * dout * din
			log_std_diff = 0.5 * tf.reduce_sum(tf.math.log(v0) - v)
			mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m)**2) / v0)
			kl += const_term + log_std_diff + mu_diff_term

			m, v = self.b_m[i], self.b_v[i]
			m0, v0 = self.prior_b_m[i], self.prior_b_v[i]
			const_term = -0.5 * dout
			log_std_diff = 0.5 * tf.reduce_sum(tf.math.log(v0) - v)
			mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m)**2) / v0)
			kl += const_term + log_std_diff + mu_diff_term

		no_tasks = len(self.W_last_m)
		print("no task ", no_tasks)
		din = self.size[-2]
		dout = self.size[-1]
		for i in range(no_tasks):
			# m, v = self.W_last_m[i], self.W_last_v[i]
			m,v = self.W_last_m[i] , self.Wlast_v[i]
			m0, v0 = self.prior_W_last_m[i], self.prior_W_last_v[i]
			const_term = -0.5 * dout * din
			log_std_diff = 0.5 * tf.reduce_sum(tf.log(v0) - v)
			mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m)**2) / v0)
			kl += const_term + log_std_diff + mu_diff_term

			m, v = self.b_last_m[i], self.b_last_v[i]
			m0, v0 = self.prior_b_last_m[i], self.prior_b_last_v[i]
			const_term = -0.5 * dout
			log_std_diff = 0.5 * tf.reduce_sum(np.log(v0) - v)
			mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m)**2) / v0)
			kl += const_term + log_std_diff + mu_diff_term
		return kl

	def merge_UV(self, U,V):
		#get shape 
		shapeU, shapeV= U.shape, V.shape 
		M,N,K = shapeU[0], shapeV[0], shapeU[1]
		Uextand  = tf.repeat(U, [N]*M , axis = 0 )
		Vextand = tf.tile(V, [M, 1])
		UVconcat = tf.concat( (Uextand , Vextand) ,axis = 1)
		return UVconcat , M , N, K
	
	def feed_forward_neural_factorize_mean(self, merge_UV, model_factorize):
		uvconcat, m,n,k = merge_UV
		w1, b1, w2, b2, w3 ,b3 = model_factorize
		h1 = tf.matmul(uvconcat , w1) + b1
		a1 = tf.nn.relu(h1)
		h2 = tf.matmul(a1, w2) + b2
		a2 = tf.nn.relu(h2)
		h3 = tf.matmul(a2,w3) + b3
		return tf.reshape(h3 , shape = [m,n])

	def feed_forward_neural_factorize_variance(self, merge_UV, model_factorize):
		uvconcat, m,n,k = merge_UV
		w1, b1, w2, b2, w3 ,b3 = model_factorize
		h1 = tf.matmul(uvconcat , w1) + b1
		a1 = tf.nn.relu(h1)
		h2 = tf.matmul(a1, w2) + b2
		a2 = tf.nn.relu(h2)
		h3 = tf.matmul(a2,w3) + b3
		return tf.reshape(h3 , shape = [m,n]) - 6.0

	def init_weights_mlp_factorize(self, k):
		# 3 layer shape [2k, k , k , 1]
		w1 = tf.truncated_normal([2*k , k] , stddev = 0.1)
		b1 = tf.truncated_normal([k] , stddev = 0.1)
		w2 = tf.truncated_normal([k , k] , stddev=0.1)
		b2 = tf.truncated_normal([k], stddev = 0.1)
		w3 = tf.truncated_normal([k , 1], stddev = 0.1)
		b3 = tf.truncated_normal([1], stddev = 0.1)
		return tf.Variable(w1) , tf.Variable(b1) , tf.Variable(w2) , tf.Variable(b2), tf.Variable(w3), tf.Variable(b3)

	def init_weight_parameter_factorize(self, din , dout, k):
		Wiu_val = tf.truncated_normal([din,k] , stddev = 0.1)
		Wiv_val = tf.truncated_normal([dout, k] , stddev = 0.1)
		Wi_weightmlp = self.init_weights_mlp_factorize(k)
		return (tf.Variable(Wiu_val), tf.Variable(Wiv_val), Wi_weightmlp)

	def assign_weight_parameter_factorize(self, previous_value, task):
		Wiu_val , Wiv_val , W1, b1 ,W2, b2, W3, b3 = previous_value[0][task], previous_value[1][task] , previous_value[2][task], previous_value[3][task], previous_value[4][task], previous_value[5][task], previous_value[6][task], previous_value[7][task]
		return (tf.Variable(Wiu_val), tf.Variable(Wiv_val), [tf.Variable(t) for t in [W1, b1 ,W2, b2, W3, b3]])

	def assign_variance_share(self, previous_value, layer):
		Wiu_val , Wiv_val , mlp = previous_value[layer][0] ,  previous_value[layer][1] ,  previous_value[layer][2] 
		W1, b1 ,W2, b2, W3, b3 = mlp
		return (tf.Variable(Wiu_val), tf.Variable(Wiv_val), [tf.Variable(t) for t in [W1, b1 ,W2, b2, W3, b3]])

	def compute_weightfromfactorize_variance(self, parameter):
		Wiu, Wiv, Wimlp = parameter[0] , parameter[1] , parameter[2]
		mergeUV = self.merge_UV(Wiu , Wiv)
		Wi = self.feed_forward_neural_factorize_variance(mergeUV , Wimlp)
		return Wi

	def create_weights(self, in_dim, hidden_size, out_dim, prev_weights, prev_variances, prev_variance_mlp):
		hidden_size = deepcopy(hidden_size)
		hidden_size.append(out_dim)
		hidden_size.insert(0, in_dim)
		no_layers = len(hidden_size) - 1
		W_m = []
		b_m = []
		W_last_m = []
		b_last_m = []
		W_v = []
		b_v = []
		b_last_v = []
		Wulast_v, Wvlast_v , W1last_v,b1last_v,W2last_v,b2last_v,W3last_v, b3last_v =[], [], [] , [] , [] , [] , [] , []
		for i in range(no_layers-1):
			din = hidden_size[i]
			dout = hidden_size[i+1]
			if prev_weights is None:
				Wi_m_val = tf.truncated_normal([din, dout], stddev=0.1)
				Wi_v_val = self.init_weight_parameter_factorize(din , dout , self.dim_factorize)

				bi_m_val = tf.truncated_normal([dout], stddev=0.1)
				bi_v_val = tf.constant(-6.0, shape=[dout])

				W_m.append(tf.Variable(Wi_m_val))
				W_v.append(Wi_v_val)
				b_m.append(tf.Variable(bi_m_val))
				b_v.append(tf.Variable(bi_v_val))
			else:
				Wi_m_val = prev_weights[0][i]
				bi_m_val = prev_weights[1][i]
				if prev_variances is None:
					Wi_v_val = self.init_weight_parameter_factorize(din , dout , self.dim_factorize)					
					bi_v_val = tf.constant(-6.0, shape=[dout])
				else:
					Wi_v_val = self.assign_variance_share(prev_variances[0],i)
					bi_v_val = prev_variances[1][i]
					#get value from previous task
				W_m.append(tf.Variable(Wi_m_val))
				b_m.append(tf.Variable(bi_m_val))
				b_v.append(tf.Variable(bi_v_val))
				W_v.append(Wi_v_val)

		# if there are previous tasks
		if prev_weights is not None and prev_variances is not None:
			prev_Wlast_m = prev_weights[2]
			prev_blast_m = prev_weights[3]
			prev_blast_v = prev_variances[2]
			prev_Wlast_mlp = prev_variance_mlp
			no_prev_tasks = len(prev_Wlast_m)
			for i in range(no_prev_tasks):
				W_i_m = prev_Wlast_m[i]
				b_i_m = prev_blast_m[i]
				b_i_v = prev_blast_v[i]

				#get Wi_v_val mlp
				(Wu_v_val , Wv_v_val, Wmlp_val) = self.assign_weight_parameter_factorize(prev_Wlast_mlp, i)
				(W1_val , b1_val , W2_val , b2_val , W3_val , b3_val ) = Wmlp_val
				Wulast_v.append(Wu_v_val)
				Wvlast_v.append(Wv_v_val)
				W1last_v.append(W1_val)
				W2last_v.append(W2_val)
				W3last_v.append(W3_val)
				b1last_v.append(b1_val)
				b2last_v.append(b2_val)
				b3last_v.append(b3_val)

				Wi_m = tf.Variable(W_i_m)
				bi_m = tf.Variable(b_i_m)
				bi_v = tf.Variable(b_i_v)
				W_last_m.append(Wi_m)
				b_last_m.append(bi_m)
				# W_last_v.append(W_i_v)
				b_last_v.append(bi_v)

		din = hidden_size[-2]
		dout = hidden_size[-1]

		# if point estimate is supplied
		if prev_weights is not None and prev_variances is None:
			Wi_m_val = prev_weights[2][0]
			bi_m_val = prev_weights[3][0]
		else:
			Wi_m_val = tf.truncated_normal([din, dout], stddev=0.1)
			bi_m_val = tf.truncated_normal([dout], stddev=0.1)
			# Wi_m_val = self.init_weight_parameter_factorize(din , dout , self.dim_factorize)
		# Wi_v_val = tf.constant(-6.0, shape=[din, dout])
		(Wu_v_val , Wv_v_val, Wmlp_val) = self.init_weight_parameter_factorize(din , dout , self.dim_factorize)
		(W1_val , b1_val , W2_val , b2_val , W3_val , b3_val ) = Wmlp_val
		Wulast_v.append(Wu_v_val)
		Wvlast_v.append(Wv_v_val)
		W1last_v.append(W1_val)
		W2last_v.append(W2_val)
		W3last_v.append(W3_val)
		b1last_v.append(b1_val)
		b2last_v.append(b2_val)
		b3last_v.append(b3_val)

		bi_v_val = tf.constant(-6.0, shape=[dout])

		bi_m = tf.Variable(bi_m_val)
		bi_v = tf.Variable(bi_v_val)
		W_last_m.append(tf.Variable(Wi_m_val))
		b_last_m.append(bi_m)
		b_last_v.append(bi_v)
			
		return [W_m, b_m, W_last_m, b_last_m], [W_v, b_v, b_last_v], [Wulast_v , Wvlast_v,W1last_v,b1last_v,W2last_v,b2last_v,W3last_v,b3last_v], hidden_size

	def create_prior(self, in_dim, hidden_size, out_dim, prev_weights, prev_variances, prev_variance_mlp ,prior_mean, prior_var):
		hidden_size = deepcopy(hidden_size)
		hidden_size.append(out_dim)
		hidden_size.insert(0, in_dim)
		no_params = 0
		no_layers = len(hidden_size) - 1
		W_m = []
		b_m = []
		W_last_m = []
		b_last_m = []
		W_v = []
		b_v = []
		W_last_v = []
		b_last_v = []
		for i in range(no_layers-1):
			din = hidden_size[i]
			dout = hidden_size[i+1]
			if prev_weights is not None and prev_variances is not None:
				Wi_m = prev_weights[0][i]
				bi_m = prev_weights[1][i]
				Wi_v = tf.math.exp(self.compute_weightfromfactorize_variance(prev_variances[0][i]))
				bi_v = np.exp(prev_variances[1][i])
			else:
				Wi_m = prior_mean
				bi_m = prior_mean
				Wi_v = prior_var
				bi_v = prior_var

			W_m.append(Wi_m)
			b_m.append(bi_m)
			W_v.append(Wi_v)
			b_v.append(bi_v)

		# if there are previous tasks
		if prev_weights is not None and prev_variances is not None:
			prev_Wlast_m = prev_weights[2]
			prev_blast_m = prev_weights[3]
			prev_blast_v = prev_variances[2]
			prev_Wlast_mlp = prev_variance_mlp
			no_prev_tasks = len(prev_Wlast_m)
			for i in range(no_prev_tasks):
				Wi_m = prev_Wlast_m[i]
				bi_m = prev_blast_m[i]

				# Wi_v = tf.math.exp(self.compute_weightfromfactorize_variance(prev_Wlast_v[i]))
				Wu_v_val , Wv_v_val, W1 , b1 ,W2, b2, W3, b3 = prev_Wlast_mlp[0][i], prev_Wlast_mlp[1][i], prev_Wlast_mlp[2][i], prev_Wlast_mlp[3][i], prev_Wlast_mlp[4][i], prev_Wlast_mlp[5][i], prev_Wlast_mlp[6][i], prev_Wlast_mlp[7][i]

				Wi_v = tf.math.exp(self.feed_forward_neural_factorize_variance( self.merge_UV(Wu_v_val, Wv_v_val) , [W1 , b1 ,W2, b2, W3, b3 ]))
				bi_v = np.exp(prev_blast_v[i])
				
				W_last_m.append(Wi_m)
				b_last_m.append(bi_m)
				W_last_v.append(Wi_v)
				b_last_v.append(bi_v)

		din = hidden_size[-2]
		dout = hidden_size[-1]
		Wi_m = prior_mean
		bi_m = prior_mean
		Wi_v = prior_var
		bi_v = prior_var
		W_last_m.append(Wi_m)
		b_last_m.append(bi_m)
		W_last_v.append(Wi_v)
		b_last_v.append(bi_v)
			
		return [W_m, b_m, W_last_m, b_last_m], [W_v, b_v, W_last_v, b_last_v]
