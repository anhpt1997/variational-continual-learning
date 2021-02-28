import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
PI = np.pi

def sample_gauss(mean, sigma):
	ep = np.random.normal()
	return mean + ep * sigma

def compute_probabilities_gauss(array, mean , sigma):
	return 1. / (sigma * tf.sqrt(2 * PI)) * tf.exp( -0.5 * (array - mean)**2 / (sigma)**2 )

def compute_list_progauss(array , mean , sigma):
	pass

def compute_probabilities_gaussmixture(array, list_pi, list_mean, list_sigma):
	num_component = tf.convert_to_tensor(list_pi).shape[0]
	result = 0
	for i in range(num_component):
		result += list_pi[i] * compute_probabilities_gauss(array, list_mean[i], list_sigma[i])
	return result

def sample_categorical(list_pi, tau):
	shape = tf.convert_to_tensor(list_pi).shape
	sample_uniform = tf.random.uniform(shape)
	gumbel_sample = -tf.log(-tf.log(sample_uniform))
	return tf.nn.softmax( 1. / tau * (gumbel_sample + tf.log(list_pi)), axis= 0)

def sample_gauss_mixture(list_pi , list_mean, list_sigma, tau):
	sample_cate = sample_categorical(list_pi, tau)
	list_sample_gauss = []
	num_component = list_pi.shape[0]
	for i in range(num_component):
		list_sample_gauss.append(sample_gauss(list_mean[i],list_sigma[i]))
	s_gauss = tf.convert_to_tensor(list_sample_gauss)
	return tf.reduce_sum(s_gauss * sample_cate , axis = 0)


def compute_kl_divergence_between_2_gauss(N, mean_1, sigma_1, mean_2, sigma_2):
	result = 0
	for i in range(N):
		sample = sample_gauss(mean_1, sigma_1)
		log_gauss_1 = tf.log(compute_probabilities_gauss(sample,mean_1 , sigma_1))
		log_gauss_2 = tf.log(compute_probabilities_gauss(sample,mean_2, sigma_2))
		result += log_gauss_1
		result -= log_gauss_2
	return tf.abs(result / N)

def compute_kl_divergence_between_gauss_gaussmixture(N, mean_1, sigma_1, list_pi, list_mean , list_sigma):
	result = 0
	for i in range(N):
		sample = sample_gauss(mean_1, sigma_1)
		log_gauss_1 = tf.log(compute_probabilities_gauss(sample,mean_1 , sigma_1))
		log_gauss_2 = tf.log(compute_probabilities_gaussmixture(sample , list_pi , list_mean ,list_sigma))
		result += log_gauss_1
		result -= log_gauss_2
	return tf.abs(result / N)

def compute_kl_divergence_between_gaussmixture_gaussmixture(N,tau, list_variable_pi, list_variable_mean, list_variable_sigma, list_prior_pi, list_prior_mean, list_prior_sigma):
	result = 0
	for i in range(N):
		sample = sample_gauss_mixture(list_variable_pi, list_variable_mean, list_variable_sigma, tau)
		log_gauss_1 = tf.log(compute_probabilities_gaussmixture(sample , list_variable_pi,list_variable_mean, list_variable_sigma))
		log_gauss_2 = tf.log(compute_probabilities_gaussmixture(sample , list_prior_pi , list_prior_mean ,list_prior_sigma))
		result += log_gauss_1
		result -= log_gauss_2
	return tf.abs(result / N)   

#true dis
list_pi = [0.25 , 0.25, 0.25 , 0.25]
list_mean = [-1. , 0., 1. , 2.]
list_sigma = [0.2, 0.2, 0.2, 0.2 ]

#approximate dis
#using gauss
mean_1 , rho_1 = tf.Variable(0.2) , tf.Variable(-3.)
sigma_1 = tf.exp(rho_1)

#using gauss mixture 2 component
tf.set_random_seed(1.)
list_pi_v = tf.Variable([0.5 , 0.5])
list_mean_v = tf.Variable([0.3, +1.3])
rho_v = tf.Variable([-3. , -3.])
sigma_v = tf.exp(rho_v)
pi_softmax_v = tf.nn.softmax(list_pi_v, axis = 0)

session = tf.Session()
# print(session.run(sigma_v))
# for i in range(10):
# 	print(session.run(sample_gauss_mixture(pi_softmax_v , list_mean_v, sigma_v, 0.01)))
loss = compute_kl_divergence_between_gaussmixture_gaussmixture(100,0.01, pi_softmax_v, list_mean_v, sigma_v, list_pi, list_mean, list_sigma)
loss_1 = compute_kl_divergence_between_gauss_gaussmixture(100, mean_1, sigma_1 , list_pi , list_mean, list_sigma)

op = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)
op_1  = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss_1)
session.run(tf.global_variables_initializer())

for i in range(5000):
    if i % 100 == 0:
    	print(i ,"loss", session.run(loss), session.run(list_mean_v),session.run(sigma_v), session.run(pi_softmax_v))
    session.run(op)


for i in range(500):
    if i % 10 == 0:
    	print(i ,"loss_1", session.run(loss_1), session.run([mean_1 , sigma_1]))
    session.run(op_1)


x = np.arange(-3,3,0.01 , dtype=np.float32)
y_2 = session.run(compute_probabilities_gauss(x , session.run(mean_1) , session.run(sigma_1)))

y = session.run(compute_probabilities_gaussmixture(x , list_pi , list_mean , list_sigma))
y_1 = session.run(compute_probabilities_gaussmixture(x, session.run(pi_softmax_v) , session.run(list_mean_v), session.run(sigma_v)))
plt.plot(x , y, label = "true distribution")
plt.plot(x, y_1, label = "gauss-mixture")
plt.plot(x, y_2, label = "gauss")
plt.legend()
plt.show()