import tensorflow as tf
import numpy as np
import time

np.random.seed(1)
tf.set_random_seed(1)
s =tf.Session()
x = tf.Variable(10.)
y= tf.Variable(10.)
z = tf.Variable(10.)
loss = x**2 + y**2 + z**2
l1  = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(loss = loss , var_list = [x ])
l2  = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(loss = loss , var_list = [y])
l3  = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(loss = loss , var_list = [z])
l = [l1 , l2 , l3 ]
s.run(tf.global_variables_initializer())
s_t = time.time()
for i in range(10000):
	s.run(l)
e_t = time.time()
print(s.run([x , y, z]))
print(e_t - s_t)