import tensorflow as tf
import numpy as np
from copy import deepcopy
import sample_gumbel_trick
import kl_gauss_mixture

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
        
    def assign_optimizer(self, learning_rate=0.001):
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

    def assign_session(self):
        # Initializing the variables
        init = tf.global_variables_initializer()

        # launch a session
        self.sess = tf.Session()
        self.sess.run(init)

    def train(self, x_train, y_train, task_idx, no_epochs=1000, batch_size=100, display_epoch=20):
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
        prob = self.sess.run([tf.nn.softmax(self.pred)], feed_dict={self.x: x_test, self.task_idx: task_idx})[0]
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

class MFVI_NN(Cla_NN):

    def __init__(self, input_size, hidden_size, output_size, training_size,
        no_train_samples=10, no_pred_samples=50, prev_means=None, prev_log_variances=None,prev_coffs= None, learning_rate=0.001, prior_mean=0., prior_var=1.0, gauss_mixture = 1,tau = 1.0):
        super(MFVI_NN, self).__init__(input_size, hidden_size, output_size, training_size)
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.gauss_mixture = gauss_mixture
        self.tau = tau
        list_variables, self.size = self.create_weights(input_size, hidden_size, output_size, prev_means, prev_log_variances,prev_coffs)
        print("self.size ", self.size)
        self.weights = list_variables
        self.list_priors = self.create_prior(input_size, hidden_size, output_size, prev_means, prev_log_variances, prev_coffs, self.prior_mean, self.prior_var)
        self.no_layers = len(self.size) - 1
        self.no_train_samples = no_train_samples
        self.no_pred_samples = no_pred_samples
        self.pred = self._prediction(self.x, self.task_idx, self.no_pred_samples)
        self.abc = tf.div(self._KL_term(), training_size)
        self.cost = tf.div(self._KL_term(), training_size) - self._logpred(self.x, self.y, self.task_idx)
        self.assign_optimizer(learning_rate)
        self.assign_session()
        print("num gauss ", self.gauss_mixture)
        print("tau ",self.tau)
        print("num train ", no_train_samples)
        print("learning rate ", learning_rate)
        print("num pred ", self.no_pred_samples)
        print("prior mean ",self.prior_mean)
        print("prior var ",self.prior_var)
        print("size of layer ",self.size)
        print("num layer ",self.no_layers)
    
    def assign_weight(self, weight):
        for  mixture in range(self.gauss_mixture):
            #assign mean
            for i in range(self.no_layers - 1):
                self.sess.run(tf.assign(self.weights[mixture][0][0][i] , weight[mixture][0][0][i]))
                self.sess.run(tf.assign(self.weights[mixture][0][1][i], weight[mixture][0][1][i]))
            self.sess.run(tf.assign(self.weights[mixture][0][2][0], weight[mixture][0][2][0]))
            self.sess.run(tf.assign(self.weights[mixture][0][3][0] , weight[mixture][0][3][0]))

        #assign variance 
            for i in range(self.no_layers - 1):
                self.sess.run(tf.assign(self.weights[mixture][1][0][i] , weight[mixture][1][0][i]))
                self.sess.run(tf.assign(self.weights[mixture][1][1][i], weight[mixture][1][1][i]))
            self.sess.run(tf.assign(self.weights[mixture][1][2][0], weight[mixture][1][2][0]))
            self.sess.run(tf.assign(self.weights[mixture][1][3][0] , weight[mixture][1][3][0]))

        #assign coff
            if mixture != 0:
                for i in range(self.no_layers - 1):
                    self.sess.run(tf.assign(self.weights[mixture][2][0][i] , weight[mixture][2][0][i]))
                    self.sess.run(tf.assign(self.weights[mixture][2][1][i], weight[mixture][2][1][i]))
                self.sess.run(tf.assign(self.weights[mixture][2][2][0], weight[mixture][2][2][0]))
                self.sess.run(tf.assign(self.weights[mixture][2][3][0] , weight[mixture][2][3][0]))

    def _prediction(self, inputs, task_idx, no_samples):
        return self._prediction_layer(inputs, task_idx, no_samples)

    def _prediction_layer(self, inputs, task_idx, no_samples):
        K = no_samples
        act = tf.tile(tf.expand_dims(inputs, 0), [K, 1, 1])
        for i in range(self.no_layers-1):
            din = self.size[i]
            dout = self.size[i+1]
            #get w
            list_mean_w = []
            list_variance_w = []
            list_coff = []
            for mixture in range(self.gauss_mixture):
                list_mean_w.append(self.weights[mixture][0][0][i])
                list_variance_w.append(tf.exp(self.weights[mixture][1][0][i]))
                list_coff.append(self.weights[mixture][2][0][i])
            weights = sample_gumbel_trick.sample_from_gumbel_softmax_trick(list_mean_w, list_variance_w , list_coff , self.tau,self.gauss_mixture , K ,din,dout, False)
            list_mean_w = []
            list_variance_w = []
            list_coff = []
            for mixture in range(self.gauss_mixture):
                list_mean_w.append(self.weights[mixture][0][1][i])
                list_variance_w.append(tf.exp(self.weights[mixture][1][1][i]))
                list_coff.append(self.weights[mixture][2][1][i])
            biases = sample_gumbel_trick.sample_from_gumbel_softmax_trick(list_mean_w, list_variance_w , list_coff , self.tau , self.gauss_mixture ,K , din,dout, True)            
            #feed vao mang
            pre = tf.add(tf.einsum('mni,mio->mno', act, weights), biases)
            act = tf.nn.relu(pre)
        #tai layer cuoi cung
        din = self.size[-2]
        dout = self.size[-1]
        Wtask_m , Wtask_v , btask_m , btask_v , coff_m , coff_b  = [] , [] , [] , [] , [] , []
        for mixture in range(self.gauss_mixture):
            Wtask_m.append(self.weights[mixture][0][2][0])
            Wtask_v.append(tf.exp(self.weights[mixture][1][2][0]))
            btask_m.append(self.weights[mixture][0][3][0])
            btask_v.append(tf.exp(self.weights[mixture][1][3] [0]))
            coff_m.append(self.weights[mixture][2][2][0])
            coff_b.append(self.weights[mixture][2][3][0])        
        weights = sample_gumbel_trick.sample_from_gumbel_softmax_trick(Wtask_m , Wtask_v , coff_m , self.tau , self.gauss_mixture ,K , din,dout,False)
        biases = sample_gumbel_trick.sample_from_gumbel_softmax_trick(btask_m , btask_v , coff_b , self.tau ,self.gauss_mixture , K , din,dout, True)
        act = tf.expand_dims(act, 3)
        weights = tf.expand_dims(weights, 1)
        pre = tf.add(tf.reduce_sum(act * weights, 2), biases)
        return pre

    def _logpred(self, inputs, targets, task_idx):
        pred = self._prediction(inputs, task_idx, self.no_train_samples)
        targets = tf.tile(tf.expand_dims(targets, 0), [self.no_train_samples, 1, 1])
        log_lik = - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=targets))
        return log_lik

    def _KL_term(self):
        kl = 0.
        for i in range(self.no_layers-1):
            #lam cho W
            #dua cac tham so ve dang mixture theo tung layer
            din = self.size[i]
            dout = self.size[i+1]
            means_prior , cov_prior , coff_prior = [] , [] , [] 
            means_variable , cov_variable , coff_variable = [], [] , []
            for mixture in range(self.gauss_mixture):

                means_prior.append(self.list_priors[mixture][0][0][i])
                means_variable.append(self.weights[mixture][0][0][i])

                cov_prior.append(self.list_priors[mixture][1][0][i])
                cov_variable.append(self.weights[mixture][1][0][i])

                coff_prior.append(self.list_priors[mixture][2][0][i])
                coff_variable.append(self.weights[mixture][2][0][i])

            mixture_1 = [means_variable , cov_variable , coff_variable]
            means_prior = np.array(means_prior , dtype= np.float32)
            cov_prior = np.array(cov_prior , dtype= np.float32)
            coff_prior = np.array(coff_prior, dtype= np.float32)
            mixture_2 = [means_prior , cov_prior , coff_prior]
            kl += kl_gauss_mixture.upperbound_kl_divergence_mixture_gauss(mixture_1 , mixture_2)
            #lam voi bias
            means_prior , cov_prior , coff_prior = [] , [] , [] 
            means_variable , cov_variable , coff_variable = [], [] , []
            for mixture in range(self.gauss_mixture):

                means_prior.append(self.list_priors[mixture][0][1][i])
                means_variable.append(self.weights[mixture][0][1][i])

                cov_prior.append(self.list_priors[mixture][1][1][i])
                cov_variable.append(self.weights[mixture][1][1][i])

                coff_prior.append(self.list_priors[mixture][2][1][i])
                coff_variable.append(self.weights[mixture][2][1][i])

            mixture_1 = [means_variable , cov_variable , coff_variable]
            means_prior = np.array(means_prior , dtype= np.float32)
            cov_prior = np.array(cov_prior , dtype= np.float32)
            coff_prior = np.array(coff_prior, dtype= np.float32)
            mixture_2 = [means_prior , cov_prior , coff_prior]
            kl += kl_gauss_mixture.upperbound_kl_divergence_mixture_gauss(mixture_1 , mixture_2)

        no_tasks = len(self.list_priors[0][0][2])
        print("len head Network >>>>>>>>>>>>>>>>>>>>>>>>>>>>", no_tasks)

        #lam cho W, b last
        for i in range(no_tasks):
            #lam cho W_last
            din = self.size[-2]
            dout = self.size[-1]
            means_prior , cov_prior , coff_prior = [] , [] , [] 
            means_variable , cov_variable , coff_variable = [], [] , []
            for mixture in range(self.gauss_mixture):

                means_prior.append(self.list_priors[mixture][0][2][i])
                means_variable.append(self.weights[mixture][0][2][i])

                cov_prior.append(self.list_priors[mixture][1][2][i])
                cov_variable.append(self.weights[mixture][1][2][i])

                coff_prior.append(self.list_priors[mixture][2][2][i])
                coff_variable.append(self.weights[mixture][2][2][i])

            mixture_1 = [means_variable , cov_variable , coff_variable]
            means_prior = np.array(means_prior , dtype= np.float32)
            cov_prior = np.array(cov_prior , dtype= np.float32)
            coff_prior = np.array(coff_prior, dtype= np.float32)
            mixture_2 = [means_prior , cov_prior , coff_prior]
            kl += kl_gauss_mixture.upperbound_kl_divergence_mixture_gauss(mixture_1 , mixture_2)

            #lam cho b_last
            means_prior , cov_prior , coff_prior = [] , [] , [] 
            means_variable , cov_variable , coff_variable = [], [] , []
            for mixture in range(self.gauss_mixture):

                means_prior.append(self.list_priors[mixture][0][3][i])
                means_variable.append(self.weights[mixture][0][3][i])

                cov_prior.append(self.list_priors[mixture][1][3][i])
                cov_variable.append(self.weights[mixture][1][3][i])

                coff_prior.append(self.list_priors[mixture][2][3][i])
                coff_variable.append(self.weights[mixture][2][3][i])

            mixture_1 = [means_variable , cov_variable , coff_variable]
            means_prior = np.array(means_prior , dtype= np.float32)
            cov_prior = np.array(cov_prior , dtype= np.float32)
            coff_prior = np.array(coff_prior, dtype= np.float32)
            mixture_2 = [means_prior , cov_prior , coff_prior]
            kl += kl_gauss_mixture.upperbound_kl_divergence_mixture_gauss(mixture_1 , mixture_2)

        return kl

    def create_weights(self, in_dim, hidden_size, out_dim, prev_weights, prev_variances, prev_coff):
        #khoi tao cho ca truong hop tong quat  voi so nguyen k
        hidden_size = deepcopy(hidden_size)
        hidden_size.append(out_dim)
        hidden_size.insert(0, in_dim)
        no_layers = len(hidden_size) - 1
        results = []
        for mixture in range(self.gauss_mixture):
            W_m = []
            b_m = []
            W_last_m = []
            b_last_m = []
            W_v = []
            b_v = []
            W_last_v = []
            b_last_v = []
            W_coff = []
            b_coff = []
            W_last_coff=[]
            b_last_coff = []
            #get weight for share parameter
            for i in range(no_layers-1):
                #mang tren cac layer
                din = hidden_size[i]
                dout = hidden_size[i+1]
                
                if prev_weights is None:
                    #khoi tao gia tri start point
                    Wi_m_val = tf.truncated_normal([din, dout], stddev=0.1)
                    bi_m_val = tf.truncated_normal([dout], stddev=0.1)
                    Wi_v_val = tf.constant(-6.0, shape=[din, dout])
                    bi_v_val = tf.constant(-6.0, shape=[dout])
                else:
                    Wi_m_val = prev_weights[mixture][0][i]
                    bi_m_val = prev_weights[mixture][1][i]
                    if prev_variances is None:
                        Wi_v_val = tf.constant(-6.0, shape=[din, dout])
                        bi_v_val = tf.constant(-6.0, shape=[dout])
                    else:
                        Wi_v_val = prev_variances[mixture][0][i]
                        bi_v_val = prev_variances[mixture][1][i]
    
                Wi_m = tf.Variable(Wi_m_val)
                bi_m = tf.Variable(bi_m_val)
                Wi_v = tf.Variable(Wi_v_val)
                bi_v = tf.Variable(bi_v_val)
                W_m.append(Wi_m)
                b_m.append(bi_m)
                W_v.append(Wi_v)
                b_v.append(bi_v)
            
            #get coff for share parameter
            for i in range(no_layers - 1):
                din = hidden_size[i]
                dout = hidden_size[i+1]
                if mixture == 0:
                    Wi_coff_val = np.full(fill_value=0. , shape=(din , dout)).astype(np.float32)
                    bi_coff_val = np.full(fill_value=0. , shape=(dout)).astype(np.float32)
                    Wi_coff = tf.constant(Wi_coff_val)
                    bi_coff = tf.constant(bi_coff_val)
                else:
                    if prev_coff is None:
                        Wi_coff_val = np.random.normal(0., 1., size=(din , dout)).astype(np.float32)
                        bi_coff_val = np.random.normal(0., 1. , size=(dout)).astype(np.float32)   
                    else:
                        Wi_coff_val = prev_coff[mixture][0][i]
                        bi_coff_val = prev_coff[mixture][1][i]
                    Wi_coff = tf.Variable(Wi_coff_val )
                    bi_coff = tf.Variable(bi_coff_val )
                W_coff.append(Wi_coff)
                b_coff.append(bi_coff)
            
            # if there are previous tasks
            if prev_weights is not None and prev_variances is not None:
                #task thu 2 tro di
                prev_Wlast_m = prev_weights[mixture][2]
                prev_blast_m = prev_weights[mixture][3]
                prev_Wlast_v = prev_variances[mixture][2]
                prev_blast_v = prev_variances[mixture][3]
                
                prev_Wlast_coff = prev_coff[mixture][2]
                prev_blast_coff = prev_coff[mixture][3]
                no_prev_tasks = len(prev_Wlast_m)
                W_i_m = prev_Wlast_m[0]
                b_i_m = prev_blast_m[0]
                Wi_m = tf.Variable(W_i_m)
                bi_m = tf.Variable(b_i_m)

                W_i_v = prev_Wlast_v[0]
                b_i_v = prev_blast_v[0]
                Wi_v = tf.Variable(W_i_v)
                bi_v = tf.Variable(b_i_v)
                
                W_last_m.append(Wi_m)
                b_last_m.append(bi_m)
                W_last_v.append(Wi_v)
                b_last_v.append(bi_v)
                
                #get coff for last parameter
                W_i_coff = prev_Wlast_coff[0]
                b_i_coff = prev_blast_coff[0]

                if mixture ==0:
                    Wi_coff_val = np.full(fill_value = 0. , shape=W_i_coff.shape).astype(np.float32)
                    bi_coff_val = np.full(fill_value = 0. , shape=b_i_coff.shape).astype(np.float32)
                    Wi_coff = tf.constant(Wi_coff_val)
                    bi_coff = tf.constant(bi_coff_val)
                else:                
                    Wi_coff = tf.Variable(W_i_coff)
                    bi_coff = tf.Variable(b_i_coff)
                
                W_last_coff.append(Wi_coff)
                b_last_coff.append(bi_coff)
    
            din = hidden_size[-2]
            dout = hidden_size[-1]
    
            if prev_weights is not None and prev_variances is None:
                #truong hop task 1
                Wi_m_val = prev_weights[mixture][2][0]
                bi_m_val = prev_weights[mixture][3][0]

                Wi_v_val = tf.constant(-6.0, shape=[din, dout])
                bi_v_val = tf.constant(-6.0, shape=[dout])
                Wi_m = tf.Variable(Wi_m_val)
                bi_m = tf.Variable(bi_m_val)
                Wi_v = tf.Variable(Wi_v_val)
                bi_v = tf.Variable(bi_v_val)

                if mixture == 0:
                    Wi_coff_val = np.full(fill_value =0. , shape=(din , dout)).astype(np.float32)
                    bi_coff_val = np.full(fill_value= 0. , shape=( dout)).astype(np.float32)
                    Wi_coff = tf.constant(Wi_coff_val)
                    bi_coff = tf.constant(bi_coff_val)
                else:
                    Wi_coff_val = np.random.normal(0., 1. ,  size=(din , dout)).astype(np.float32)
                    bi_coff_val = np.random.normal(0., 1. , size=( dout)).astype(np.float32)
                    Wi_coff = tf.Variable(Wi_coff_val)
                    bi_coff = tf.Variable(bi_coff_val)

                W_last_m.append(Wi_m)
                b_last_m.append(bi_m)
                W_last_v.append(Wi_v)
                b_last_v.append(bi_v)
                
                W_last_coff.append(Wi_coff )
                b_last_coff.append(bi_coff )

            m =    [W_m, b_m, W_last_m, b_last_m]
            v =    [W_v, b_v, W_last_v, b_last_v]
            c =     [W_coff , b_coff , W_last_coff , b_last_coff]
            results.append( [m , v ,c ])
        return results, hidden_size


    def create_prior(self, in_dim, hidden_size, out_dim, prev_weights, prev_variances,prev_coff, prior_mean, prior_var):
        hidden_size = deepcopy(hidden_size)
        hidden_size.append(out_dim)
        hidden_size.insert(0, in_dim)
        no_layers = len(hidden_size) - 1
        results = []
        for mixture in range(self.gauss_mixture):
            W_m = []
            b_m = []
            W_last_m = []
            b_last_m = []
            W_v = []
            b_v = []
            W_last_v = []
            b_last_v = []
            W_coff = []
            b_coff = []
            W_last_coff=[]
            b_last_coff = []
            for i in range(no_layers-1):
                if prev_weights is not None and prev_variances is not None:
                    Wi_m = prev_weights[mixture][0][i]
                    bi_m = prev_weights[mixture][1][i]
                    Wi_v = np.exp(prev_variances[mixture][0][i])
                    bi_v = np.exp(prev_variances[mixture][1][i])

                    Wi_coff = prev_coff[mixture][0][i]
                    bi_coff = prev_coff[mixture][1][i]
                else:
                    Wi_m = np.full(shape  = (hidden_size[i],hidden_size[i+1]) , fill_value = prior_mean)
                    bi_m = np.full(shape  = (hidden_size[i+1],) , fill_value = prior_mean)
                    Wi_v = np.full(shape  = (hidden_size[i],hidden_size[i+1]) , fill_value = prior_var)
                    bi_v = np.full(shape  = (hidden_size[i+1],) , fill_value = prior_var)
                    #chinh o cho nay
                    if mixture == 0:
                        Wi_coff = np.full(shape  = (hidden_size[i],hidden_size[i+1]) , fill_value = 0.).astype(np.float32)
                        bi_coff = np.full(shape  = (hidden_size[i+1],) , fill_value = 0.).astype(np.float32)
                    else:
                        Wi_coff = np.full(shape  = (hidden_size[i],hidden_size[i+1]) , fill_value = 0.).astype(np.float32)
                        bi_coff = np.full(shape  = (hidden_size[i+1],) , fill_value = 0.).astype(np.float32) 
                W_m.append(Wi_m)
                b_m.append(bi_m)
                W_v.append(Wi_v)
                b_v.append(bi_v)

                W_coff.append(Wi_coff)
                b_coff.append(bi_coff)

            # if there are previous tasks
            if prev_weights is not None and prev_variances is not None:
                prev_Wlast_m = prev_weights[mixture][2]
                prev_blast_m = prev_weights[mixture][3]
                prev_Wlast_v = prev_variances[mixture][2]
                prev_blast_v = prev_variances[mixture][3]

                prev_Wlast_coff = prev_coff[mixture][2]
                prev_blast_coff = prev_coff[mixture][3]

                no_prev_tasks = len(prev_Wlast_m)
                Wi_m = prev_Wlast_m[0]
                bi_m = prev_blast_m[0]
                Wi_v = np.exp(prev_Wlast_v[0])
                bi_v = np.exp(prev_blast_v[0])
                
                W_last_m.append(Wi_m)
                b_last_m.append(bi_m)
                W_last_v.append(Wi_v)
                b_last_v.append(bi_v)

                Wi_coff = prev_Wlast_coff[0]
                bi_coff = prev_blast_coff[0]

                W_last_coff.append(Wi_coff)
                b_last_coff.append(bi_coff)
            else:
                #truong hop task dau tien
                Wi_m = np.full(shape  = (hidden_size[-2],hidden_size[-1]) , fill_value = prior_mean).astype(np.float32)
                bi_m = np.full(shape  = (hidden_size[-1],) , fill_value = prior_mean).astype(np.float32)
                Wi_v = np.full(shape  = (hidden_size[-2],hidden_size[-1]) , fill_value = prior_var).astype(np.float32)
                bi_v = np.full(shape  = (hidden_size[-1],) , fill_value = prior_var).astype(np.float32)

                if mixture == 0:
                    Wi_coff = np.full(shape  = (hidden_size[-2],hidden_size[-1]) , fill_value = 0.).astype(np.float32)
                    bi_coff = np.full(shape  = (hidden_size[-1],) , fill_value = 0.).astype(np.float32)
                else:
                    Wi_coff = np.full(shape  = (hidden_size[-2],hidden_size[-1]) , fill_value = 0.).astype(np.float32)
                    bi_coff = np.full(shape  = (hidden_size[-1],) , fill_value = 0.).astype(np.float32)

                W_last_m.append(Wi_m)
                b_last_m.append(bi_m)
                W_last_v.append(Wi_v)
                b_last_v.append(bi_v)

                W_last_coff.append(Wi_coff)
                b_last_coff.append(bi_coff)

            m = [W_m, b_m, W_last_m, b_last_m]
            v = [W_v, b_v, W_last_v, b_last_v]
            c = [W_coff , b_coff , W_last_coff , b_last_coff]

            results.append([m,v,c])
            
        return results