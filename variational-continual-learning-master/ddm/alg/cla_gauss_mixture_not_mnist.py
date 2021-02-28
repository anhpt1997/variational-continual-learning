import tensorflow as tf
import numpy as np
from copy import deepcopy
import sample_gumbel_trick
import kl_gauss_mixture
#chi co phan share la mixture, head van la gauss
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

    def train(self, x_train, y_train, task_idx, no_epochs=1000, batch_size=100, display_epoch=30):
        N = x_train.shape[0]
        if batch_size > N:
            batch_size = N

        sess = self.sess
        costs = []
        # Training cycle
        for epoch in range(no_epochs):
            if epoch %display_epoch == 0:
                print(epoch , self.sess.run(self.abc))
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
            # if epoch % display_epoch == 0:
            #     print("Epoch:", '%04d' % (epoch+1), "cost=", \
            #         "{:.9f}".format(avg_cost))
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
        no_train_samples=10, no_pred_samples=50, prev_means=None, prev_log_variances=None,prev_coffs= None, learning_rate=0.001, prior_mean=0., prior_var=1.0, gauss_mixture = 1,tau = 1.0,num_div = 20):
        super(MFVI_NN, self).__init__(input_size, hidden_size, output_size, training_size)
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.gauss_mixture = gauss_mixture
        self.tau = tau
        self.training_size = training_size
        list_variables, self.size = self.create_weights(input_size, hidden_size, output_size, prev_means, prev_log_variances,prev_coffs)
        self.weights = list_variables
        self.list_priors = self.create_prior(input_size, hidden_size, output_size, prev_means, prev_log_variances, prev_coffs, self.prior_mean, self.prior_var)
        self.no_layers = len(self.size) - 1
        self.no_train_samples = no_train_samples
        self.no_pred_samples = no_pred_samples
        self.pred = self._prediction(self.x, self.task_idx, self.no_pred_samples)
        self.abc = tf.div(self._KL_term(), training_size)
        # self.cost = tf.div(self._KL_term(), training_size) - self._logpred(self.x, self.y, self.task_idx)
        self.cost = tf.div(self._KL_term(), training_size) - self.compute_all_log_pred(self.x, self.y, self.task_idx ,num_div)
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

    def _prediction(self, inputs, task_idx, no_samples):
        return self._prediction_layer(inputs, task_idx, no_samples)

    def _prediction_layer(self, inputs, task_idx, no_samples):
        K = no_samples
        act = tf.tile(tf.expand_dims(inputs, 0), [K, 1, 1])
        weight_share = self.weights[0]
        weight_head = self.weights[1]
        for i in range(self.no_layers-1):
            din = self.size[i]
            dout = self.size[i+1]
            #get w
            list_mean_w = []
            list_variance_w = []
            list_coff = []
            for mixture in range(self.gauss_mixture):
                list_mean_w.append(weight_share[mixture][0][0][i])
                list_variance_w.append(tf.exp(weight_share[mixture][1][0][i]))
                list_coff.append(weight_share[mixture][2][0][i])
            weights = sample_gumbel_trick.sample_from_gumbel_softmax_trick(list_mean_w, list_variance_w , list_coff , self.tau,self.gauss_mixture , K ,din,dout, False)
            list_mean_w = []
            list_variance_w = []
            list_coff = []
            for mixture in range(self.gauss_mixture):
                list_mean_w.append(weight_share[mixture][0][1][i])
                list_variance_w.append(tf.exp(weight_share[mixture][1][1][i]))
                list_coff.append(weight_share[mixture][2][1][i])
            biases = sample_gumbel_trick.sample_from_gumbel_softmax_trick(list_mean_w, list_variance_w , list_coff , self.tau , self.gauss_mixture ,K , din,dout, True)            
            #feed vao mang
            pre = tf.add(tf.einsum('mni,mio->mno', act, weights), biases)
            act = tf.nn.relu(pre)
        #tai layer cuoi cung
        din = self.size[-2]
        dout = self.size[-1]

        eps_w = tf.random_normal((K, din, dout), 0, 1, dtype=tf.float32)
        eps_b = tf.random_normal((K, 1, dout), 0, 1, dtype=tf.float32)

        Wtask_m = tf.gather(weight_head[0][0], task_idx)
        Wtask_v = tf.gather(weight_head[1][0], task_idx)
        btask_m = tf.gather(weight_head[0][1], task_idx)
        btask_v = tf.gather(weight_head[1][1], task_idx)
        weights = tf.add(tf.multiply(eps_w, tf.exp(0.5*Wtask_v)), Wtask_m)
        biases = tf.add(tf.multiply(eps_b, tf.exp(0.5*btask_v)), btask_m)

        act = tf.expand_dims(act, 3)
        weights = tf.expand_dims(weights, 1)
        pre = tf.add(tf.reduce_sum(act * weights, 2), biases)
        return pre

    def _logpred(self, inputs, targets, task_idx):
        pred = self._prediction(inputs, task_idx, self.no_train_samples)
        targets = tf.tile(tf.expand_dims(targets, 0), [self.no_train_samples, 1, 1])
        log_lik = - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=targets))
        return log_lik

    def compute_all_log_pred(self, x , y , task_idx , num_div):
        print("num div ",num_div)
        list_pred = []
        total_batch = num_div
        print("total batch ", total_batch)
        for i in range(total_batch):
            start_ind = i* 1000
            end_ind = (i+1) *1000
            batch_x = x[start_ind:end_ind, :]
            batch_y = y[start_ind:end_ind, :]
            list_pred.append(self._logpred(batch_x , batch_y , task_idx))
        return tf.reduce_mean(list_pred)


    def _KL_term(self):
        kl = 0

        variable_weight_share = self.weights[0]
        variable_weight_head = self.weights[1]
        prior_weight_share = self.list_priors[0]
        prior_weight_head = self.list_priors[1]

        for i in range(self.no_layers-1):

            means_prior , cov_prior , coff_prior = [] , [] , []
            means_variable , cov_variable , coff_variable = [], [] , []
            for mixture in range(self.gauss_mixture):

                means_prior.append(prior_weight_share[mixture][0][0][i])
                means_variable.append(variable_weight_share[mixture][0][0][i])

                cov_prior.append(prior_weight_share[mixture][1][0][i])
                cov_variable.append(variable_weight_share[mixture][1][0][i])

                coff_prior.append(prior_weight_share[mixture][2][0][i])
                coff_variable.append(variable_weight_share[mixture][2][0][i])

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

                means_prior.append(prior_weight_share[mixture][0][1][i])
                means_variable.append(variable_weight_share[mixture][0][1][i])

                cov_prior.append(prior_weight_share[mixture][1][1][i])
                cov_variable.append(variable_weight_share[mixture][1][1][i])

                coff_prior.append(prior_weight_share[mixture][2][1][i])
                coff_variable.append(variable_weight_share[mixture][2][1][i])

            mixture_1 = [means_variable , cov_variable , coff_variable]
            means_prior = np.array(means_prior , dtype= np.float32)
            cov_prior = np.array(cov_prior , dtype= np.float32)
            coff_prior = np.array(coff_prior, dtype= np.float32)
            mixture_2 = [means_prior , cov_prior , coff_prior]
            kl += kl_gauss_mixture.upperbound_kl_divergence_mixture_gauss(mixture_1 , mixture_2)
        no_tasks = len(prior_weight_head[0][0])
        print("no_task >>>>>><<<<<<< ", no_tasks)
        #lam cho W, b last
        list_kl = []
        for i in range(no_tasks):
            a = 0.
            #lam cho W_last
            din = self.size[-2]
            dout = self.size[-1]

            m, v = variable_weight_head[0][0][i], variable_weight_head[1][0][i]
            m0, v0 = prior_weight_head[0][0][i], prior_weight_head[1][0][i]

            const_term = -0.5 * dout * din
            log_std_diff = 0.5 * tf.reduce_sum(np.log(v0) - v)
            mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            #lam cho b_last
            m, v = variable_weight_head[0][1][i], variable_weight_head[1][1][i]
            m0, v0 = prior_weight_head[0][1][i], prior_weight_head[1][1][i]

            const_term = -0.5 * dout
            log_std_diff = 0.5 * tf.reduce_sum(np.log(v0) - v)
            mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            list_kl.append(a)
        return kl

    def create_weights(self, in_dim, hidden_size, out_dim, prev_weights, prev_variances, prev_coff):

        prev_weight_share  =  prev_weights[0]
        prev_weight_head = prev_weights[1]
        if prev_variances is not None:
            prev_variance_share = prev_variances[0]
            prev_variance_head = prev_variances[1]
            prev_coff_share = prev_coff
        #head khong phai la mixture khong can coff

        hidden_size = deepcopy(hidden_size)
        hidden_size.append(out_dim)
        hidden_size.insert(0, in_dim)
        no_layers = len(hidden_size) - 1
        weight_share = []
        weight_head =  []
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
                    Wi_m_val = prev_weight_share[mixture][0][i]
                    bi_m_val = prev_weight_share[mixture][1][i]
                    if prev_variances is None:
                        Wi_v_val = tf.constant(-6.0, shape=[din, dout])
                        bi_v_val = tf.constant(-6.0, shape=[dout])
                    else:
                        Wi_v_val = prev_variance_share[mixture][0][i]
                        bi_v_val = prev_variance_share[mixture][1][i]
    
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
                        Wi_coff_val = prev_coff_share[mixture][0][i]
                        bi_coff_val = prev_coff_share[mixture][1][i]
                    Wi_coff = tf.Variable(Wi_coff_val )
                    bi_coff = tf.Variable(bi_coff_val )
                W_coff.append(Wi_coff)
                b_coff.append(bi_coff)

            m =    [W_m, b_m]
            v =    [W_v, b_v]
            c =     [W_coff , b_coff]
            weight_share.append( [m , v ,c ])
        #create start point weight head
        # if there are previous tasks
        if prev_weights is not None and prev_variances is not None:
            prev_Wlast_m = prev_weight_head[0]
            prev_blast_m = prev_weight_head[1]
            prev_Wlast_v = prev_variance_head[0]
            prev_blast_v = prev_variance_head[1]

            no_prev_tasks = len(prev_Wlast_m)
            for i in range(no_prev_tasks):
                W_i_m = prev_Wlast_m[i]
                b_i_m = prev_blast_m[i]
                Wi_m = tf.Variable(W_i_m)
                bi_m = tf.Variable(b_i_m)

                W_i_v = prev_Wlast_v[i]
                b_i_v = prev_blast_v[i]
                Wi_v = tf.Variable(W_i_v)
                bi_v = tf.Variable(b_i_v)
                
                W_last_m.append(Wi_m)
                b_last_m.append(bi_m)
                W_last_v.append(Wi_v)
                b_last_v.append(bi_v)

        din = hidden_size[-2]
        dout = hidden_size[-1]

        # if point estimate is supplied
        # if prev_weights is not None and prev_variances is None:
        #     Wi_m_val = prev_weight_head[0]
        #     bi_m_val = prev_weight_head[1]
        # else:
        Wi_m_val = tf.truncated_normal([din, dout], stddev=0.1)
        bi_m_val = tf.truncated_normal([dout], stddev=0.1)

        Wi_v_val = tf.constant(-6.0, shape=[din, dout])
        bi_v_val = tf.constant(-6.0, shape=[dout])
        Wi_m = tf.Variable(Wi_m_val)
        bi_m = tf.Variable(bi_m_val)
        Wi_v = tf.Variable(Wi_v_val)
        bi_v = tf.Variable(bi_v_val)

        W_last_m.append(Wi_m)
        b_last_m.append(bi_m)
        W_last_v.append(Wi_v)
        b_last_v.append(bi_v)

        m = [W_last_m , b_last_m]
        v = [W_last_v , b_last_v]

        weight_head = [ m , v]
        results = [ weight_share , weight_head ]
        return results, hidden_size

    def create_prior(self, in_dim, hidden_size, out_dim, prev_weights, prev_variances,prev_coff, prior_mean, prior_var):
        
        hidden_size = deepcopy(hidden_size)
        hidden_size.append(out_dim)
        hidden_size.insert(0, in_dim)
        no_layers = len(hidden_size) - 1

        prev_weight_share  =  prev_weights[0]
        prev_weight_head = prev_weights[1]
        if prev_variances is not None:
            prev_variance_share = prev_variances[0]
            prev_variance_head = prev_variances[1]
            prev_coff_share = prev_coff
        weight_share = []
        weight_head = []
        
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
            for i in range(no_layers-1):
                if prev_weights is not None and prev_variances is not None:
                    Wi_m = prev_weight_share[mixture][0][i]
                    bi_m = prev_weight_share[mixture][1][i]
                    Wi_v = np.exp(prev_variance_share[mixture][0][i])
                    bi_v = np.exp(prev_variance_share[mixture][1][i])

                    Wi_coff = prev_coff_share[mixture][0][i]
                    bi_coff = prev_coff_share[mixture][1][i]
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
            m = [W_m, b_m]
            v = [W_v, b_v]
            c = [W_coff , b_coff]

            weight_share.append([m,v,c])
        # if there are previous tasks
        if prev_weights is not None and prev_variances is not None:
            prev_Wlast_m = prev_weight_head[0]
            prev_blast_m = prev_weight_head[1]
            prev_Wlast_v = prev_variance_head[0]
            prev_blast_v = prev_variance_head[1]

            no_prev_tasks = len(prev_Wlast_m)
            for i in range(no_prev_tasks):
                Wi_m = prev_Wlast_m[i]
                bi_m = prev_blast_m[i]
                Wi_v = np.exp(prev_Wlast_v[i])
                bi_v = np.exp(prev_blast_v[i])
                
                W_last_m.append(Wi_m)
                b_last_m.append(bi_m)
                W_last_v.append(Wi_v)
                b_last_v.append(bi_v)

        Wi_m = np.full(shape  = (hidden_size[-2],hidden_size[-1]) , fill_value = prior_mean).astype(np.float32)
        bi_m = np.full(shape  = (hidden_size[-1],) , fill_value = prior_mean).astype(np.float32)
        Wi_v = np.full(shape  = (hidden_size[-2],hidden_size[-1]) , fill_value = prior_var).astype(np.float32)
        bi_v = np.full(shape  = (hidden_size[-1],) , fill_value = prior_var).astype(np.float32)

        W_last_m.append(Wi_m)
        b_last_m.append(bi_m)
        W_last_v.append(Wi_v)
        b_last_v.append(bi_v)

        m = [W_last_m , b_last_m]
        v = [W_last_v , b_last_v]

        weight_head = [m,v]
        return [weight_share , weight_head]