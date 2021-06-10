import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()
tf.set_random_seed(2)
np.random.seed(2)


class DDPG(object):
    def __init__(
        self,
        a_dim,
        s_dim,
        a_bound,
        hidden_dim=30,
        LR_A=0.001,
        LR_C=0.002,
        GAMMA=0.9,
        memory_size=10000,
        batch_size=32,
        TAU=0.01
    ):
        self.n_adim = a_dim
        self.n_sdim = s_dim
        self.a_bound = a_bound
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = np.zeros((memory_size, 2 * s_dim + a_dim + 1), dtype=np.float)
        self.memory_counter = 0
        self.sess = tf.Session()
        # build network
        self.S = tf.placeholder(tf.float32, [None, s_dim], 'S')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 'S_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        with tf.variable_scope('Actor'):
            self.a = self.build_actor_net(self.S, 'eval', True)
            a_ = self.build_actor_net(self.S_, 'target', False)
        with tf.variable_scope('Critic'):
            q = self.build_critic_net(self.S, self.a, 'eval', True)
            q_ = self.build_critic_net(self.S_, a_, 'target', False)
        # retrieve parameters from network
        ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Actor/eval')
        at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Actor/target')
        ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Critic/eval')
        ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Critic/target')
        # soft_replace
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in
                             zip(at_params + ct_params, ae_params + ce_params)]
        # train operator
        q_target = self.R + GAMMA * q_
        loss = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain_op = tf.train.AdamOptimizer(LR_C).minimize(loss, var_list=ce_params)
        
        exp_v = tf.reduce_mean(q)
        self.atrain_op = tf.train.AdamOptimizer(LR_A).minimize(-exp_v, var_list=ae_params)
        # sess initialization
        self.sess.run(tf.global_variables_initializer())
    
    def build_actor_net(self, s, scope=None, trainable=True):
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(inputs=s, units=self.hidden_dim, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0, 0.1),
                                 bias_initializer=tf.constant_initializer(0.1), name='l1', trainable=trainable)
            l2 = tf.layers.dense(inputs=l1, units=self.hidden_dim/2, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0, 0.1),
                                 bias_initializer=tf.constant_initializer(0.1), name='l2', trainable=trainable)
            actions = tf.layers.dense(inputs=l2, units=self.n_adim, activation=tf.nn.tanh,
                                      kernel_initializer=tf.random_normal_initializer(0, 0.1),
                                      bias_initializer=tf.constant_initializer(0.1), name='l3', trainable=trainable)
            return tf.multiply(actions, self.a_bound, name='scaled_action')
    
    def build_critic_net(self, s, a, scope=None, trainable=True):
        with tf.variable_scope(scope):
            hidden_units = self.hidden_dim  # small problem: 30
            w1_a = tf.get_variable('w1_a', [self.n_adim, hidden_units],
                                   initializer=tf.random_normal_initializer(0, 0.1), trainable=trainable)
            w1_s = tf.get_variable('w1_s', [self.n_sdim, hidden_units],
                                   initializer=tf.random_normal_initializer(0, 0.1), trainable=trainable)
            b1 = tf.get_variable('b1', [1, hidden_units], initializer=tf.constant_initializer(0.1), trainable=trainable)
            net = tf.nn.relu(tf.matmul(a, w1_a) + tf.matmul(s, w1_s) + b1)
            l2 = tf.layers.dense(inputs=net, units=self.hidden_dim / 2, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0, 0.1),
                                 bias_initializer=tf.constant_initializer(0.1), name='l2', trainable=trainable)
            return tf.layers.dense(l2, 1, trainable=trainable)
    
    def choose_action(self, s):
        return self.sess.run(self.a, feed_dict={self.S: s[np.newaxis, :]})[0]  # deterministic choosing
    
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def learn(self):
        self.sess.run(self.soft_replace)
        
        indices = np.random.choice(self.memory_counter if self.memory_counter < self.memory_size else self.memory_size,
                                   self.batch_size)
        transition_batch = self.memory[indices, :]
        bs = transition_batch[:, :self.n_sdim]
        ba = transition_batch[:, self.n_sdim:self.n_sdim + self.n_adim]
        br = transition_batch[:, -self.n_sdim - 1:-self.n_sdim]
        bs_ = transition_batch[:, -self.n_sdim:]
        
        self.sess.run(self.atrain_op, feed_dict={self.S: bs})
        self.sess.run(self.ctrain_op, feed_dict={self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
