import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()
tf.set_random_seed(2)
np.random.seed(2)


class Actor(object):
    def __init__(
            self,
            sess,
            n_features,
            action_bound,
            lr=0.0001
    ):
        self.sess = sess
        self.n_features = n_features
        self.action_bound = action_bound
        self.lr = lr
        self.build_net()

    def build_net(self):
        self.s = tf.placeholder(tf.float32, [1, self.n_features], name='s')
        self.a = tf.placeholder(tf.float32, None, name='a')     # note that here a is float32, continuous value
        self.td_error = tf.placeholder(tf.float32, None, 'td_error')
        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(inputs=self.s, units=30, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0, 0.1),
                                 bias_initializer=tf.constant_initializer(0.1), name='l1')
            mu = tf.layers.dense(inputs=l1, units=1, activation=tf.nn.tanh,
                                 kernel_initializer=tf.random_normal_initializer(0, 0.1),
                                 bias_initializer=tf.constant_initializer(0.1), name='mu')
            sigma = tf.layers.dense(inputs=l1, units=1, activation=tf.nn.softplus,
                                    kernel_initializer=tf.random_normal_initializer(0, 0.1),
                                    bias_initializer=tf.constant_initializer(0.1), name='sigma')
        self.mu, self.sigma = tf.squeeze(mu * 2), tf.squeeze(sigma + 0.1)   # Note point 1: 2--> action_bound.high
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)
        self.action = tf.clip_by_value(self.normal_dist.sample(1), self.action_bound[0],
                                       self.action_bound[1])
        global_step = tf.Variable(initial_value=0, trainable=False)
        with tf.name_scope('exp_v'):
            log_prob = self.normal_dist.log_prob(self.a)
            self.exp_v = log_prob * self.td_error
            self.exp_v += 0.01 * self.normal_dist.entropy()     # Note point 2, entropy计算方法见 A3C
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v, global_step)  # Note point 3

    def choose_action(self, observation):
        actions = self.sess.run(self.action, {self.s: observation[np.newaxis, :]})  # deterministic choose one value
        return actions

    def learn(self, s, a, td_error):
        _, exp_v = self.sess.run([self.train_op, self.exp_v],
                                 {self.s: s[np.newaxis, :], self.a: a, self.td_error: td_error})
        return exp_v


class Critic(object):
    def __init__(
            self,
            sess,
            n_features,
            lr=0.01,
            decay_rate=0.9
    ):
        self.sess = sess
        self.n_features = n_features
        self.lr = lr
        self.gamma = decay_rate
        self.build_net()

    def build_net(self):
        self.s = tf.placeholder(tf.float32, [1, self.n_features], name='s')
        self.r = tf.placeholder(tf.float32, None, name='r')
        self.v_ = tf.placeholder(tf.float32, [1, 1], name='v_')
        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(inputs=self.s, units=30, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0, 0.1),
                                 bias_initializer=tf.constant_initializer(0.1), name='l1')
            self.v = tf.layers.dense(inputs=l1, units=1, activation=None,
                                     kernel_initializer=tf.random_normal_initializer(0, 0.1),
                                     bias_initializer=tf.constant_initializer(0.1), name='v')

        with tf.variable_scope('td_error'):
            self.td_error = tf.reduce_mean(self.r + self.gamma * self.v_ - self.v)
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def learn(self, s, r, s_):
        v_ = self.sess.run(self.v, {self.s: s_[np.newaxis, :]})
        td_error, _ = self.sess.run([self.td_error, self.train_op], {self.s: s[np.newaxis, :], self.r: r, self.v_: v_})
        return td_error
