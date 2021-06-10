import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()
tf.set_random_seed(1)
np.random.seed(1)


class PPO(object):
    def __init__(self, A_BOUND, s_dim=3, a_dim=1, LR_A=0.0001, LR_C=0.0002, action_update_steps=10,
                 critic_update_steps=10, beta_low=1. / 1.5, beta_high=1.5, alpha=2.,
                 method_index=0, output_graph=True
                 ):
        self.action_bound = A_BOUND
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.action_update_steps = action_update_steps
        self.critic_update_steps = critic_update_steps
        self.beta_low = beta_low
        self.beta_high = beta_high
        self.alpha = alpha
        self.method = [
            dict(name='kl_penalty', kl_target=0.01, lam=0.5),
            dict(name='clipping', epsilon=0.2)
        ][method_index]
        self.sess = tf.Session()

        # Critic
        self.s = tf.placeholder(tf.float32, [None, s_dim], name='s')
        with tf.variable_scope('Critic'):
            self.v = self.build_critic_net()
            self.discounted_reward = tf.placeholder(tf.float32, [None, 1], name='discounted_reward')
            self.advantage = self.discounted_reward - self.v
            with tf.variable_scope('c_loss'):
                c_loss = tf.reduce_mean(tf.square(self.advantage))  # No need to define c_loss as class variable
            with tf.variable_scope('c_train_op'):
                self.c_train_op = tf.train.AdamOptimizer(LR_C).minimize(c_loss)

        # Actor
        pi, pi_params = self.build_action_net(scope='pi', trainable=True)
        old_pi, old_pi_params = self.build_action_net(scope='old_pi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_action_op = tf.squeeze(pi.sample(1), axis=0)  # Note that here squeeze axis=0
        with tf.variable_scope('update_pi_params'):
            self.update_op = [tf.assign(op, p) for op, p in zip(old_pi_params, pi_params)]

        self.a = tf.placeholder(tf.float32, [None, a_dim], name='a')
        self.adv = tf.placeholder(tf.float32, [None, 1], name='adv')
        with tf.variable_scope('a_loss'):
            with tf.variable_scope('surrogate'):  # Note that here we add a surrogate scope, to have better coding
                ratio = pi.prob(self.a) / (old_pi.prob(self.a) + 1e-5)
                surrogate = ratio * self.adv
            if self.method['name'] == 'kl_penalty':
                kl = tf.distributions.kl_divergence(old_pi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.lam = tf.placeholder(tf.float32, None, name='lam')
                a_loss = -tf.reduce_mean(surrogate - self.lam * kl)  # Note: need to add '-' here
            else:
                J_theta = tf.minimum(surrogate,
                                     tf.clip_by_value(ratio, 1. - self.method['epsilon'],
                                                      1. + self.method['epsilon']) * self.adv)
                a_loss = -tf.reduce_mean(J_theta)  # No need to define a_loss as class variable
        with tf.variable_scope('a_train_op'):
            self.a_train_op = tf.train.AdamOptimizer(LR_A).minimize(a_loss)

        if output_graph:
            tf.summary.FileWriter('log_2/', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def build_critic_net(self):
        l1 = tf.layers.dense(inputs=self.s, units=100, activation=tf.nn.relu,
                             kernel_initializer=tf.random_normal_initializer(0, 0.1),
                             bias_initializer=tf.constant_initializer(0.1), name='critic_l1')
        v = tf.layers.dense(inputs=l1, units=1, activation=None,
                            kernel_initializer=tf.random_normal_initializer(0, 0.1),
                            bias_initializer=tf.constant_initializer(0.1), name='v')
        return v

    def build_action_net(self, scope, trainable):
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(inputs=self.s, units=100, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0, 0.1),
                                 bias_initializer=tf.constant_initializer(0.1), name='actor_l1', trainable=trainable)
            mu = 2 * tf.layers.dense(inputs=l1, units=self.a_dim, activation=tf.nn.tanh,  # Note: here has :  2* ...
                                     kernel_initializer=tf.random_normal_initializer(0, 0.1),
                                     bias_initializer=tf.constant_initializer(0.1), name='mu', trainable=trainable)
            sigma = tf.layers.dense(inputs=l1, units=self.a_dim, activation=tf.nn.softplus,
                                    kernel_initializer=tf.random_normal_initializer(0, 0.1),
                                    bias_initializer=tf.constant_initializer(0.1), name='sigma', trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        a_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return norm_dist, a_params

    def choose_action(self, s):
        action = self.sess.run(self.sample_action_op, feed_dict={self.s: s[np.newaxis, :]})[0]  # Note: [0]
        return np.clip(action, a_min=self.action_bound[0], a_max=self.action_bound[1])

    def get_v(self, s):
        if s.ndim < 2:  # Note: here is < 2, we add the batch dimension
            s = s[np.newaxis, :]
        return self.sess.run(self.v, feed_dict={self.s: s})[0, 0]  # Note:[0,0]

    def update(self, s, a, r):
        self.sess.run(self.update_op)  # Note: no need to add feed_dict here
        adv = self.sess.run(self.advantage, feed_dict={self.discounted_reward: r, self.s: s})
        # adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        # update actor
        if self.method['name'] == 'kl_penalty':
            for _ in range(self.action_update_steps):
                _, kl_mean = self.sess.run([self.a_train_op, self.kl_mean],
                                           feed_dict={self.a: a, self.adv: adv, self.s: s,
                                                      self.lam: self.method['lam']})
                if kl_mean > 4 * self.method['kl_target']:
                    break  # Early stopping
            # Adaptive kl penalty  -->
            # kl_mean is small, it means that policy update is small, so we need to decrease the KL penalty item.
            # If kl_mean is large, it means that policy update is large, we need tp increase the KL penalty item
            if kl_mean < self.beta_low * self.method['kl_target']:
                self.method['lam'] = self.method['lam'] / self.alpha
            elif kl_mean > self.beta_high * self.method['kl_target']:
                self.method['lam'] = self.method['lam'] * self.alpha
            # KL clipping
            self.method['lam'] = np.clip(self.method['lam'], 1e-4, 10)
        else:
            [self.sess.run(self.a_train_op, feed_dict={self.a: a, self.adv: adv, self.s: s}) for _ in
             range(self.action_update_steps)]

        # update critic
        [self.sess.run(self.c_train_op, feed_dict={self.s: s, self.discounted_reward: r}) for _ in
         range(self.critic_update_steps)]
