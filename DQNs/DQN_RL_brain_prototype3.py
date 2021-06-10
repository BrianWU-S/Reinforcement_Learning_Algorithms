import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

tf.disable_v2_behavior()
np.random.seed(1)
tf.set_random_seed(1)


class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            decay_rate=0.9,
            epsilon_max=0.9,
            memory_size=500,
            replace_iter_num=200,
            batch_size=32,
            epsilon_increment=None,
            output_graph=False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = learning_rate
        self.gamma = decay_rate
        self.epsilon_max = epsilon_max
        self.epsilon_increment = epsilon_increment
        self.epsilon = 0 if epsilon_increment is not None else epsilon_max
        self.memory_size = memory_size
        self.memory = np.zeros((memory_size, 2 * n_features + 2))
        self.batch_size = batch_size
        self.replace_iter_num = replace_iter_num
        self.learn_step_counter = 0
        self.build_net()
        self.sess = tf.Session()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net")
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="eval_net")
        with tf.variable_scope("hard_replacement"):
            self.replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        self.r = tf.placeholder(tf.float32, [None, ], name='r')
        self.a = tf.placeholder(tf.int32, [None, ], name='a')
        w_initializer, b_initializer = tf.random_normal_initializer(mean=0, stddev=0.3), tf.constant_initializer(
            value=0.1)

        with tf.variable_scope("eval_net"):
            e1 = tf.layers.dense(inputs=self.s, units=20, activation=tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(inputs=e1, units=self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='e2')
        with tf.variable_scope("target_net"):
            t1 = tf.layers.dense(inputs=self.s_, units=20, activation=tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(inputs=t1, units=self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        with tf.variable_scope("q_target"):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax')
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope("q_eval"):
            # self.a 为action的index, tf.stack([tf.range,self.a],axis=1)给出了每个动作在self.q_eval中的位置
            # 返回的self._q_eval_wrt_a 为每个action
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.alpha).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, "memory_counter"):
            self.memory_counter = 0
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = np.hstack((s, [a, r], s_))
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action_values = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_values)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_iter_num == 0:
            self.sess.run(self.replace_op)
            print("target params updated")

        if self.memory_counter < self.memory_size:
            batch_index = np.random.choice(self.memory_counter, self.batch_size)
        else:
            batch_index = np.random.choice(self.memory_size, self.batch_size)
        transition_batch = self.memory[batch_index, :]

        _, cost = self.sess.run(
            [self.train_op, self.loss],
            feed_dict={
                self.s: transition_batch[:, :self.n_features],
                self.a: transition_batch[:, self.n_features],
                self.r: transition_batch[:, self.n_features + 1],
                self.s_: transition_batch[:, -self.n_features:]
            }
        )

        self.cost_his.append(cost)
        self.learn_step_counter += 1
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.xlabel("Steps")
        plt.ylabel("Cost")
        plt.title("RL cost graph")
        plt.show()
