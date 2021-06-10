import multiprocessing
import threading
import tensorflow.compat.v1 as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt
import time

tf.disable_v2_behavior()
tf.set_random_seed(1)
np.random.seed(1)

OUTPUT_GRAPH = True
MAX_GLOBAL_EP = 2000
GLOBAL_EP = 0
UPDATE_GLOBAL_ITER = 10
GLOBAL_RUNNING_R = []

env = gym.make('Pendulum-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]


class ACNet(object):
    def __init__(self, scope, global_net=None, entropy_beta=0.01):
        if scope == 'Global_Net':
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], name='S')
                self.a_params, self.c_params = self.build_net(scope)[-2:]
        else:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], name='S')
                self.a = tf.placeholder(tf.float32, [None, N_A], name='A')  # Note : float32
                self.v_ = tf.placeholder(tf.float32, [None, 1], name='V_')
                mu, sigma, self.v, self.a_params, self.c_params = self.build_net(scope)
                
                td_error = tf.subtract(self.v_, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td_error))
                
                with tf.name_scope('warp_out'):
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4
                normal_dist = tf.distributions.Normal(mu, sigma)
                with tf.name_scope('choose_action'):
                    self.action = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), A_BOUND[0],
                                                   A_BOUND[1])
                
                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a)
                    exp_v = log_prob * tf.stop_gradient(td_error)
                    entropy = normal_dist.entropy()
                    self.exp_v = exp_v + entropy_beta * entropy
                    self.a_loss = tf.reduce_mean(-self.exp_v)
                
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)
            
            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [tf.assign(la, ga) for la, ga in zip(self.a_params, global_net.a_params)]
                    self.pull_c_params_op = [tf.assign(lc, gc) for lc, gc in zip(self.c_params, global_net.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = optimizer_A.apply_gradients(zip(self.a_grads, global_net.a_params))
                    self.update_c_op = optimizer_C.apply_gradients(zip(self.c_grads, global_net.c_params))
    
    def build_net(self, scope):
        w_init = tf.random_normal_initializer(0, 0.1)
        with tf.variable_scope('Actor'):  # 局部性的用variable scope,全局性的用name_scope
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='Layer_actor')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='Mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='Sigma')
        with tf.variable_scope('Critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='Layer_critic')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')
        a_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope + '/Actor')
        c_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope + '/Critic')
        return mu, sigma, v, a_params, c_params
    
    def choose_action(self, s):
        action = sess.run(self.action, feed_dict={self.s: s[np.newaxis, :]})
        return action
    
    def update_global(self, feed_dict):
        sess.run([self.update_a_op, self.update_c_op], feed_dict=feed_dict)
    
    def pull_global(self):
        sess.run([self.pull_a_params_op, self.pull_c_params_op])


class Worker(object):
    def __init__(self, scope, globalNet, max_episode_step=200, decay_rate=0.9):
        self.env = gym.make('Pendulum-v0').unwrapped
        self.name = scope
        self.AC = ACNet(scope, globalNet)
        self.max_episode_step = max_episode_step
        self.gamma = decay_rate
    
    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, MAX_GLOBAL_EP
        total_step = 0
        buffer_s, buffer_a, buffer_r = [], [], []
        while not Coordinator.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            observation = self.env.reset()
            episode_reward = 0
            for ep_step in range(self.max_episode_step):
                if self.name == 'W_0':
                    self.env.render()
                action = self.AC.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                done = True if ep_step == self.max_episode_step - 1 else False
                episode_reward += reward
                buffer_s.append(observation)
                buffer_a.append(action)
                buffer_r.append((reward + 8) / 8)  # Note 3
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = sess.run(self.AC.v, feed_dict={self.AC.s: observation_[np.newaxis, :]})[0, 0]  # Note 1
                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + self.gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()
                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                        buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a: buffer_a,
                        self.AC.v_: buffer_v_target
                    }
                    self.AC.update_global(feed_dict)
                    self.AC.pull_global()
                    buffer_s, buffer_a, buffer_r = [], [], []
                observation = observation_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:
                        GLOBAL_RUNNING_R.append(episode_reward)
                    else:
                        GLOBAL_RUNNING_R.append(0.1 * episode_reward + 0.9 * GLOBAL_RUNNING_R[-1])
                    print(self.name, "Ep:", GLOBAL_EP, "| Ep_r: %i" % GLOBAL_RUNNING_R[-1])
                    GLOBAL_EP += 1
                    break  # note


def print_info():
    print(env.observation_space)
    print(env.action_space)
    print(env.observation_space.high)
    print(env.observation_space.low)


if __name__ == '__main__':
    print_info()
    sess = tf.Session()
    
    with tf.device('/cpu:0'):
        optimizer_A = tf.train.RMSPropOptimizer(learning_rate=0.001, name='RMSProp_A')
        optimizer_C = tf.train.RMSPropOptimizer(learning_rate=0.001, name='RMSProp_C')
        Global_Net = ACNet(scope='Global_Net')
        workers = []
        num_workers = multiprocessing.cpu_count()
        for i in range(4):
            i_name = 'W_%i' % i
            workers.append(Worker(scope=i_name, globalNet=Global_Net))
    
    Coordinator = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    if OUTPUT_GRAPH:
        if os.path.exists('./log'):
            shutil.rmtree('./log')
        tf.summary.FileWriter('./log', sess.graph)
    
    t1 = time.time()
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    Coordinator.join(worker_threads)
    print('Running time:', time.time() - t1)
    
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel("Steps")
    plt.ylabel("Running Reward")
    plt.show()
    GLOBAL_RUNNING_R = np.array(GLOBAL_RUNNING_R)
    np.save('plots/GLOBAL_RUNNING_R_%s' % str(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
