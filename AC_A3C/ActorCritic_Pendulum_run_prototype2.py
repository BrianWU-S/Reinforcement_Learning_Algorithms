import tensorflow.compat.v1 as tf
import gym
from RL_brain_AC_Pendulum_prototype2 import Actor, Critic
import matplotlib.pyplot as plt
import time
import numpy as np


tf.disable_v2_behavior()
tf.set_random_seed(2)
OUTPUT_GRAPH = True
GLOBAL_RUNNING_R = []


def gym_run():
    global GLOBAL_RUNNING_R
    render = False
    threshold = -500
    max_ep_time = 200       # original:1000 --> change for comparison with A3C
    for episode in range(2000):
        observation = env.reset()
        t = 0
        ep_rs = 0
        while True:
            if render:
                env.render()
            action = actor.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            reward /= 10
            ep_rs += reward
            td_error = critic.learn(observation, reward, observation_)
            actor.learn(observation, action, td_error)
            observation = observation_
            t += 1
            if t > max_ep_time:
                if 'running_reward' not in globals():
                    running_reward = ep_rs
                else:
                    running_reward = running_reward * 0.9 + ep_rs * 0.1
                if running_reward > threshold:
                    render = True
                print('Episode:', episode, " reward", int(running_reward))
                GLOBAL_RUNNING_R.append(running_reward)
                break


def print_info():
    print(env.observation_space)
    print(env.action_space)
    print(env.observation_space.high)
    print(env.observation_space.low)


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    env.seed(1)
    env = env.unwrapped
    print_info()

    sess = tf.Session()
    actor = Actor(sess=sess, n_features=env.observation_space.shape[0],
                  action_bound=[-env.action_space.high, env.action_space.high], lr=0.001)
    critic = Critic(sess=sess, n_features=env.observation_space.shape[0], lr=0.01, decay_rate=0.9)

    sess.run(tf.global_variables_initializer())
    if OUTPUT_GRAPH:
        tf.summary.FileWriter('logs/', sess.graph)
    
    t1=time.time()
    gym_run()
    print('Running time:', time.time() - t1)
    
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel("Steps")
    plt.ylabel("Running Reward")
    plt.show()