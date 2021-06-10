import gym
from RL_brain_DoubleDQN import DoubleDQN        # 与DQN区别： 1. choose_action中多了记录Q-max  2. q_target计算有两种选择
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def train(RL):
    total_steps = 0
    observation = env.reset()
    while True:
        if total_steps - MEMORY_SIZE > 8000: env.render()
        action = RL.choose_action(observation)
        f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)  # convert to [-2 ~ 2] float actions
        observation_, reward, done, info = env.step(np.array([f_action]))
        reward /= 10
        RL.store_transition(observation, action, reward, observation_)
        if total_steps > MEMORY_SIZE:
            RL.learn()
        if total_steps - MEMORY_SIZE > 20000:
            break
        observation = observation_
        total_steps += 1
    return RL.q


if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    env.seed(1)
    MEMORY_SIZE = 3000
    ACTION_SPACE = 11

    sess = tf.Session()
    with tf.variable_scope('Natural_DQN'):
        natural_DQN = DoubleDQN(
            n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001, double_q=False, sess=sess
        )
    with tf.variable_scope('Double_DQN'):
        double_DQN = DoubleDQN(
            n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)
    sess.run(tf.global_variables_initializer())

    q_natural = train(natural_DQN)
    q_double = train(double_DQN)

    natural_DQN.plot_cost("natural_DQN")
    double_DQN.plot_cost("double_DQN")

    plt.plot(np.array(q_natural), c='r', label='natural')
    plt.plot(np.array(q_double), c='b', label='double')
    plt.legend(loc='best')
    plt.ylabel('Q eval')
    plt.xlabel('training steps')
    plt.grid()
    plt.show()

    print("Simulation finished. Congratulations,sir!")
