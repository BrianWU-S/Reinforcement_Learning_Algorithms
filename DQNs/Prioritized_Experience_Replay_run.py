import gym
from RL_brain_PrioritizedER import DQNPrioritizedReplay
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()
tf.set_random_seed(1)


def train(RL):
    total_steps = 0
    steps = []
    episodes = []
    reward_list = []
    for i_episode in range(20):
        observation = env.reset()
        episode_reward = 0
        while True:
            # env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)
            if done: reward = 10
            episode_reward += reward
            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()

            if done:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                reward_list.append(episode_reward)
                break

            observation = observation_
            total_steps += 1
    return np.vstack((episodes, steps)), reward_list


def plot(model1, model2=None):
    # compare based on first success
    plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN')
    plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DQN with prioritized replay')
    plt.legend(loc='best')
    plt.ylabel('total training steps')
    plt.xlabel('episode')
    plt.grid()
    plt.title("RL training step graph")
    plt.show()

    plt.figure(figsize=[20, 10])
    plt.plot(np.arange(len(model1.cost_his)), model1.cost_his, label='natural DQN')
    plt.plot(np.arange(len(model2.cost_his)), model2.cost_his, label='DQN with prioritized replay')
    plt.xlabel("Steps", fontdict={'size': 18})
    plt.ylabel("Cost", fontdict={'size': 18})
    plt.title("RL cost graph", fontdict={'size': 18})
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='best')
    plt.show()

    plt.figure(figsize=[20, 10])
    plt.plot(np.arange(len(reward_natural)), reward_natural, label='natural DQN', linewidth=2)
    plt.plot(np.arange(len(reward_prio)), reward_prio, label='DQN with prioritized replay', linewidth=2)
    plt.xlabel("Episode", fontdict={'size': 18})
    plt.ylabel("Reward per episode", fontdict={'size': 18})
    plt.title("RL reward graph", fontdict={'size': 18})
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='best')
    plt.show()


def set_soft_gpu(soft_gpu):
    if soft_gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


if __name__ == "__main__":
    set_soft_gpu(True)
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    env.seed(1)
    MEMORY_SIZE = 10000  # original 10000
    sess = tf.Session()
    with tf.variable_scope('natural_DQN'):
        RL_natural = DQNPrioritizedReplay(
            n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.00005, sess=sess, prioritized=False, output_graph=False
        )
    with tf.variable_scope('DQN_with_prioritized_replay'):
        RL_prio = DQNPrioritizedReplay(
            n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.00005, sess=sess, prioritized=True, output_graph=True,
        )
    sess.run(tf.global_variables_initializer())

    his_natural, reward_natural = train(RL_natural)
    his_prio, reward_prio = train(RL_prio)
    plot(model1=RL_natural, model2=RL_prio)

    print("Simulation finished. Congratulations,sir!")