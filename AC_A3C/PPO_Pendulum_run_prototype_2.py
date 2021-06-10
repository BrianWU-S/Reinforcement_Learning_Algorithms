from RL_brain_PPO_prototype_2 import PPO
import numpy as np
import gym
import matplotlib.pyplot as plt
import time

np.random.seed(1)


def gym_run():
    render = False
    threshold = -800
    all_episode_reward = []
    for episode in range(1000):
        observation = env.reset()
        episode_reward = 0
        buffer_s, buffer_a, buffer_r = [], [], []
        for t in range(200):
            if render:
                env.render()
            action = ppo.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            if t == 200 - 1:
                done = True
            episode_reward += reward
            buffer_s.append(observation)
            buffer_a.append(action)
            buffer_r.append((reward + 8) / 8)
            observation = observation_
            if (t + 1) % 32 == 0 or done:  # Note: here is t
                v_s_ = ppo.get_v(observation_)
                discounted_reward = []
                for r in buffer_r[::-1]:
                    v_s_ = v_s_ * 0.9 + r
                    discounted_reward.append(v_s_)
                discounted_reward.reverse()

                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_reward)[:, np.newaxis]
                ppo.update(bs, ba, br)
                buffer_s, buffer_a, buffer_r = [], [], []  # Note that here is buffer_...
        # if episode_reward > threshold:
        #     render = True
        if episode == 0:
            all_episode_reward.append(episode_reward)
        else:
            all_episode_reward.append(0.9 * all_episode_reward[-1] + 0.1 * episode_reward)
        print("Episode: ", episode, " |Episode_reward: %.2f " % episode_reward,
              ('|Lambda:%.4f' % ppo.method['lam']) if ppo.method['name'] == 'kl_penalty' else '')
    plt.plot(np.arange(len(all_episode_reward)), all_episode_reward)
    plt.xlabel("Episodes")
    plt.ylabel("Moving average episode reward")
    plt.show()


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    env.seed(2)
    env = env.unwrapped
    ppo = PPO(A_BOUND=[env.action_space.low, env.action_space.high], method_index=1)
    t1=time.time()
    gym_run()
    print('Running time:', time.time() - t1)
    print("Simulation finished. Congratulations,sir!")
