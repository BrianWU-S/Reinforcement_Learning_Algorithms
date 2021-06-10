from RL_brain_DDPG import DDPG
import numpy as np
import gym
import time
import matplotlib.pyplot as plt


def gym_run(env, ddpg, max_action, min_action, EPISODE=100, start_step=25000, step_per_epoch=5000,
            render_threshold=24500, use_explore=True, explore_decay=0.99998, reward_factor=10):
    render = False
    var = max_action
    threshold = render_threshold
    max_exp_step = start_step
    all_episode_reward = []
    t1 = time.time()
    for episode in range(EPISODE):
        observation = env.reset()
        reward_sum = 0
        for j in range(step_per_epoch):
            if render:
                env.render()
            action = ddpg.choose_action(observation)
            if use_explore:
                action = np.clip(np.random.normal(action, var), min_action, max_action)  # action exploration
            else:
                action = np.clip(action, min_action, max_action)
            observation_, reward, done, info = env.step(action)
            ddpg.store_transition(observation, action, reward/reward_factor, observation_)  # reward factor here
            if ddpg.memory_counter > max_exp_step:
                var *= explore_decay  # decay the action randomness
                ddpg.learn()
            reward_sum += reward
            observation = observation_
        print('Episode:', episode, ' Reward: %i' % int(reward_sum), 'Explore: %.2f' % var)
        if reward_sum > threshold:
            render = True
        all_episode_reward.append(reward_sum)
    print('Running time: ', time.time() - t1)
    print("Total training process average reward:", np.mean(all_episode_reward))
    print("Last 20 episode training reward", np.mean(all_episode_reward[-20:]))
    print(all_episode_reward)
    plt.plot(np.arange(len(all_episode_reward)) * step_per_epoch, all_episode_reward)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward")
    plt.title("Timestep reward graph (1 episode: %i steps)" % step_per_epoch)
    plt.show()


def print_info(env):
    action_shape = env.action_space.shape or env.action_space.n
    state_shape = env.observation_space.shape or env.observation_space.n
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    print("Actions shape:", action_shape, "\n Observations shape:", state_shape)
    print("\n Action range:", "\n Low:", env.action_space.low, "\n High:", env.action_space.high,
          "\n Observations range:", "\n Low:", env.observation_space.low, "\n High:", env.observation_space.high)


def run_DDPG(env_name='Pendulum-v0', seed=0, memory_size=100000, hidden_size=256, LR_A=0.0001, LR_C=0.001, GAMMA=0.99,
             TAU=0.005, batch_size=256, EPISODE=100, start_step=25000, step_per_epoch=2500, render_threshold=24500,
             use_explore=True,explore_decay=0.99998, reward_factor=10.):
    env = gym.make(env_name)
    env.seed(seed)
    env = env.unwrapped
    print_info(env)
    action_shape = env.action_space.shape or env.action_space.n
    state_shape = env.observation_space.shape or env.observation_space.n
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]
    ddpg = DDPG(a_dim=action_shape[0], s_dim=state_shape[0], a_bound=max_action,
                hidden_dim=hidden_size, LR_A=LR_A, LR_C=LR_C, GAMMA=GAMMA, memory_size=memory_size,
                batch_size=batch_size, TAU=TAU)
    gym_run(env, ddpg, max_action=max_action, min_action=min_action, EPISODE=EPISODE, start_step=start_step,
            step_per_epoch=step_per_epoch, render_threshold=render_threshold, use_explore=use_explore,
            explore_decay=explore_decay, reward_factor=reward_factor)
    print('Simulation finished. Congratulations,sir!')


if __name__ == '__main__':
    # Hyper parameters
    env_name = "Hopper-v1"
    seed = 0
    memory_size = 1000000  # more complex problem: 1000000
    hidden_size = 512
    LR_A = 0.0001
    LR_C = 0.001  # Generally, we set the critic learning rate larger than actor learning rate to get a more stable learning
    GAMMA = 0.99  # Up to the environment, in simpler env, we set GAMMA to smaller value
    TAU = 0.001  # soft replacement, default: 0.01
    EPISODE = 400
    start_step = 10000
    step_per_epoch = 1000
    batch_size = 64  # depend on your running environment, set to a smaller value if GPU memory is not big enough
    render_threshold = 2000
    use_explore = False
    explore_decay = 0.99998
    reward_factor = 0.5
    # training
    run_DDPG(env_name=env_name, seed=seed, memory_size=memory_size, hidden_size=hidden_size, LR_A=LR_A, LR_C=LR_C,
             GAMMA=GAMMA,
             TAU=TAU, batch_size=batch_size, EPISODE=EPISODE, start_step=start_step, step_per_epoch=step_per_epoch,
             render_threshold=render_threshold, use_explore=use_explore,explore_decay=explore_decay, reward_factor=reward_factor)
