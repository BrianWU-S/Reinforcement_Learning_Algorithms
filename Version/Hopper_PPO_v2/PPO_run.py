from RL_brain_PPO import PPO
import numpy as np
import gym
import matplotlib.pyplot as plt
import time

np.random.seed(1)


def gym_run(env, ppo, max_action, min_action, EPISODE=100, done_step=200, step_per_epoch=1000,
            render_threshold=2500, use_explore=False, explore_decay=0.99998, reward_factor=10., memory_batch=32,
            smooth_factor=0.9):
    render = False
    var = max_action
    threshold = render_threshold
    all_episode_reward = []
    t1 = time.time()
    for episode in range(EPISODE):
        observation = env.reset()
        episode_reward = 0
        buffer_s, buffer_a, buffer_r = [], [], []
        for t in range(step_per_epoch):
            if render:
                env.render()
            action = ppo.choose_action(observation)
            if use_explore:
                action = np.clip(np.random.normal(action, var), min_action, max_action)  # action exploration
            else:
                action = np.clip(action, min_action, max_action)
            observation_, reward, done, info = env.step(action)
            if t == done_step - 1:
                done = True
            episode_reward += reward
            buffer_s.append(observation)
            buffer_a.append(action)
            buffer_r.append((reward + reward_factor) / reward_factor)
            observation = observation_
            if (t + 1) % memory_batch == 0 or done:  # Note: here is t
                var *= explore_decay
                v_s_ = ppo.get_v(observation_)
                discounted_reward = []
                for r in buffer_r[::-1]:
                    v_s_ = v_s_ * smooth_factor + r
                    discounted_reward.append(v_s_)
                discounted_reward.reverse()
                
                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_reward)[:, np.newaxis]
                ppo.update(bs, ba, br)
                buffer_s, buffer_a, buffer_r = [], [], []  # Note that here is buffer_...
        if episode_reward > threshold:
            render = True
        if episode == 0:
            all_episode_reward.append(episode_reward)
        else:
            all_episode_reward.append(smooth_factor * all_episode_reward[-1] + (1 - smooth_factor) * episode_reward)
        print("Episode: ", episode, " |Episode_reward: %.2f " % episode_reward, 'Explore: %.2f' % var,
              ('|Lambda:%.4f' % ppo.method['lam']) if ppo.method['name'] == 'kl_penalty' else '')
    print('Running time:', time.time() - t1)
    print("Total training process average reward:", np.mean(all_episode_reward))
    print("Last 20 episode training reward", np.mean(all_episode_reward[-20:]))
    print(all_episode_reward)
    plt.plot(np.arange(len(all_episode_reward)) * step_per_epoch, all_episode_reward)
    plt.xlabel("Timesteps")
    plt.ylabel(" Moving average episode reward")
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


def run_PPO(env_name='Pendulum-v0', seed=0, hidden_size=256, LR_A=0.0001, LR_C=0.001, EPISODE=100, done_step=200,
            step_per_epoch=1000, render_threshold=2500, use_explore=False, explore_decay=0.9, reward_factor=8.,
            memory_batch=32, action_update_steps=10, critic_update_steps=10, beta_low=1. / 1.5, beta_high=1.5, alpha=2,
            method_index=1, smooth_factor=0.9, lam=0.5, kl_target=0.01, epsilon=0.2):
    env = gym.make(env_name)
    env.seed(seed)
    env = env.unwrapped
    print_info(env)
    action_shape = env.action_space.shape or env.action_space.n
    state_shape = env.observation_space.shape or env.observation_space.n
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]
    ppo = PPO(A_BOUND=[min_action, max_action], s_dim=state_shape[0], a_dim=action_shape[0], hidden_units=hidden_size,
              LR_A=LR_A, LR_C=LR_C,
              action_update_steps=action_update_steps, critic_update_steps=critic_update_steps, beta_low=beta_low,
              beta_high=beta_high, alpha=alpha, method_index=method_index,
              output_graph=True, lam=lam, kl_target=kl_target, epsilon=epsilon)
    gym_run(env, ppo, max_action=max_action, min_action=min_action, EPISODE=EPISODE, done_step=done_step,
            step_per_epoch=step_per_epoch, render_threshold=render_threshold, use_explore=use_explore,
            explore_decay=explore_decay, reward_factor=reward_factor, memory_batch=memory_batch,
            smooth_factor=smooth_factor)
    print("Simulation finished. Congratulations,sir!")


if __name__ == '__main__':
    # Hyper parameters
    env_name = 'Hopper-v1'
    seed = 0
    hidden_size = 64
    LR_A = 0.0001
    LR_C = 0.0003  # Generally, we set the critic learning rate larger than actor learning rate to get a more stable learning
    EPISODE = 50
    done_step = 1000
    step_per_epoch = 1000
    render_threshold = 5000
    use_explore = False
    explore_decay = 0.99995
    smooth_factor = 0
    reward_factor = 8
    memory_batch = 128
    action_update_steps = 20
    critic_update_steps = 20
    beta_low = 1. / 3.
    beta_high = 3
    alpha = 2
    method_index = 1
    lam = 0.98
    kl_target = 0.01
    epsilon = 0.2
    run_PPO(env_name=env_name, seed=seed, hidden_size=hidden_size, LR_A=LR_A, LR_C=LR_C,
            EPISODE=EPISODE, done_step=done_step, step_per_epoch=step_per_epoch,
            render_threshold=render_threshold, use_explore=use_explore, explore_decay=explore_decay,
            reward_factor=reward_factor, memory_batch=memory_batch, action_update_steps=action_update_steps,
            critic_update_steps=critic_update_steps, beta_low=beta_low, beta_high=beta_high, alpha=alpha,
            method_index=method_index, smooth_factor=smooth_factor, lam=lam, kl_target=kl_target, epsilon=epsilon)
