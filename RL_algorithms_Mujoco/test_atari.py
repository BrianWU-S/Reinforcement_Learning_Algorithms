import gym


legal_env = [['VideoPinball-ramNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4', 'BoxingNoFrameskip-v4'],
             ['Hopper-v1', 'Humanoid-v1', 'HalfCheetah-v1', 'Ant-v1']]
legal_method = [['DQN'], ['A3C', 'DDPG', 'DDPG_noise']]


if __name__ == '__main__':
    env = gym.make("Ant-v1")
    observation = env.reset()
    for i in range(10000):
        print(i)
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
    env.close()