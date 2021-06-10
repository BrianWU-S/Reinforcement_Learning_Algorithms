from DQN_maze_env import Maze
from DQN_RL_brain_prototype3 import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(300):
        observation = env.reset()
        while True:
            env.render()
            action = RL.choose_action(observation)
            observation_, reward, done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            if step > 200 and step % 5 == 0:
                RL.learn()
            observation = observation_
            if done:
                break
            step += 1
    env.destroy()
    print("Simulation finished. Congratulations,sir!")


if __name__ == '__main__':
    env = Maze()
    RL = DeepQNetwork(n_actions=env.n_actions, n_features=env.n_features, memory_size=2000, output_graph=True)
    env.after(5, run_maze())
    env.mainloop()
    RL.plot_cost()
