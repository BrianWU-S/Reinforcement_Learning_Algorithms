import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random


def generateEpisode():
    initState = random.choice(states[1:-1])
    episode = []
    while True:
        if list(initState) in terminationStates:
            return episode  # when reach the terminal state, just return the whole episode
        action = random.choice(actions)
        finalState = np.array(initState) + np.array(action)
        if -1 in list(finalState) or gridSize in list(finalState):
            finalState = initState
        episode.append([list(initState), action, rewardSize, list(finalState)])  # one stage: (s_t,a,r,s_t+1)
        initState = finalState


def MC_method_first_visit(V, deltas, plot_flag=False):
    for it in tqdm(range(numIterations)):
        episode = generateEpisode()
        G = 0
        if it in [1, 10, 100, 1000]:
            print("MC episode:", episode)
        for i, step in enumerate(episode[::-1]):
            # reverse the episode and update each stage estimation
            G = gamma * G + step[2]
            if step[0] not in [x[0] for x in episode[::-1][len(episode) - i:]]:
                # only choose the state that we first visited
                idx = (step[0][0], step[0][1])  # index of position (x,y)
                returns[idx].append(G)
                newValue = np.average(returns[idx])
                deltas[idx[0], idx[1]].append(np.abs(V[idx[0], idx[1]] - newValue))
                V[idx[0], idx[1]] = newValue
    if plot_flag:
        plotting_results(plot_data=deltas)
    return V


def MC_method_every_visit(V, deltas, plot_flag=False):
    for it in tqdm(range(numIterations)):
        episode = generateEpisode()
        G = 0
        if it in [1, 10, 100, 1000, numIterations - 1]:
            print("MC episode:", episode)
        for i, step in enumerate(episode[::-1]):
            # Every time-step t that state s is visited in an episode, we update the value
            G = gamma * G + step[2]
            idx = (step[0][0], step[0][1])  # state index
            returns[idx].append(G)
            newValue = np.average(returns[idx])
            deltas[idx[0], idx[1]].append(np.abs(V[idx[0], idx[1]] - newValue))
            V[idx[0], idx[1]] = newValue
    if plot_flag:
        plotting_results(plot_data=deltas)
    return V


def plotting_results(plot_data):
    plt.figure(figsize=(20, 10))
    all_series = [list(x)[:50] for x in plot_data.values()]
    for series in all_series:
        plt.plot(series)
    plt.xlabel("Number of iterations")
    plt.ylabel("Delta values")
    plt.title("Delta value graph")
    plt.show()


if __name__ == '__main__':
    # settings
    gamma = 0.6
    rewardSize = -1
    gridSize = 6
    numIterations = 10000  # use a large number to simulate infinite sampling number (episode number)
    terminationStates = [[0, 1], [gridSize - 1, gridSize - 1]]
    actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
    init_valueMap = np.zeros((gridSize, gridSize))
    states = [[i, j] for i in range(gridSize) for j in range(gridSize)]
    returns = {(i, j): list() for i in range(gridSize) for j in range(gridSize)}
    deltas = {(i, j): list() for i in range(gridSize) for j in range(gridSize)}
    print("First-Visit MC")
    # result_valueMap_fv = MC_method_first_visit(V=init_valueMap, deltas=deltas, plot_flag=True)
    print("Every-Visit MC")
    result_valueMap_ev = MC_method_every_visit(V=init_valueMap, deltas=deltas, plot_flag=True)
    # print("\n The first-visit result is: \n", result_valueMap_fv, '\n')
    print("\n The every-visit result is: \n", result_valueMap_ev, '\n')
    print("Simulation finished. Congratulations,sir!")
