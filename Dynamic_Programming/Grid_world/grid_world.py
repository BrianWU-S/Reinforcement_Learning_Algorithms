import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random


def actionRewardFunction(initialPosition, action):
    if initialPosition in terminationStates:
        return initialPosition, 0

    reward = rewardSize
    finalPosition = np.array(initialPosition) + np.array(action)
    if -1 in finalPosition or 6 in finalPosition:
        finalPosition = initialPosition

    return finalPosition, reward


def policy_evaluation(valueMap, actionMap, plot_flag=False):
    deltas = []  # deltas in current policy
    value_stable = False
    it = 0
    while not value_stable:
        copyValueMap = np.copy(valueMap)  # valueMap is the previous value
        deltaState = []
        for si, state in enumerate(states):
            weightedRewards = 0
            for ai, action in enumerate(actions):
                # original: have the same prob to choose the action (random choose), after update: have the prob
                if actionMap[si][ai] > 0:
                    finalPosition, reward = actionRewardFunction(state, action)
                    weightedRewards += actionMap[si][ai] * (
                            reward + (gamma * valueMap[finalPosition[0], finalPosition[1]]))
            deltaState.append(np.abs(copyValueMap[state[0], state[1]] - weightedRewards))
            copyValueMap[state[0], state[1]] = weightedRewards  # copyValueMap is the updated value
        deltas.append(deltaState)
        if np.all(copyValueMap == valueMap):
        # if np.all(np.abs(copyValueMap - valueMap) < 1e-1):  # original: np.all(copyValueMap == valueMap)
            value_stable = True
            print("Policy evaluation finished in Iteration {}".format(it + 1))
            print(valueMap)
            print("")
        valueMap = copyValueMap
        it += 1
    if plot_flag:
        plotting_results(plot_data=deltas)
    return valueMap


def plotting_results(plot_data):
    plt.figure(figsize=(20, 10))
    plt.plot(plot_data)
    plt.xlabel("Number of iterations")
    plt.ylabel("Delta values")
    plt.title("Delta value graph")
    plt.show()


def policy_iteration(valueMap, actionMap):
    policy_stable = False
    it = 0
    while not policy_stable:
        print("Policy Iteration {}".format(it + 1))
        valueMap = policy_evaluation(valueMap=valueMap,
                                     actionMap=actionMap)  # the sequence of policy evaluation and iteration is important
        policy_stable = True
        for i, state in enumerate(states):
            old_action = np.copy(actionMap[i])  # note: need to use np.copy() here, otherwise they are the same
            a_list = []
            for action in actions:
                finalPosition, reward = actionRewardFunction(state, action)
                estimated_reward = reward + (gamma * valueMap[finalPosition[0], finalPosition[1]])  # transition prob=1
                a_list.append(estimated_reward)
            actionMap[i] = 0
            actionMap[i][np.argmax(a_list)] = 1  # greedy update
            if policy_stable and not np.all(actionMap[i] == old_action):
                policy_stable = False  # when the policy is not stable (still changing), continue to update
        it += 1
    return valueMap, actionMap


if __name__ == "__main__":
    # settings
    gamma = 1
    rewardSize = -1
    gridSize = 6
    terminationStates = [[0, 1], [gridSize - 1, gridSize - 1]]
    actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
    init_valueMap = np.zeros((gridSize, gridSize))
    states = [[i, j] for i in range(gridSize) for j in range(gridSize)]  # a state is a position (x,y)
    init_actionMap = 0.25 * np.ones((np.shape(states)[0], np.shape(actions)[0]))  # init the actionMap to random action
    _ = policy_evaluation(valueMap=init_valueMap, actionMap=init_actionMap, plot_flag=True)  # plot the init policy
    result_valueMap, result_actionMap = policy_iteration(init_valueMap, init_actionMap)
    # _ = policy_evaluation(valueMap=result_valueMap, actionMap=result_actionMap, plot_flag=True)

    print("\n The result is: \n", result_valueMap, '\n', result_actionMap)
    print("Simulation finished. Congratulations,sir!")
