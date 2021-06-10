import numpy as np

np.random.seed(1)
ROWS = 4
COLS = 12
S = (3, 0)
G = (3, 11)


class Cliff:

    def __init__(self):
        self.end = False
        self.pos = S
        self.board = np.zeros([4, 12])  # positions except cliff as 0
        self.board[3, 1:11] = -1        # cliff marked as -1

    def nxtPosition(self, action):
        if action == "up":
            nxtPos = (self.pos[0] - 1, self.pos[1])
        elif action == "down":
            nxtPos = (self.pos[0] + 1, self.pos[1])
        elif action == "left":
            nxtPos = (self.pos[0], self.pos[1] - 1)
        else:
            nxtPos = (self.pos[0], self.pos[1] + 1)
        # check legitimacy
        if nxtPos[0] >= 0 and nxtPos[0] <= 3:
            if nxtPos[1] >= 0 and nxtPos[1] <= 11:
                self.pos = nxtPos
        if self.pos == G:
            self.end = True
            print("Game ends reaching goal")
        if self.board[self.pos] == -1:
            self.end = True
            print("Game ends falling off cliff")
        return self.pos

    def giveReward(self):
        if self.pos == G or self.board[self.pos] == 0:
            return -1       # all positions except cliff: -1
        else:
            return -100     # cliff reward: -100

    def show(self):
        for i in range(0, ROWS):
            print('-------------------------------------------------')
            out = '| '
            for j in range(0, COLS):
                if self.board[i, j] == -1:
                    token = '*'
                if self.board[i, j] == 0:
                    token = '0'
                if (i, j) == self.pos:
                    token = 'S'
                if (i, j) == G:
                    token = 'G'
                out += token + ' | '
            print(out)
        print('-------------------------------------------------')


class Agent:
    def __init__(self, exp_rate=0.3, lr=0.1, sarsa=True):
        self.cliff = Cliff()
        self.actions = ["up", "left", "right", "down"]
        self.states = []  # record position and action of each episode
        self.pos = S
        self.exp_rate = exp_rate
        self.lr = lr
        self.sarsa = sarsa
        self.q_table = {}     # Q-table
        for i in range(ROWS):
            for j in range(COLS):
                self.q_table[(i, j)] = {}
                for a in self.actions:
                    self.q_table[(i, j)][a] = 0

    def chooseAction(self):
        mx_nxt_reward = -999
        action = ""
        # epsilon-greedy
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            for a in self.actions:
                current_position = self.pos
                nxt_reward = self.q_table[current_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action

    def reset(self):
        self.states = []
        self.cliff = Cliff()
        self.pos = S

    def play(self, rounds=1000):
        for episode in range(rounds):
            while 1:
                curr_state = self.pos
                cur_reward = self.cliff.giveReward()
                action = self.chooseAction()
                self.cliff.pos = self.cliff.nxtPosition(action)
                self.pos = self.cliff.pos
                self.states.append([curr_state, action, cur_reward])
                if self.cliff.end:
                    break
            # game end update estimates
            reward = self.cliff.giveReward()
            print("End game reward", reward)
            for a in self.actions:
                self.q_table[self.pos][a] = reward   # reward of all actions in end state is same
            if self.sarsa:
                for s in reversed(self.states):
                    pos, action, r = s[0], s[1], s[2]
                    current_value = self.q_table[pos][action]
                    reward = current_value + self.lr * (r + reward - current_value)
                    self.q_table[pos][action] = round(reward, 3)
            else:
                for s in reversed(self.states):
                    pos, action, r = s[0], s[1], s[2]
                    current_value = self.q_table[pos][action]
                    reward = current_value + self.lr * (r + reward - current_value)
                    self.q_table[pos][action] = round(reward, 3)
                    reward = np.max(list(self.q_table[pos].values()))  # update using the max value of S'
            self.reset()    # reset for each episode


def showRoute(states):
    board = np.zeros([4, 12])
    # add cliff marked as -1
    board[3, 1:11] = -1
    for i in range(0, ROWS):
        print('-------------------------------------------------')
        out = '| '
        for j in range(0, COLS):
            token = '0'
            if board[i, j] == -1:
                token = '*'
            if (i, j) in states:
                token = 'R'
            if (i, j) == G:
                token = 'G'
            out += token + ' | '
        print(out)
    print('-------------------------------------------------')


if __name__ == "__main__":
    # Sarsa
    print("sarsa training ... ")
    ag = Agent(exp_rate=0.15, sarsa=True)
    ag.play(rounds=1000)
    print("sarsa final route:")
    ag_op = Agent(exp_rate=0)
    ag_op.q_table = ag.q_table
    states = []
    while 1:
        curr_state = ag_op.pos
        action = ag_op.chooseAction()
        states.append(curr_state)
        print("current position {} |action {}".format(curr_state, action))
        # next position
        ag_op.cliff.pos = ag_op.cliff.nxtPosition(action)
        ag_op.pos = ag_op.cliff.pos
        if ag_op.cliff.end:
            break
    showRoute(states)

    # Q-learning
    print("q-learning training ... ")
    ag = Agent(exp_rate=0.1, sarsa=False)
    ag.play(rounds=1000)
    print("q-learning final route:")
    ag_op = Agent(exp_rate=0)
    ag_op.q_table = ag.q_table
    states = []
    while 1:
        curr_state = ag_op.pos
        action = ag_op.chooseAction()
        states.append(curr_state)
        print("current position {} |action {}".format(curr_state, action))
        # next position
        ag_op.cliff.pos = ag_op.cliff.nxtPosition(action)
        ag_op.pos = ag_op.cliff.pos
        if ag_op.cliff.end:
            break
    showRoute(states)
