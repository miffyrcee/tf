import math

import gym
import matplotlib.pyplot as plt
import numpy as np


def loc(state):
    print(np.unravel_index(state, (5, 5, 5, 4)))
    return np.unravel_index(state, (5, 5, 5, 4))


def de_loc(loc):
    x, y, s, t = loc
    print(np.array([100, 20, 4, 1]) @ [x, y, s, t])


class q_learning(object):
    def __init__(self, env, epsilon=0.05, alpha=0.6, gamma=0.95):
        self.q = np.zeros((env.nS, env.nA))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = env

    def action_selector(self, state):
        if np.random.uniform() > self.epsilon:
            action = self.q[state].argmax()
        else:
            action = np.random.randint(self.env.nA)
        return int(action)

    def learn(self, state, action, next_state, reward, done):
        self.q[state][action] += (
            1 - self.alpha) * self.q[state][action] + self.alpha * (
                reward + self.gamma * self.q[next_state].max() * (1 - done))


def play_one(env, agent):
    state = env.reset()
    for _ in range(env.nS):
        if env.render():
            env.render()

        action = agent.action_selector(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, next_state, reward, done)
        state = next_state
        if done:
            break


class double_q_learning(q_learning):
    def __init__(self, env):
        super().__init__(env)
        self.qa = np.zeros((env.nS, env.nA))
        self.qb = np.zeros((env.nS, env.nA))

    def learn(self, state, action, next_state, reward, done):
        qa_a = self.qa[next_state].argmax()
        self.qa[state][action] += (
            1 - self.alpha) * self.qa[state][action] + self.alpha * (
                reward + self.gamma * self.qb[next_state][qa_a]) * (1 - done)
        if np.random.uniform() > 0.05:
            self.qa, self.qb = self.qb, self.qa
        self.q = self.qa


def main():
    env = gym.make('Taxi-v3')
    agent = double_q_learning(env)
    for _ in range(5):
        play_one(env, agent)
    print(agent.qa)
    # agent = q_learning(env)
    # for _ in range(100):
    #     play_one(env, agent)
    # print(agent.q)


if __name__ == "__main__":
    main()
