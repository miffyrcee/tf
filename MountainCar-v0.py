import gym
import numpy as np
import tensorflow as tf


class dqn(tf.keras.Model):
    def __init__(self, input_size=2, hidden_size=8, units=16, output_size=3):
        super().__init__()
        self.input_size = input_size
        self.units = units
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.q = list()
        self.build_model()

    def build_model(self):
        self.d1 = tf.keras.layers.Dense(self.units,
                                        input_shape=(self.input_size, ),
                                        activation='relu')
        self.d2 = tf.keras.layers.Dense(self.units, activation='relu')
        self.d3 = tf.keras.layers.Dense(self.output_size, activation='softmax')

    def call(self, x):
        x = self.d1(x)
        for _ in range(self.hidden_size):
            x = self.d2(x)
        x = self.d3(x)
        return x

    def loss(self):
        pass


def action_selector(states, episode=0.05):
    if np.random.rand() > episode:
        action = np.rand.randint(3)
    else:
        action = agent(states)


env = gym.make('MountainCar-v0')
agent = dqn()


# agent.summary()
def play_one():
    states = list()
    state = env.reset()
    for _ in range(1000):
        action = 0
        state, reward, done, _ = env.step(action)
        action = 2
        state, reward, done, _ = env.step(action)
        action = np.random.randint(3)
        state, reward, done, _ = env.step(action)
        if done:
            break
        states.append(state)
    states = np.array(states)
    print(agent(states))


def gradidents(w):
    w = np.array(w)
    x = tf.constant(0.)
    with tf.GradientTape() as tg:
        tg.watch(x)
        y = tf.ones((3, 3))
        tg.watch(y)
    dw_dx = tg.gradient(y, x)
    print(dw_dx)


play_one()
# t = np.array([[2, 3, 4], [2, 3, 1], [6, 2, 9]])
# t.var()
