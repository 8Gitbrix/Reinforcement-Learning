### Ashwin Jeyaseelan
### Q-Learning with Boltzmann Exploration
from math import exp
import random
import numpy as np
import gym
from gym import wrappers
env = gym.make('FrozenLake8x8-v0')
#env = gym.wrappers.Monitor(env, "gym_results")

q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        q[(s,a)] = 0.0

# update q table:
def update(state, action, reward, nextstate, alpha, gamma):
    # select action that yields greatest stored value for nextstate and store its value
    qa = max([q[(nextstate, a)] for a in range(env.action_space.n)])
    q[(state,action)] += alpha * (reward + gamma * qa - q[(state,action)])

# used to select action to take
def policy(state, t):
    p = np.array([q[(state,x)]/t for x in range(env.action_space.n)])
    prob_actions = np.exp(p) / np.sum(np.exp(p))
    cumulative_probability = 0.0
    choice = random.uniform(0,1)
    for a,pr in enumerate(prob_actions):
        cumulative_probability += pr
        if cumulative_probability > choice:
            return a

alpha = 0.8
gamma = 0.999
temp = 3.0

for _ in range(30000):
    state = env.reset()
    while True:
        env.render()
        action = policy(state, temp)
        newstate, reward, done, _ = env.step(action)
        if reward == 0:
            reward = -0.01
        update(state, action, reward, newstate, alpha, gamma)
        state = newstate
        if done:
            break
    if temp > 1.0:
        temp -= 0.01

env.close()
#gym.upload('gym_results', api_key="####")
