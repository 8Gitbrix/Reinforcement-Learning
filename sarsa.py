### Ashwin Jeyaseelan
### Sarsa (an on-policy TD control algorithm)
import gym
from gym import wrappers
import random
import numpy as np
env = gym.make('Taxi-v1')
env = gym.wrappers.Monitor(env, "gym_results", force=True)

Q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        Q[(s,a)] = 0.0

def policy(state, t):
    p = np.array([Q[(state,x)]/t for x in range(env.action_space.n)])
    prob_actions = np.exp(p) / np.sum(np.exp(p))
    cumulative_probability = 0.0
    choice = random.uniform(0,1)
    for a,pr in enumerate(prob_actions):
        cumulative_probability += pr
        if cumulative_probability > choice:
            return a

alpha = 0.85
gamma = 0.90
t = 4.0

for _ in range(4000):
    r = 0 # r keeps track of accumulated score (used to measure performance at each episode!)
    state = env.reset()
    action = policy(state,t)
    while True:
        #env.render()
        state2, reward, done, _ = env.step(action)
        action2 = policy(state2, t)
        Q[(state,action)] += alpha * (reward + gamma * Q[(state2,action2)]-Q[(state,action)])

        action = action2
        state = state2
        r += reward

        if done:
            t = 1.00
            break

    print("total reward: ", r)

env.close()
