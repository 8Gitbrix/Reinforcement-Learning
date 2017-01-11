### Ashwin Jeyaseelan
### Sarsa (an on-policy TD control algorithm)
import gym

import random
env = gym.make('Taxi-v1')
from math import exp


Q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        Q[(s,a)] = 0.0

def policy(state, t):
    sm = sum([exp(Q[(state,a)]/t) for a in range(env.action_space.n)])
    random_choice = random.uniform(0,sm)
    cumulative_probability = 0.0

    for x in range(env.action_space.n):
        cumulative_probability += exp(Q[(state,x)]/t)
        if cumulative_probability > random_choice:
            return x

alpha = 0.45
gamma = 0.90
t = 2.0

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
            break

    print("total reward: ", r)

env.close()
