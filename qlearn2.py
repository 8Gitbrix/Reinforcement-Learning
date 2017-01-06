### Ashwin Jeyaseelan
### Q-Learning with E-Greedy Selection

import random
import gym
from gym import wrappers
env = gym.make('Taxi-v1')
env = gym.wrappers.Monitor(env, "gym_results", force=True)

q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        q[(s,a)] = 0.0

# update q table:
def update(state, action, reward, nextstate, alpha, gamma):
    # select action that yields greatest stored value for nextstate and store its value
    qa = max([q[(nextstate, a)] for a in range(env.action_space.n)])
    q[(state,action)] += alpha * (reward + gamma * qa - q[(state,action)])

# used e-greedy to select action to take
def epsilon_greedy(state, epsilon):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else: #takes list of actions,lambda function finds action with max value
        return max(list(range(env.action_space.n)), key = lambda x: q[(state,x)])


alpha = 0.4
gamma = 0.999
epsilon = 0.017

r = 0

for _ in range(8000):
    r = 0
    state = env.reset()
    while True:
        #env.render()
        action = epsilon_greedy(state, epsilon)
        newstate, reward, done, _ = env.step(action)
        update(state, action, reward, newstate, alpha, gamma)
        state = newstate

        r += reward

        if done:
            break

    print("total reward: ", r)

env.close()
gym.upload('gym_results', api_key="####")
