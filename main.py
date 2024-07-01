import thelastofus
import numpy as np
import random
from utils import save_q_table, load_q_table, save_enviroment

env = thelastofus.TheLastOfUsEnv(desc=None)

action_size = env.action_space.n
state_size = env.observation_space.n

total_episodes = 10000
learning_rate = 0.8
max_steps = 200
gamma = 0.95

# Parâmetro de temperatura
tau = 0.5

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001  # Adjusted decay rate for more exploration

rewards = []

actions = [0, 1, 2, 3]
inverse_actions = [2, 3, 0, 1]
qtable = np.zeros((state_size, action_size))
max_qtd_supply = 0

try:
    qtable = load_q_table()
    print("Q-table load success.")
except FileNotFoundError:
    print("No Q-table found, creating a new one.")

print(env.P)

def getProbArgMax(possible_actions):
    probabilities = np.exp(possible_actions / tau) / np.sum(np.exp(possible_actions / tau))
    return np.random.choice(actions, p=probabilities)

def getAction(state, old_action, random_action=False):
    action = -1
    if random_action:
        action = env.action_space.sample()
    else:
        action = getProbArgMax(qtable[state, :])

    while inverse_actions[action] == old_action:
        if random_action:
            action = env.action_space.sample()
        else:
            possibilities = qtable[state, :].copy()
            possibilities[action] = -100
            action = getProbArgMax(possibilities)
    return action

for episode in range(total_episodes):
    state = env.reset()[0]
    old_action = -1
    step = 0
    done = False
    total_rewards = 0
    qtd_supply = 0
    tired = step == max_steps

    while not tired and not done:
        exp_exp_tradeoff = random.uniform(0, 1)
        random_action = exp_exp_tradeoff > epsilon
        action = getAction(state, old_action, random_action=random_action)
        
        new_state, reward, done, info, _ = env.step(action)

        qtable[state, action] += learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        old_action = action
        if reward == 10:
            qtd_supply += 1
        elif reward == -20:
            old_action = -1

        if qtd_supply + 1 > max_qtd_supply and reward == 0:
            done = False

        total_rewards += reward
        state = new_state
        step += 1
        tired = step == max_steps

    print('episode:', episode, 'steps:', step, 'rewards:', total_rewards)
    max_qtd_supply = max(max_qtd_supply, qtd_supply)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    save_q_table(qtable)
    save_enviroment(max_qtd_supply)
    rewards.append(total_rewards)

print("Score over time: " + str(sum(rewards) / total_episodes))
print("QTABLE =", qtable)
