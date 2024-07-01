import thelastofus
#from custonsmap import customMap5
import numpy as np
from utils import load_q_table, load_enviroment

env = thelastofus.TheLastOfUsEnv(desc=None, render_mode="human")


action_size = env.action_space.n
state_size = env.observation_space.n

max_steps = 99
total_episodes = 10

tau = 0.8

actions = [0,1, 2, 3] 
invers_actions = [2, 3, 0, 1]
            
rewards = []
qtable = []
qtd_max_supply = 0

try:
    qtable = load_q_table()
    qtd_max_supply = load_enviroment()
    print("Q-table load success.")
except FileNotFoundError:
    print("No Q-table found, creating a new one.")

def getProbArgMax(possible_actions):
    probabilities = np.exp(possible_actions / tau) / np.sum(np.exp(possible_actions / tau))
    # Seleção da ação baseada nas probabilidades

    return np.random.choice(actions, p=probabilities)

def getAction(state, old_action, random_action=False):
    action = -1
    if random_action:
        action = env.action_space.sample() 
    else:
        action = getProbArgMax(qtable[state,:])

    while(invers_actions[action] == old_action):
        if (random_action):
            action = env.action_space.sample()
        else:
            possibilities =  qtable[state,:].copy()
            possibilities[action] = -100
            action = getProbArgMax(possibilities)
    return action



for episode in range(total_episodes):
    state = env.reset()[0]
    step = 0
    old_action = -1
    done = False

    qtd_supply = 0
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        action = getAction(state, old_action)
        
        new_state, reward, done, info, _ = env.step(action)
        

        old_action = action
        if (reward==10):
            qtd_supply+=1
        elif reward==-20:
            old_action=-1

        if (reward==0 and qtd_max_supply>qtd_supply):
            done = False

        if done:
            env.render()
            
            print("Number of steps", step)
            break
        state = new_state
        
        
        
        old_action = action
env.close()