
fuck_this_teacher = True





if not fuck_this_teacher:

    import random
    import crashbaracuta
    import numpy as np
    from utils import *


    # SETAR AS VARIAVELS AQUI
    env = crashbaracuta.CrashBaracuta(desc=None)

    action_size = env.action_space.n
    state_size = env.observation_space.n

    total_episodes = 20000
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
    max_supply_ammount = 0


    # SETAR AS FUNÇÃO AQUI
    def load():
        global qtable
        try:
            qtable = load_q_table()
            print("Q-table load success.")
        except FileNotFoundError:
            print("No Q-table found, creating a new one.")

    def getProbArgMax(possible_actions):
        probabilities = np.exp(possible_actions / tau) / np.sum(np.exp(possible_actions / tau))
        return np.random.choice(actions, p=probabilities)
    

    def search_action(action, old_action):
        global inverse_actions



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
    


    # MAIN
    def main():

        #carregar o qtable
        load()

        for episode in range(total_episodes):
            global epsilon
            global supply_ammount
            global max_supply_ammount

            state = env.reset()[0]
            old_action = -1
            step = 0
            done = False
            total_rewards = 0
            supply_ammount = 0
            step_limit = step == max_steps

            while not step_limit and not done:
                exp_exp_tradeoff = random.uniform(0, 1)
                random_action = exp_exp_tradeoff > epsilon
                action = getAction(state, old_action, random_action=random_action)
                
                new_state, reward, done, info, _ = env.step(action)

                qtable[state, action] += learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

                old_action = action
                if reward == 10:
                    supply_ammount += 1
                elif reward == -20:
                    old_action = -1

                if supply_ammount + 1 > max_supply_ammount and reward == 0:
                    done = False

                total_rewards += reward
                state = new_state
                step += 1
                step_limit = step == max_steps

            if(episode % 200 == 0):
                print('Episodio:', episode, 'Paços:', step, 'Recompensas:', total_rewards)
            max_supply_ammount = max(max_supply_ammount, supply_ammount)
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
            save_q_table(qtable)
            save_enviroment(max_supply_ammount)
            rewards.append(total_rewards)

        print("Score over time: " + str(sum(rewards) / total_episodes))
        print("QTABLE =", qtable)

    if __name__ == "__main__":
        main()

             
import random
import numpy as np
from utils import load_q_table, save_q_table, load_enviroment, save_enviroment
from crashbaracuta import CrashBaracuta

#SETAR AS VARIAVELS AQUI

# Initialize the environment
env = CrashBaracuta(desc=None)

action_size = env.action_space.n
state_size = env.observation_space.n

# Training hyperparameters
total_episodes = 20000


learning_rate = 0.8
max_steps = 200
gamma = 0.95
tau = 0.5
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001

# Play hyperparameters
play_max_steps = 99
play_total_episodes = 10
play_tau = 0.8

# Q-table initialization
qtable = np.zeros((state_size, action_size))
max_supply_ammount = 0

# Action and inverse action definitions
actions = [0, 1, 2, 3]
inverse_actions = [2, 3, 0, 1]

#SETAR AS FUNÇÃO AQUI
def load():
    global qtable
    try:
        qtable = load_q_table()
        print("Q-table load success.")
    except FileNotFoundError:
        print("No Q-table found, creating a new one.")

def get_prob_argmax(possible_actions, tau):
    probabilities = np.exp(possible_actions / tau) / np.sum(np.exp(possible_actions / tau))
    return np.random.choice(actions, p=probabilities)

def search_action(state, action, old_action, random_action=False):
    global inverse_actions

    while inverse_actions[action] == old_action:

        action = env.action_space.sample()

        if not random_action:
            possibilities = qtable[state, :].copy()
            possibilities[action] = -100
            action = get_prob_argmax(possibilities, tau)

    return action

def get_action(state, old_action, random_action=False, tau=tau):
    if random_action:
        action = env.action_space.sample()
    else:
        action = get_prob_argmax(qtable[state, :], tau)

    return search_action(state, action, old_action, random_action)


#MAIN
def main(training=True, play=True):

    if not training  and not play:
        print("You need to set training=True or play=True")
        return

    # CARREGAR Q-table
    load()
    
    #SETAR AS VARIAVEL GLOBAL
    global epsilon
    global supply_ammount
    global max_supply_ammount
    
    #SE EU QUISER TREINAR BOLA PRA FRENTE
    if training:
        # Training loop
        rewards = []
        for episode in range(total_episodes):
            state = env.reset()[0]
            old_action = -1
            step = 0
            done = False
            total_rewards = 0
            supply_amount = 0

            while step < max_steps and not done:
                exp_exp_tradeoff = random.uniform(0, 1)
                random_action = exp_exp_tradeoff > epsilon
                action = get_action(state, old_action, random_action)

                new_state, reward, done, info, _ = env.step(action)

                qtable[state, action] += learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

                old_action = action
                if reward == 10:
                    supply_amount += 1
                elif reward == -20:
                    old_action = -1

                if supply_amount + 1 > max_supply_ammount and reward == 0:
                    done = False

                total_rewards += reward
                state = new_state
                step += 1

            if episode % 200 == 0:
                print(f'Episode: {episode}, Steps: {step}, Rewards: {total_rewards}')
            
            max_supply_ammount = max(max_supply_ammount, supply_amount)
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
            

            # SALVAR A CADA ITERAÇÃO
            save_q_table(qtable)
            save_enviroment(max_supply_ammount)
            rewards.append(total_rewards)

        print("Score over time: " + str(sum(rewards) / total_episodes))
        print("QTABLE =", qtable)

    #SE EU QUISER MOSTRAR O JOGUIN BOLA PRA FRENTE
    if play:

        # Play loop
        for episode in range(play_total_episodes):
            env.render_mode = 'human'
            state = env.reset()[0]
            old_action = -1
            done = False
            supply_amount = 0

            print("******")
            print("EPISODE ", episode)

            for step in range(play_max_steps):
                action = get_action(state, old_action, tau=play_tau)
                
                new_state, reward, done, info, _ = env.step(action)

                old_action = action
                if reward == 10:
                    supply_amount += 1
                elif reward == -20:
                    old_action = -1

                if reward == 0 and max_supply_ammount > supply_amount:
                    done = False

                if done:
                    #env.render()
                    print("Number of steps", step)
                    break
                
                state = new_state
                
        env.close()

    

if __name__ == "__main__":
    main(training=False, play=True)