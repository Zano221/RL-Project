import pickle

directory_path_training = './training/'

def save_q_table(q_table, filename='q_table.pkl'):
    with open(directory_path_training+filename, 'wb') as f:
        pickle.dump(q_table, f)

def load_q_table(filename='q_table.pkl'):
    with open(directory_path_training+filename, 'rb') as f:
        return pickle.load(f)

def save_enviroment(max_qtd_supply,filename="q_enviroment.pkl"):
    with open(directory_path_training+filename, 'wb') as f:
        pickle.dump(max_qtd_supply, f)


def load_enviroment(filename='q_enviroment.pkl'):
    with open(directory_path_training+filename, 'rb') as f:
        return pickle.load(f)
