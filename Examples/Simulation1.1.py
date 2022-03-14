
# This example mainly shows the learning process of DRL_Searcher, corresponding to Simulation1 in our paper.

# To be more specific. Here we take MuRES1 in the House environment as an example.

# In order to get more accurate results, we repeatedly generate 50 groups of Z_matrix, and perform 500 search simulations respectively.
import sys
sys.path.append("..")

from Core.DRL_Searcher_Trainer import DRL_Searcher_Trainer
from Core.DRL_Searcher_Executor import DRL_Searcher_Executor

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from copy import copy,deepcopy


def train():

    Searcher1_trainer = DRL_Searcher_Trainer(Environment_num = 1, target_motion_num = 1, MuRES_objective_num = 1)

    Searcher1_trainer.Z_training(purpose_num = 2, learning_step = 5000, Emax = 40000, training_num = 50)
def execute():
    Searcher1_executor = DRL_Searcher_Executor(Environment_num = 1, target_motion_num = 1, MuRES_objective_num = 1)

    Searcher1_executor.PTB = 1

    learning_step_set = [2000, 5000, 10000, 20000, 30000, 40000]
    for i in learning_step_set:
        Searcher1_executor.Execute_learning(learning_step_num = i, Z_nums = 50, capture_times = 500)


def plot_func():
    font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 22,
        }

    font2 = {'family' : 'Times New Roman',
        'weight' : 'light',
        'size'   : 18,
        }

    labels = ["2000", "5000", "10000", "20000", "30000", "40000"]
    data = []
    for i in labels:
        data_in = []
        for j in range(50):
            data_in.extend( (np.load("../Outputs/HOUSE_MR1/PathLen_" + str(j) +'_mid' + str(i) + '.npy').tolist()))
        data.append(copy(data_in))

    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] > 30:
                data[i][j] = 30

    plt.figure(figsize=(8, 6))

    plt.boxplot(data, labels = labels, meanline = True, showmeans=True, showfliers=False,patch_artist=True, medianprops = {'color':'blue','linestyle' : '--'},
                meanprops = {'color' : 'black','linestyle':'-'},boxprops = {'facecolor':'blue'}, positions = range(0, 6))
    plt.xlabel("Episodes", font1)
    plt.ylabel("Capture Time", font1)
    plt.yticks(size =16)
    plt.xticks(size =16)
    plt.show()

train()
execute()
plot_func()
