

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
    Searcher1_trainer = DRL_Searcher_Trainer(Environment_num = 1, target_motion_num = 1, MuRES_objective_num = 2)

    Searcher1_trainer.Z_training(purpose_num = 2, learning_step = 5000, Emax = 40000, training_num = 50)

def execute():
    Searcher1_executor = DRL_Searcher_Executor(Environment_num = 1, target_motion_num = 1, MuRES_objective_num = 2)

    learning_step_set = [2000, 5000, 10000, 15000, 20000, 30000]
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

    labels = ["2000", "5000", "10000", "15000", "20000", "30000"]

    data = []
    for i in labels:
        data_in = []
        for j in range(50):
            data_in.extend( (np.load("../Outputs/HOUSE_MR2/PathLen_" + str(j) +'_mid' + str(i) + '.npy').tolist()))
        data.append(copy(data_in))

    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] > 30:
                data[i][j] = 30
    y_list = []
    for i in range(len(data)):
        su = 0
        su_m = 0
        for j in range(len(data[i])):
            su += 1
            if data[i][j] <= 5:
                su_m += 1
        y_list.append(su_m/su)

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.boxplot(data, labels = labels, meanline = True, showmeans=False, showfliers=False,patch_artist=True, medianprops = {'color':'blue','linestyle' : '--'},
                boxprops = {'facecolor':'blue'}, positions = range(0, 6))
    ax1.set_ylabel('Capture Time',font1)
    ax1.set_xlabel("Episodes",font1)
    plt.yticks(size =16)
    plt.xticks(size =16)

    ax2 = ax1.twinx()

    ax2.plot(labels, y_list, marker = '^', color='green',label = 'Capture Probability')
    ax2.set_ylabel("Capture Probability",font1)
    plt.ylim((0, 1))
    plt.yticks(size =16)
    plt.xticks(size =16)
    plt.legend(prop = font2)
    plt.show()

train()
execute()
plot_func()
