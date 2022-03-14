

import sys
sys.path.append("..")

from Inputs.Environment import G_house, G_office, G_museum
from Inputs.Target_Motion import target_motion_house1, target_motion_house2, target_motion_office1, target_motion_office2, target_motion_museum1, target_motion_museum2


import networkx as nx
import numpy as np
import random
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from math import ceil, floor

class DRL_Searcher_Executor:
    def __init__(self, Environment_num, target_motion_num, MuRES_objective_num):
        self.MuRES_obj = MuRES_objective_num
        self.Environment_num = Environment_num
        self.PTB = 0.8
        if Environment_num == 1:
            self.Environment_name = "HOUSE"
            self.G = G_house
            self.T = 5
            self.M = 51
            self.V_min = 0
            self.V_max = 50
            self.R_max = 10
            if target_motion_num == 1:
                self.target_motion = target_motion_house1
            else:
                self.target_motion = target_motion_house2

        elif Environment_num == 2:
            self.Environment_name = "OFFICE"
            self.G = G_office
            self.T = 10
            self.M = 101
            self.V_min = 0
            self.V_max = 100
            self.R_max = 20
            if target_motion_num == 1:
                self.target_motion = target_motion_office1
            else:
                self.target_motion = target_motion_office2
        else :
            self.Environment_name = "MUSEUM"
            self.G = G_museum
            self.T = 20
            self.M = 101
            self.V_min = 0
            self.V_max = 100
            self.R_max = 30
            if target_motion_num == 1:
                self.target_motion = target_motion_museum1
            else:
                self.target_motion = target_motion_museum2


    def Execute(self, Z_nums, capture_times, robot_num, beta):

        address = "../Mid_Outputs/" + self.Environment_name + "_MR" + str(self.MuRES_obj) + "/Z_"
        # if purpose_num == 1:
        for Z_num in range (Z_nums):
            Z_matrix = np.load(address + str(Z_num) + '.npy' )
            self.__Execute_process(Z_matrix, capture_times, robot_num, beta, Z_num)

            print("The Path has been stored")
        # else:
        #     for Z_num in range (Z_nums):
        #         Z_matrix = np.load(address + str(Z_num) +'_mid' + str(learning_step_num) +  '.npy' )
        #         self.__Execute_midstep(Z_matrix, capture_times, robot_num, beta, Z_num, learning_step_num)
        #     print("The Path has been stored")


    def Execute_learning(self, learning_step_num, Z_nums, capture_times):

        address = "../Mid_Outputs/" + self.Environment_name + "_MR" + str(self.MuRES_obj) + "/Z_"
        for Z_num in range (Z_nums):
            Z_matrix = np.load(address + str(Z_num) +'_mid' + str(learning_step_num) +  '.npy' )
            self.__Execute_learning_process(Z_matrix, learning_step_num, capture_times, Z_num)
        print("The Path has been stored")



    def __decision_making(self, T, robot_cur_position, target_cur_position, Z_matrix, recency_matrix, beta):

        G, MuRES_objective, M, V_min, V_max, R_max = self.G, self.MuRES_obj,self.M, self.V_min, self.V_max, self.R_max

        node_num1 = G.number_of_nodes()

        cur_state = robot_cur_position * (node_num1 + 1) + target_cur_position
        state_list = []
        for i in range(1,node_num1+1):
            if i != robot_cur_position:
                state_list.append(i + robot_cur_position * (node_num1 + 1))
        robot_act_list = list(G.adj[robot_cur_position])
        robot_act_list.append(robot_cur_position)
        random.shuffle(robot_act_list)
        best_act = 0
        min_act = 10000000000
        max_act = 0
        #     min t ：
        if MuRES_objective == 1:
            for j in robot_act_list:
                su = 0
                step = (V_max-V_min)/(M-1)
                for k in range(M):
                    su += Z_matrix[cur_state][j][k] * (V_min + k*step) * self.PTB
                    for state_sample in state_list:
                        su += Z_matrix[state_sample][j][k] * (V_min + k*step) * ((1-self.PTB)/(node_num1-1))
                su = beta * su + (1- beta) * recency_matrix[j]
                if su <= min_act:
                    best_act = j
                    min_act = su
            # TO DO :CDF
            return best_act
        #     max PD：
        else:
            for j in robot_act_list:
                su = 0
                step = (V_max-V_min)/(M-1)
                for k in range(M):
                    su += Z_matrix[cur_state][j][k] * self.PTB
                    for state_sample in state_list:
                        su += Z_matrix[state_sample][j][k] * ((1-self.PTB)/(node_num1-1))
                    if V_min + (k+1) * step > T:
                        break
                su = beta * su + (1- beta) * recency_matrix[j]/R_max
                if su >= max_act:
                    best_act = j
                    max_act = su
            return best_act

    def __decision_making_learning(self, T, robot_cur_position, target_cur_position, Z_matrix):

        G, MuRES_objective, M, V_min, V_max, R_max = self.G, self.MuRES_obj,self.M, self.V_min, self.V_max, self.R_max

        node_num1 = G.number_of_nodes()

        cur_state = robot_cur_position * (node_num1 + 1) + target_cur_position
        state_list = []
        for i in range(1,node_num1+1):
            if i != robot_cur_position:
                state_list.append(i + robot_cur_position * (node_num1 + 1))
        robot_act_list = list(G.adj[robot_cur_position])
        robot_act_list.append(robot_cur_position)
        random.shuffle(robot_act_list)
        best_act = 0
        min_act = 10000000000
        max_act = 0
        #     min t ：
        if MuRES_objective == 1:
            for j in robot_act_list:
                su = 0
                step = (V_max-V_min)/(M-1)
                for k in range(M):
                    su += Z_matrix[cur_state][j][k] * (V_min + k*step) * self.PTB
                    for state_sample in state_list:
                        su += Z_matrix[state_sample][j][k] * (V_min + k*step) * ((1-self.PTB)/(node_num1-1))
                # su = beta * su + (1- beta) * recency_matrix[j]
                if su <= min_act:
                    best_act = j
                    min_act = su
            # TO DO :CDF
            return best_act
        #     max PD：
        else:
            for j in robot_act_list:
                su = 0
                step = (V_max-V_min)/(M-1)
                for k in range(M):
                    su += Z_matrix[cur_state][j][k] * self.PTB
                    for state_sample in state_list:
                        su += Z_matrix[state_sample][j][k] * ((1-self.PTB)/(node_num1-1))
                    if V_min + (k+1) * step > T:
                        break
                # su = beta * su + (1- beta) * recency_matrix[j]/R_max
                if su >= max_act:
                    best_act = j
                    max_act = su
            return best_act

    def __position_initialize(self):
        Environment_num = self.Environment_num
        robot_start_position1 = 0
        target_start_position1 = 0
        possible_postion_list1 = []
        if Environment_num == 1:
            robot_start_position1 = 3
            possible_position_list1 = [1+x for x in range(9)]
            possible_position_list1.pop(robot_start_position1-1)
            target_start_position1 = random.choice(possible_position_list1)
        elif Environment_num == 2:
            robot_start_position1 = 22
            possible_position_list1 = [1+x for x in range(60)]
            possible_position_list1.pop(robot_start_position1-1)
            target_start_position1 = random.choice(possible_position_list1)
        else:
            robot_start_position1 = 1
            possible_position_list1 = [1+x for x in range(70)]
            possible_position_list1.pop(robot_start_position1-1)
            target_start_position1 = random.choice(possible_position_list1)
        return robot_start_position1, target_start_position1

    def __target_moving(self, target_position):
        target_motion = self.target_motion
        rand1 = random.random()
        target_list = target_motion[target_position-1]
        sum1 = 0
        target_nex = 0
        for i in range(len(target_list)):
            sum1 += target_list[i]
            if(sum1 > rand1):
                target_nex = i+1
                break
        return target_nex

    def __recency_update(self, recency_matrix1, robot_cur_position, robot_cur_action):
        R_max, MuRES_objective = self.R_max, self.MuRES_obj
        if MuRES_objective == 1:
            for i in range(len(recency_matrix1)):
                if recency_matrix1[i] > 0:
                    recency_matrix1[i] -= 1

            recency_matrix1[robot_cur_action] = R_max
            return recency_matrix1
        else:
            for i in range(len(recency_matrix1)):
                if recency_matrix1[i] < R_max:
                    recency_matrix1[i] += 1

            recency_matrix1[robot_cur_action] = 0
            return recency_matrix1

    def __Execute_process(self, Z_matrix, Cmax, robot_num, beta, Path_num):

        Environment_num, G, target_motion, MuRES_objective = self.Environment_num, self.G, self.target_motion, self.MuRES_obj
        T, M, V_min, V_max, R_max = self.T, self.M, self.V_min, self.V_max, self.R_max
        address = "../Outputs/" + self.Environment_name + "_MR" + str(self.MuRES_obj) + "/Path_" + str(Path_num) +'_robotNum' + str(robot_num) + '.npy'
        address1 = "../Outputs/" + self.Environment_name + "_MR" + str(self.MuRES_obj) + "/PathLen_" + str(Path_num) +'_robotNum' + str(robot_num) + '.npy'
        node_num = G.number_of_nodes()
        path_len = []
        time_set = []
        counter = 0
        path_set = []
        while counter < Cmax:
            # 1. recency_reward array initialize  2. robot_current_position_set initialize, 3.path.append(robot_current_position_set)
            print("capture times: " +str(counter))
            time_execute = 0
            path = []
            robot_current_position, target_current_position = self.__position_initialize()
            robot_current_position_set = [robot_current_position for x in range(robot_num)]
            recency_matrix = [R_max/2 for x in range(node_num + 1)]
            path.append(robot_current_position_set)
            while target_current_position not in robot_current_position_set:
                robot_next_position_set = []
                start_time = time.clock()
                for robot_current_position in robot_current_position_set:
                    current_state = robot_current_position * (node_num + 1) + target_current_position

                    robot_current_action = self.__decision_making(T-len(path)+1, robot_current_position, target_current_position, Z_matrix,recency_matrix, beta)

                    robot_next_position = robot_current_action

                    recency_matrix = self.__recency_update(recency_matrix, robot_current_position, robot_current_action)

                    robot_next_position_set.append(robot_next_position)

                end_time = time.clock()
                time_execute += (end_time-start_time)

                robot_current_position_set = copy(robot_next_position_set)
                target_next_position = self.__target_moving(target_current_position)
                target_current_position = target_next_position
                path.append(robot_current_position_set)

            time_set.append(time_execute)
            counter += 1
            path_len.append(len(path))
            path_set.append(deepcopy(path))
        # if Environment_num == 2:
        #     if MuRES_objective == 1:
        #         np.save('/Users/pqh/Desktop/DRL_Searcher/Multi_Office_data/office_path_PTB_robotNum_' + str(robot_num) + '.npy', np.array(copy(path_len)))
        #     else:
        #         np.save('/Users/pqh/Desktop/DRL_Searcher/Multi_Office_data/office_path_PTB_MR2_robotNum_' + str(robot_num) + '.npy', np.array(copy(path_len)))
        # else:
        #     if MuRES_objective == 1:
        #         np.save('/Users/pqh/Desktop/DRL_Searcher/Multi_Museum_data/museum_path_PTB_robotNum_' + str(robot_num) + '.npy', np.array(copy(path_len)))
        #     else:
        #         np.save('/Users/pqh/Desktop/DRL_Searcher/Multi_Museum_data/museum_path_PTB_MR2_robotNum_' + str(robot_num) + '.npy', np.array(copy(path_len)))
        np.save(address, np.array(deepcopy(path_set)))
        np.save(address, np.array(deepcopy(path_len)))
        # print(np.mean(path_len))
        # print(np.std(path_len))
        # print(np.mean(time_set))

    def __Execute_learning_process(self, Z_matrix, learning_step_num, Cmax, Path_num):
        Environment_num, G, target_motion, MuRES_objective = self.Environment_num, self.G, self.target_motion, self.MuRES_obj
        T, M, V_min, V_max = self.T, self.M, self.V_min, self.V_max
        address = "../Outputs/" + self.Environment_name + "_MR" + str(self.MuRES_obj) + "/Path_" + str(Path_num) +'_mid' + str(learning_step_num) + '.npy'
        address1 = "../Outputs/" + self.Environment_name + "_MR" + str(self.MuRES_obj) + "/PathLen_" + str(Path_num) +'_mid' + str(learning_step_num) + '.npy'
        node_num = G.number_of_nodes()
        path_set = []
        path_len = []
        counter = 0
        while counter < Cmax:
            path = []
            robot_current_position, target_current_position = self.__position_initialize()
            path.append(robot_current_position)
            while robot_current_position != target_current_position:
                current_state = robot_current_position * (node_num + 1) + target_current_position

                robot_current_action = self.__decision_making_learning(T - len(path) + 1, robot_current_position, target_current_position, Z_matrix)

                robot_next_position = robot_current_action
                target_next_position = self.__target_moving(target_current_position)

                robot_current_position = robot_current_action
                target_current_position = target_next_position
                path.append(robot_current_position)
            counter += 1
            path_len.append(len(path))
            path_set.append(deepcopy(path))
        np.save(address, np.array(deepcopy(path_set)))
        np.save(address1, np.array(deepcopy(path_len)))
