
import sys
sys.path.append("..")

from Inputs.Environment import G_house, G_office, G_museum
from Inputs.Target_Motion import target_motion_house1, target_motion_house2, target_motion_office1, target_motion_office2, target_motion_museum1, target_motion_museum2

import networkx as nx
import numpy as np
import random

from copy import copy, deepcopy
from math import ceil, floor

class DRL_Searcher_Trainer:

    def __init__(self, Environment_num, target_motion_num, MuRES_objective_num):
        self.alpha = 0.01
        self.epsilon = 0.05
        self.MuRES_obj = MuRES_objective_num
        self.Environment_num = Environment_num
        if Environment_num == 1:
            self.Environment_name = "HOUSE"
            self.G = G_house
            self.T = 5
            self.M = 51
            self.V_min = 0
            self.V_max = 50
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
            if target_motion_num == 1:
                self.target_motion = target_motion_museum1
            else:
                self.target_motion = target_motion_museum2

    def Z_training(self, purpose_num = 1, learning_step = 5000, Emax = 40000, training_num = 1):
        if purpose_num == 1:
            for Z_num in range (training_num):
                self.__Z_training_process(Emax, Z_num)
                print("The " + str(Z_num+1) + " training finished")
            print("The final Z_matrix has been stored")
        else:
            for Z_num in range (training_num):
                self.__Z_learning_process(Emax, Z_num, learning_step)
                print("The " + str(Z_num+1) + " training finished")
            print("The learning process of Z_matrix has been stored")




    def __Z_warm_start(self):

        G, M, V_min, V_max = self.G, self.M, self.V_min, self.V_max

        node_num1 = G.number_of_nodes()
        state_num = (node_num1 + 1) ** 2
        action_num = node_num1 + 1
        Z_matrix1 = [[[0 for x in range(M)]for x in range(action_num)] for x in range(state_num)]
        for robot_p in range(1, node_num1 + 1):
            robot_a_list = list(G.adj[robot_p])
            robot_a_list.append(robot_p)
            for action_p in robot_a_list:
                for target_p in range(1, node_num1 + 1):
                    state_cur = robot_p * (node_num1 + 1) + target_p
                    dis = nx.dijkstra_path_length(G, source = action_p, target = target_p)
                    step = (V_max - V_min)/(M - 1)

                    grid = (dis - V_min)/step

                    grid_ceil = ceil(grid)
                    grid_floor = ceil(grid)
                    if grid_ceil == grid_floor :
                        Z_matrix1[state_cur][action_p][grid_ceil] = 1
                    else:
                        Z_matrix1[state_cur][action_p][grid_ceil] = (grid-grid_floor)/(grid_ceil-grid_floor)
                        Z_matrix1[state_cur][action_p][grid_floor] = (grid_ceil - grid)/(grid_ceil-grid_floor)

        #             without warm start:

        #             In the House Environment environment, when the MuRES target is the minimum capture time,
        #             if Dijkstra is used for warm start, the capture effect will be close to the final result from the beginning.
        #             Since we need to observe the learning curve, there is no need to use a warm start with Dijstra.
        #
        #             In big maps, we focus on the performance of our DRL_Searcher, so that the warm start is involved

                    if self.Environment_num == 1 and self.MuRES_obj == 1:
                        if action_p == robot_p:
                            Z_matrix1[state_cur][action_p][0] = 1
                            for k in range(1,M):
                                Z_matrix1[state_cur][action_p][k] = 0
                        else:
                            for k in range(M):
                                Z_matrix1[state_cur][action_p][k] = 1/M


        # _______________________________________
        return deepcopy(Z_matrix1)

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

    def __decision_making(self, T, robot_cur_position, target_cur_position, Z_matrix):
        G, MuRES_objective, M, V_min, V_max, epsilon = self.G, self.MuRES_obj, self.M, self.V_min, self.V_max, self.epsilon
        node_num1 = G.number_of_nodes()
        cur_state = robot_cur_position * (node_num1 + 1) + target_cur_position
        robot_act_list = list(G.adj[robot_cur_position])
        robot_act_list.append(robot_cur_position)
        random.shuffle(robot_act_list)
        act = 0
        best_act = 0
        min_act = 10000000000
        max_act = 0
        #     min t ：
        if MuRES_objective == 1:
            for j in robot_act_list:
                su = 0
                step = (V_max-V_min)/(M-1)
                for k in range(M):
                    su += Z_matrix[cur_state][j][k] * (V_min + k*step)
                if su < min_act:
                    best_act = j
                    min_act = su
            # TO DO :CDF
            if random.random() >= epsilon:
                act = best_act
            else:
                act = random.choice(robot_act_list)
            return act
        #     max PD：
        else:
            for j in robot_act_list:
                su = 0
                step = (V_max-V_min)/(M-1)
                for k in range(M):
                    su += Z_matrix[cur_state][j][k]
                    if V_min + (k+1) * step > T:
                        break
                if su >= max_act:
                    best_act = j
                    max_act = su
            # TO DO :CDF
            if random.random() >= epsilon:
                act = best_act
            else:
                act = random.choice(robot_act_list)
            return act


    def __Z_matrix_update(self, robot_cur_position, target_cur_position, cur_action, target_nex_position, next_action, Z_matrix):

        alpha, M, V_min, V_max = self.alpha, self.M, self.V_min, self.V_max

        node_num = self.G.number_of_nodes()
        cur_state = robot_cur_position * (node_num + 1) + target_cur_position
        next_state = cur_action * (node_num + 1) + target_nex_position

        next_Z = Z_matrix[next_state][next_action]
        cur_Z = Z_matrix[cur_state][cur_action]
        next_Z_mapping = [0 for x in range(len(next_Z))]
        step = (V_max-V_min)/(M-1)
        for k in range(M):
            p_value = V_min + k * step + 1
            p_p = (p_value-V_min)/step
            p_l = floor(p_p)
            p_u = ceil(p_p)


            if p_l == p_u and p_value < V_max:
                next_Z_mapping[p_l] += next_Z[k]
            elif p_value >= V_max:
                next_Z_mapping[M-1] += next_Z[k]
            else:
                next_Z_mapping[p_l] += (p_u - p_p)/ (p_u - p_l) * next_Z[k]
                next_Z_mapping[p_u] += (p_p - p_l)/ (p_u - p_l) * next_Z[k]
        delta1 = []
        for i in range(M):
            delta1.append(next_Z_mapping[i] - cur_Z[i])
        for k in range(M):
            cur_Z[k] += alpha * delta1[k]

    def __Z_learning_process(self, Emax, Z_num, step):
        Environment_num, G, target_motion, MuRES_objective = self.Environment_num, self.G, self.target_motion, self.MuRES_obj
        T, M, V_min, V_max, alpha, epsilon = self.T, self.M, self.V_min, self.V_max, self.alpha, self.epsilon
        Z_matrix = self.__Z_warm_start()
        counter = 0
        node_num = G.number_of_nodes()
        address = "../Mid_Outputs/" + self.Environment_name + "_MR" + str(self.MuRES_obj) + "/Z_" + str(Z_num)
        while counter <= Emax:
            if(counter % 1000 == 0 and counter > 0 and counter < 10000):
                # print(Z_matrix[38][4])
                np.save( address+ '_mid'+ str(counter) +'.npy' ,np.array(deepcopy(Z_matrix)))
            if(counter % step == 0):
                np.save( address+ '_mid'+ str(counter) +'.npy' ,np.array(deepcopy(Z_matrix)))
            robot_current_position, target_current_position = self.__position_initialize()
            path = []
            while robot_current_position != target_current_position:
                path.append(robot_current_position)

                current_state = robot_current_position * (node_num + 1) + target_current_position

                robot_current_action = self.__decision_making( T - len(path) + 1, robot_current_position, target_current_position, Z_matrix)

                robot_next_position = robot_current_action
                target_next_position = self.__target_moving(target_current_position)

                robot_next_action = self.__decision_making( T - len(path) + 1, robot_next_position, target_next_position, Z_matrix)

                self.__Z_matrix_update(robot_current_position, target_current_position, robot_current_action, target_next_position, robot_next_action, Z_matrix)

                robot_current_position = robot_current_action
                target_current_position = target_next_position
            counter += 1

    def __Z_training_process(self, Emax, Z_num):
        Environment_num, G, target_motion, MuRES_objective = self.Environment_num, self.G, self.target_motion, self.MuRES_obj
        T, M, V_min, V_max, alpha, epsilon = self.T, self.M, self.V_min, self.V_max, self.alpha, self.epsilon
        Z_matrix = self.__Z_warm_start()
        counter = 0
        node_num = G.number_of_nodes()
        address = "../Mid_Outputs/" + self.Environment_name + "_MR" + str(self.MuRES_obj) + "/Z_" + str(Z_num)
        while counter <= Emax:
            robot_current_position, target_current_position = self.__position_initialize()
            path = []
            while robot_current_position != target_current_position:
                path.append(robot_current_position)

                current_state = robot_current_position * (node_num + 1) + target_current_position

                robot_current_action = self.__decision_making( T - len(path) + 1, robot_current_position, target_current_position, Z_matrix)

                robot_next_position = robot_current_action
                target_next_position = self.__target_moving(target_current_position)

                robot_next_action = self.__decision_making( T - len(path) + 1, robot_next_position, target_next_position, Z_matrix)

                self.__Z_matrix_update(robot_current_position, target_current_position, robot_current_action, target_next_position, robot_next_action, Z_matrix)

                robot_current_position = robot_current_action
                target_current_position = target_next_position
            counter += 1
        np.save( address+ '.npy' ,np.array(deepcopy(Z_matrix)))
