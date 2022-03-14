

# This example mainly shows how DRL_Searcher works. DRL_Searcher has two main steps:
# 1. In the training step, DRL_Searcher obtains Z_matrix through training accoring to a specific map, MuRES objective, and target motion model.
# 2. In the execution step, Searchers combine Z_matrix, PTB and Recency reward to make decision to search the target.

# In this example, we assume that three robots are in the House environment, searching for a target that moves according to motion model1 based on MuRES objective1, and simulate 100 searches.
# The output is the movement trajectory of the robot team.

import sys
sys.path.append("..")

from Core.DRL_Searcher_Trainer import DRL_Searcher_Trainer
from Core.DRL_Searcher_Executor import DRL_Searcher_Executor




Searcher1_trainer = DRL_Searcher_Trainer(Environment_num = 1, target_motion_num = 1, MuRES_objective_num = 1)

Searcher1_trainer.Z_training(purpose_num = 1, learning_step = 5000, Emax = 40000, training_num = 2)


Searcher1_trainer.Z_training(purpose_num = 2, learning_step = 5000, Emax = 40000, training_num = 1)


Searcher1_executor = DRL_Searcher_Executor(Environment_num = 1, target_motion_num = 1, MuRES_objective_num = 1)


Searcher1_executor.Execute( Z_nums = 2, capture_times= 100, robot_num = 2, beta = 0.9)


Searcher1_executor.Execute_leanring(learning_step_num = 30000, Z_nums = 1, capture_times = 100)
