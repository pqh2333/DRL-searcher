# DRL_Searcher
## Overview

DRL_Searcher employs distributional reinforcement learning (DRL) to solve the multi-robot efficient search(MuRES) problem in a graph-represented environment. The main contributions:

- DRL-Searcher serves as a **unified** solution to The Multi-Robot Efficient Search (MuRES) problem;
- DRL-Searcher is **fast** in the online decision-making stage and adaptive with real-time environmental feedback;
- DRL-Searcher provides a **model-free** solution to MuRES problem.


## Structures:
The core code of DRL_Searcher can be divided into two parts: Training process and Execution process.

DRL_Searcher  
├── Core  
│   ├── DRL_Searcher_Trainer.py  
│   ├── DRL_Searcher_Executor.py  
├── Inputs  
│   ├── Environment.py  
│   ├── Target_Motion.py  
├── Examples  
│   ├── Example1.py  
│   ├── Simulation1.1.py  
│   ├── Simulation1.2.py  
│   ├── Simulation2.1.py  
│   └── Simulation2.2.py  
├── Mid_Outputs  
│   ├── HOUSE_MR1  
│   ├── HOUSE_MR2  
│   ├── MUSEUM_MR1  
│   ├── MUSEUM_MR2  
│   ├── OFFICE_MR1  
│   └── OFFICE_MR2  
├── Outputs  
│   ├── HOUSE_MR1  
│   ├── HOUSE_MR2  
│   ├── MUSEUM_MR1  
│   ├── MUSEUM_MR2  
│   ├── OFFICE_MR1  
│   └── OFFICE_MR2  

The `Core` file is the core code of DRL_Searcher. Z_matrix is trained by calling `DRL_Searcher_Trainer.py`, and search is performed by calling `DRL_Searcher_Executor.py`.
The `Inputs` file stores the inputs to the DRL_Searcher, including the environment and target movement mode.
The `Mid_Outputs` file and the `Outputs` file are used to store the trained Z_matrix of DRL_Searcher and the simulated search results, respectively.
The `Examples` file details the calling method of DRL_Searcher and the related simulations in the corresponding paper.
For the specific training and execution process, please refer to `example1.py`.
### Training process:
- Simulate and generate multiple sets of robot and target trajectories as the training set.

### Execution process:
- Using robot position and probabilistic target belief (PTB) as input;
- Combining with recency reward to implement implicit cooperation;
- The robot execute action with the biggest returns based on MuRES objective.

## Parameters:
In this part, we briefly introduce some important parameters of DRL_Searcher, and default values.  

**alpha**: learning rate of DRL, default is 0.01.  
**epsilon**: exploration probability in training step, default is 0.05.  
**Emax**: maximum number of training episodes, default is 40000.  
**M, V_min, V_max**: num of basis/ maximum/ minimum of Z value, default is (51, 0, 50) in HOUSE, and (101, 0, 100) in OFFICE and MUSEUM.  
**beta**: balance parameter between Z_value and recency reward, default value is 0.9.  
**Environment_num**: indicates different environments. Environment_num = 1, 2, 3 represent HOUSE, OFFICE, and MUSEUM environment.  
**target_motion_num**: indicates different target motion models. One can enter various baised walking by yourself.  
**MuRES_objective_num**: indicates different MuRES objective. MuRES_objective_num = 1, 2 represent minimize capture time and maximize capture probability within specific time T.  
**training_num**: training times, recommended to be larger(>=50).  
**capture_times**: number of simulated searches, recommended to be larger to obtain a more accurate average number of search steps(>=500).  


## Examples:
A few examples involved to let you know more about the use process of DRL_Searcher.

`Example1`: Shows how to train a Z_matrix and implement search decisions based on the trained Z_matrix. It is worth mentioning that the training process can be done offline.  One can refer to this example to quick start.
