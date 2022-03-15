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
### Training process:
- Simulate and generate multiple sets of robot and target trajectories as the training set.

### Execution process:
- Using robot position and probabilistic target belief (PTB) as input;
- Combining with recency reward to implement implicit cooperation;
- The robot execute action with the biggest returns based on MuRES objective.

## Parameters:
In this part, we briefly introduce some important parameters of DRL_Searcher, and default values.


## Examples:
A few examples involved to let you know more about the use process of DRL_Searcher.
