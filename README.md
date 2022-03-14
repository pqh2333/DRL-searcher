# DRL_Searcher
## Overview

DRL_Searcher employs distributional reinforcement learning (DRL) to solve the multi-robot efficient search(MuRES) problem in a graph-represented environment. The main contributions:

- DRL-Searcher serves as a **unified** solution to The Multi-Robot Efficient Search (MuRES) problem;
- DRL-Searcher is **fast** in the online decision-making stage and adaptive with real-time environmental feedback;
- DRL-Searcher provides a **model-free** solution to MuRES problem.


## Structures:
The core code of DRL_Searcher can be divided into two parts: Training process and Execution process.

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
