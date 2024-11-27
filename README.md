# Social Robotics - Final Project 2024/25
This repository contains the implementation and analysis of a reinforcement learning project utilizing the TAMER framework (Training an Agent Manually via Evaluative Reinforcement). The work is part of the Social Robotics course under the Parcours Ingénierie pour la Santé program.

## Project Overview
TAMER is a framework designed to train reinforcement learning agents through human feedback. It allows agents to learn directly from human evaluators by incorporating positive and negative feedback into their action-value function. This method is particularly advantageous in environments where defining explicit reward functions is complex.

In this project, we implemented TAMER in the following environments:

*  **MountainCar**: An agent must navigate a car to the top of a hill using momentum.
*  **CartPole**: The goal is to balance a pole on a cart.
*  **Acrobot**: A two-link robotic arm must swing up to a goal position.
  
We also explored the integration of TAMER with reinforcement learning (TAMER+RL) and compared its performance with standard reinforcement learning techniques.

## Key Features
*  **Human Feedback Integration**: Real-time feedback provided via a keyboard interface (`W` for positive, `A` for negative feedback).
*  **Comparative Analysis**: Performance of agents trained with and without human feedback.
*  **Alternative Environments**: Implementation of TAMER in CartPole and Acrobot environments.
*  **TAMER+RL Hybrid**: Testing the combination of human feedback with Q-learning to try improving learning stability and efficiency.

## Repository Structure
* `/src`: Python scripts for training agents in different environments.
  *  `run.py`: Main script to run experiments.
  *  `tamer.py`: Implementation of the TAMER framework.
* `/docs`: Documentation, including the project report and references.
* `/results`: Performance data and visualizations from experiments.

## How to Use
### Requirements
*  Python 3.8 or higher
*  Libraries: `gym`, `numpy`, `pygame`, `matplotlib`

### Run the Experiments
1. Clone this repository:

```git clone https://github.com/sabelaara/Social_Robotics_Final_Project```

2. Navigate to the project directory:

```cd social-robotics-final```

3. Start an experiment:

```python src/run.py```

### Results Summary
Our experiments demonstrated that:

*  **TAMER with human feedback** accelerates learning compared to standard reinforcement learning, particularly in complex environments like MountainCar.
*  **TAMER+RL** can combine the strengths of human feedback and reward signals but showed variability in results due to alignment challenges between feedback and reward systems.
  
## Contributors
Sabela Ara Solla

Sofía Hernández Pérez

Alejandra Valle López

## Acknowledgments
This work is based on the TAMER framework and utilizes OpenAI Gym environments for reinforcement learning experiments.


