"""
Implementation of TAMER (Knox & Stone, 2009).
During training, use 'W' and 'A' keys for positive and negative feedback, respectively.
"""
"""
Implementation of TAMER+RL.
Combines TAMER with Q-Learning to integrate human feedback and environmental rewards.
"""

import asyncio
import gymnasium as gym
from tamer.agent import TamerPlusRL, Tamer

async def main():
    # nv = gym.make('MountainCar-v0', render_mode='rgb_array')  # Uncomment to use the MountainCar environment
    # env = gym.make('CartPole-v1', render_mode='rgb_array')  # Uncomment to use the CartPole environment
    env = gym.make('Acrobot-v1', render_mode='rgb_array')  # Uncomment to use the Acrobot environment

    #################################################################
    # ----------------- TAMER Hyperparameters ------------------------
    discount_factor = 1  # Discount factor for Q-Learning
    epsilon = 0  # Initial probability for exploration in Q-Learning
    min_eps = 0  # Minimum epsilon value after decay
    tame = True
    num_episodes = 5  # Number of training episodes
    ##################################################################

    # ##################################################################
    # # ---------------- TAMER+RL Hyperparameters ----------------------
    # discount_factor = 0.99  # Discount factor for Q-Learning
    # epsilon = 0.1  # Initial probability for exploration in Q-Learning
    # min_eps = 0.01  # Minimum epsilon value after decay
    # num_episodes = 10  # Number of training episodes
    # ###################################################################

    # Time per step for human feedback (in seconds)
    tamer_training_timestep = 0.3

    #################################################################
    # --------------- TAMER Agent Initialization --------------------
    agent = Tamer(env, num_episodes, discount_factor, epsilon, min_eps, tame, tamer_training_timestep, model_file_to_load=None)
    #################################################################

    # ##################################################################
    # # ---------- TAMER+RL Agent Initialization -----------------------
    # agent = TamerPlusRL(
    #     env=env,
    #     num_episodes=num_episodes,
    #     discount_factor=discount_factor,
    #     epsilon=epsilon,
    #     min_eps=min_eps,
    #     ts_len=tamer_training_timestep,
    #     model_file_to_load=None  # Optional: load a pre-trained model
    # )
    # ##################################################################

    print("Awaiting agent.train")
    await agent.train(model_file_to_save='autosave')  # Save the model after training

    print("Starting agent.play")
    agent.play(n_episodes=1, render=True)  # Play 1 episode with graphics rendered
    agent.evaluate(n_episodes=75)  # Evaluate the agent for 75 episodes

if __name__ == '__main__':
    asyncio.run(main())
