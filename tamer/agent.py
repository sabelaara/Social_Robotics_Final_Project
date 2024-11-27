import datetime as dt
import os
import pickle
import time
import uuid
from itertools import count
from pathlib import Path
from sys import stdout
from csv import DictWriter

import numpy as np
from sklearn import pipeline, preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

import matplotlib.pyplot as plt
import cv2

# MOUNTAINCAR_ACTION_MAP = {0: 'left', 1: 'none', 2: 'right'} # Discomment if using MountainCar Environment
# CARTPOLE_ACTION_MAP = {0: 'left', 1: 'none', 2: 'right'} # Discomment if using CartPole Environment
ACROBOT_ACTION_MAP = {
    0: 'Down',  # Apply negative torque to the first joint (move down)
    1: 'None',         # Do not apply any torque
    2: 'Up'    # Apply positive torque to the first joint (move up)
} # Discomment if using Acrobot Environment 
MODELS_DIR = Path(__file__).parent.joinpath('saved_models')
LOGS_DIR = Path(__file__).parent.joinpath('logs')


class SGDFunctionApproximator:
    """SGD function approximator with RBF preprocessing."""
    def __init__(self, env):
        # Feature preprocessing: Normalize to zero mean and unit variance
        observation_examples = np.array(
            [env.observation_space.sample() for _ in range(10000)], dtype='float64'
        )
        print(observation_examples[0, :])
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # RBF kernels with different variances
        self.featurizer = pipeline.FeatureUnion(
            [
                ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
                ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
                ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
                ('rbf4', RBFSampler(gamma=0.5, n_components=100)),
            ]
        )
        self.featurizer.fit(self.scaler.transform(observation_examples))
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate='constant')
            model.partial_fit([self.featurize_state(env.reset()[0])], [0])
            self.models.append(model)

    def predict(self, state, action=None):
        features = self.featurize_state(state)
        if action is None:
            return [m.predict([features])[0] for m in self.models]
        else:
            return self.models[action].predict([features])[0]

    def update(self, state, action, td_target):
        features = self.featurize_state(state)
        self.models[action].partial_fit([features], [td_target])

    def featurize_state(self, state):
        """Returns the featurized representation for a state."""
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]

class Tamer:
    """
    QLearning Agent adapted to TAMER using steps from:
    http://www.cs.utexas.edu/users/bradknox/kcap09/Knox_and_Stone,_K-CAP_2009.html
    """
    def __init__(
        self,
        env,
        num_episodes,
        discount_factor=1,  # only affects Q-learning
        epsilon=0, # only affects Q-learning
        min_eps=0,  # minimum value for epsilon after annealing
        tame=True,  # set to false for normal Q-learning
        ts_len=0.2,  # length of timestep for training TAMER
        output_dir=LOGS_DIR,
        model_file_to_load=None  # filename of pretrained model
    ):
        self.tame = tame
        self.ts_len = ts_len
        self.env = env
        self.uuid = uuid.uuid4()
        self.output_dir = output_dir

        # init model
        if model_file_to_load is not None:
            print(f'Loaded pretrained model: {model_file_to_load}')
            self.load_model(filename=model_file_to_load)
        else:
            if tame:
                self.H = SGDFunctionApproximator(env)  # init H function
            else:  # optionally run as standard Q Learning
                self.Q = SGDFunctionApproximator(env)  # init Q function

        # hyperparameters
        self.discount_factor = discount_factor
        self.epsilon = epsilon if not tame else 0
        self.num_episodes = num_episodes
        self.min_eps = min_eps

        # calculate episodic reduction in epsilon
        self.epsilon_step = (epsilon - min_eps) / num_episodes

        # reward logging
        self.reward_log_columns = [
            'Episode',
            'Ep start ts',
            'Feedback ts',
            'Human Reward',
            'Environment Reward',
        ]
        self.reward_log_path = os.path.join(self.output_dir, f'{self.uuid}.csv')

    def act(self, state):
        """ Epsilon-greedy Policy """
        if np.random.random() < 1 - self.epsilon:
            preds = self.H.predict(state) if self.tame else self.Q.predict(state)
            return np.argmax(preds)
        else:
            return np.random.randint(0, self.env.action_space.n)

    def _train_episode(self, episode_index, disp, max_timesteps=None):
        print(f'Episode: {episode_index + 1}  Timestep:', end='')
        cv2.namedWindow('OpenAI Gymnasium Training', cv2.WINDOW_NORMAL)

        rng = np.random.default_rng()
        tot_reward = 0
        state, _ = self.env.reset()
        ep_start_time = dt.datetime.now().time()
        with open(self.reward_log_path, 'a+', newline='') as write_obj:
            dict_writer = DictWriter(write_obj, fieldnames=self.reward_log_columns)
            # Move dict_writer.writeheader() to only be called once
            if episode_index == 0:  # Write header only for the first episode
                dict_writer.writeheader()
            
            for ts in count():
                if max_timesteps and ts >= max_timesteps:
                    print(f"Timestep limit reached ({max_timesteps}), ending episode.")
                    break
                print(f' {ts}', end='')
                frame_bgr = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                cv2.imshow('OpenAI Gymnasium Training', frame_bgr)
                key = cv2.waitKey(25)
                if key == 27:
                    break

                action = self.act(state)
                if self.tame:
                    disp.show_action(action)

                next_state, reward, done, info, _ = self.env.step(action)

                if not self.tame:
                    if done and next_state[0] >= 0.5:
                        td_target = reward
                    else:
                        td_target = reward + self.discount_factor * np.max(self.Q.predict(next_state))
                    self.Q.update(state, action, td_target)
                else:
                    now = time.time()
                    while time.time() < now + self.ts_len:
                        time.sleep(0.01)
                        human_reward = disp.get_scalar_feedback()
                        feedback_ts = dt.datetime.now().time()
                        if human_reward != 0:
                            dict_writer.writerow({
                                'Episode': episode_index + 1,
                                'Ep start ts': ep_start_time,
                                'Feedback ts': feedback_ts,
                                'Human Reward': human_reward,
                                'Environment Reward': reward
                            })
                            self.H.update(state, action, human_reward)
                            break

                tot_reward += reward
                if done:
                    print(f'  Reward: {tot_reward}')
                    cv2.destroyAllWindows()
                    break

                stdout.write('\b' * (len(str(ts)) + 1))
                state = next_state

        # Decay epsilon
        if self.epsilon > self.min_eps:
            self.epsilon -= self.epsilon_step

        return tot_reward  # Return the total reward for this episode


    async def train(self, model_file_to_save=None, max_timesteps=None):
        """
        TAMER (or Q learning) training loop
        Args:
            model_file_to_save: save Q or H model to this filename
        """
        disp = None
        if self.tame:
            # only init pygame display if we're actually training tamer
            from .interface import Interface_Video, Interface_Keyboard
            # disp = Interface_Video(action_map=MOUNTAINCAR_ACTION_MAP) # Discomment if using MountainCar Environment
            # disp = Interface_Video(action_map=CARTPOLE_ACTION_MAP) # Discomment if using CarPole Environment
            disp = Interface_Video(action_map=ACROBOT_ACTION_MAP) # Discomment if using AcroBot Environment
        
        episode_rewards = []  # Initialize a list to store rewards

        for i in range(self.num_episodes):
            print(f"Num episode : {i}")
            # Call the _train_episode method without await and store the total reward
            tot_reward = self._train_episode(i, disp, max_timesteps=max_timesteps)  # This should be a direct call
            episode_rewards.append(tot_reward)

        print('\nCleaning up...')
        self.env.close()
        if model_file_to_save is not None:
            self.save_model(filename=model_file_to_save)

        return episode_rewards  # Return the list of rewards


    def play(self, n_episodes=1, render=False, max_timesteps=500):
        """
        Run episodes with trained agent
        Args:
            n_episodes: number of episodes
            render: optionally render episodes
            max_timesteps: maximum number of timesteps to run each episode

        Returns: list of cumulative episode rewards
        """
        if render:            
            cv2.namedWindow('OpenAI Gymnasium Playing', cv2.WINDOW_NORMAL)

        self.epsilon = 0  # Disable exploration for playing
        ep_rewards = []  # List to store rewards for each episode
        for i in range(n_episodes):
            state = self.env.reset()[0]
            done = False
            tot_reward = 0
            timestep = 0  # Initialize timestep counter

            while not done and timestep < max_timesteps:
                action = self.act(state)
                next_state, reward, done, info, _ = self.env.step(action)
                tot_reward += reward
                timestep += 1  # Increment timestep counter
                
                if render:
                    frame_bgr = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                    cv2.imshow('OpenAI Gymnasium Playing', frame_bgr)
                    key = cv2.waitKey(25)
                    if key == 27:  # Press ESC to break out
                        done = True
                        break
                
                state = next_state
            
            ep_rewards.append(tot_reward)  # Store total reward for this episode
            print(f'Episode: {i + 1} Reward: {tot_reward} Timesteps: {timestep}')
        
        self.env.close()
        if render:
            cv2.destroyAllWindows()

        return ep_rewards


    def evaluate(self, n_episodes=100):
        print('Evaluating agent')
        rewards = self.play(n_episodes=n_episodes)  # Play episodes to get rewards
        avg_reward = np.mean(rewards)
        print(
            f'Average total episode reward over {n_episodes} '
            f'episodes: {avg_reward:.2f}'
        )
        return rewards  # Return the list of rewards instead of avg_reward

    def save_model(self, filename):
        """
        Save H or Q model to models dir
        Args:
            filename: name of pickled file
        """
        model = self.H if self.tame else self.Q
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, filename):
        """
        Load H or Q model from models dir
        Args:
            filename: name of pickled file
        """
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'rb') as f:
            model = pickle.load(f)
        if self.tame:
            self.H = model
        else:
            self.Q = model



class TamerPlusRL:
    """
    TAMER+RL agent that combines human feedback and environmental rewards
    to train two models: one for human feedback (H) and another for Q-learning (Q).
    """
    def __init__(
        self,
        env,
        num_episodes,
        discount_factor=1,
        epsilon=0,
        min_eps=0,
        ts_len=0.2,
        output_dir=LOGS_DIR,
        model_file_to_load=None
    ):
        self.env = env
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_eps = min_eps
        self.ts_len = ts_len
        self.output_dir = output_dir
        self.uuid = uuid.uuid4()
        self.epsilon_step = (epsilon - min_eps) / num_episodes

        # Initialize models
        if model_file_to_load is not None:
            print(f'Loaded pretrained model: {model_file_to_load}')
            self.load_model(filename=model_file_to_load)
        else:
            self.H = SGDFunctionApproximator(env)  # Human feedback model
            self.Q = SGDFunctionApproximator(env)  # Q-learning model

        # Reward logging
        self.reward_log_columns = [
            'Episode',
            'Ep start ts',
            'Feedback ts',
            'Human Reward',
            'Environment Reward',
        ]
        self.reward_log_path = os.path.join(self.output_dir, f'{self.uuid}.csv')

    def act(self, state):
        """
        Select an action by combining predictions from H and Q models.
        """
        if np.random.random() < 1 - self.epsilon:
            h_preds = self.H.predict(state)
            q_preds = self.Q.predict(state)
            combined_preds = np.add(h_preds, q_preds)  # Combine predictions
            return np.argmax(combined_preds)
        else:
            return np.random.randint(0, self.env.action_space.n)

    def _train_episode(self, episode_index, disp):
        """
        Train for one episode, updating both H and Q models.
        """
        print(f'Episode: {episode_index + 1}  Timestep:', end='')
        cv2.namedWindow('OpenAI Gymnasium Training', cv2.WINDOW_NORMAL)

        rng = np.random.default_rng()
        tot_reward = 0
        state, _ = self.env.reset()
        ep_start_time = dt.datetime.now().time()
        with open(self.reward_log_path, 'a+', newline='') as write_obj:
            dict_writer = DictWriter(write_obj, fieldnames=self.reward_log_columns)
            dict_writer.writeheader()
            for ts in count():
                print(f' {ts}', end='')

                # Render the environment
                frame_bgr = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                cv2.imshow('OpenAI Gymnasium Training', frame_bgr)
                key = cv2.waitKey(25)
                if key == 27:
                    break

                # Determine next action
                action = self.act(state)
                disp.show_action(action)

                # Take action and observe result
                next_state, reward, done, info, _ = self.env.step(action)

                # Update Q model with environmental rewards
                if done and next_state[0] >= 0.5:  # Goal reached
                    td_target = reward
                else:
                    td_target = reward + self.discount_factor * np.max(
                        self.Q.predict(next_state)
                    )
                self.Q.update(state, action, td_target)

                # Collect human feedback and update H model
                now = time.time()
                while time.time() < now + self.ts_len:
                    time.sleep(0.01)
                    human_reward = disp.get_scalar_feedback()
                    feedback_ts = dt.datetime.now().time()
                    if human_reward != 0:
                        dict_writer.writerow(
                            {
                                'Episode': episode_index + 1,
                                'Ep start ts': ep_start_time,
                                'Feedback ts': feedback_ts,
                                'Human Reward': human_reward,
                                'Environment Reward': reward
                            }
                        )
                        self.H.update(state, action, human_reward)
                        break

                tot_reward += reward
                if done:
                    print(f'  Reward: {tot_reward}')
                    cv2.destroyAllWindows()
                    break

                stdout.write('\b' * (len(str(ts)) + 1))
                state = next_state

        # Decay epsilon
        if self.epsilon > self.min_eps:
            self.epsilon -= self.epsilon_step

    async def train(self, model_file_to_save=None):
        """
        Train the TAMER+RL agent.
        """
        disp = None
        from .interface import Interface_video, Interface_Keyboard
        # disp = Interface_Video(action_map=MOUNTAINCAR_ACTION_MAP) # Discomment if using MountainCar Environment
        # disp = Interface_Video(action_map=CARTPOLE_ACTION_MAP) # Discomment if using CarPole Environment
        disp = Interface_Keyboard(action_map=ACROBOT_ACTION_MAP) # Discomment if using AcroBot Environment

        for i in range(self.num_episodes):
            print(f"Num episode : {i}")
            self._train_episode(i, disp)

        print('\nCleaning up...')
        self.env.close()
        if model_file_to_save is not None:
            self.save_model(filename=model_file_to_save)

    def play(self, n_episodes=1, render=False):
        """
        Run episodes with trained agent
        Args:
            n_episodes: number of episodes
            render: optionally render episodes

        Returns: list of cumulative episode rewards
        """
        if render:            
            cv2.namedWindow('OpenAI Gymnasium Playing', cv2.WINDOW_NORMAL)

        self.epsilon = 0
        ep_rewards = []
        for i in range(n_episodes):
            state = self.env.reset()[0]
            done = False
            tot_reward = 0
            # TODO setup a duration criterion in case of impossibility to find a solution
            while not done:
                action = self.act(state)
                next_state, reward, done, info, _ = self.env.step(action)
                tot_reward += reward
                if render:
                    # self.env.render()
                    frame_bgr = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                    cv2.imshow('OpenAI Gymnasium Playing', frame_bgr)
                    key = cv2.waitKey(25)  # Adjust the delay (25 milliseconds in this case)
                    if key == 27:
                        break
                state = next_state
            ep_rewards.append(tot_reward)
            print(f'Episode: {i + 1} Reward: {tot_reward}')
        self.env.close()
        if render:
            cv2.destroyAllWindows()
        return ep_rewards

    def evaluate(self, n_episodes=100):
        print('Evaluating agent')
        rewards = self.play(n_episodes=n_episodes)
        avg_reward = np.mean(rewards)
        print(
            f'Average total episode reward over {n_episodes} '
            f'episodes: {avg_reward:.2f}'
        )
        return avg_reward

    def save_model(self, filename):
        """
        Save both H and Q models to disk.
        """
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'wb') as f:
            pickle.dump({'H': self.H, 'Q': self.Q}, f)

    def load_model(self, filename):
        """
        Load both H and Q models from disk.
        """
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'rb') as f:
            models = pickle.load(f)
        self.H = models['H']
        self.Q = models['Q']
