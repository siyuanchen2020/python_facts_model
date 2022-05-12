import time
import os
import csv
import random
from random import randrange, uniform
from pathlib import Path
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback



state_path = 'C:\\Facts\\Developer\\DQN-Dispatcher\\dqn_state_file.csv'
action_path = 'C:\\Facts\\Developer\\DQN-Dispatcher\\dqn_action_file.csv'
state_file = Path(state_path)
action_file = Path(action_path)


class BasicEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Discrete(1000)

    def step(self, action):
        # if we took an action, we were in state 1
        #state = 0
        #station ID
        #action rule
        if state_file.is_file():
            csvfile = open(state_path, "r")
            line = csvfile.readline()
            row = line.split(";")
            # print('State ID=', row[0], ' Station ID=', row[1])
            csvfile.close()
            os.remove(state_path)
            f = open(action_path, 'w')
            writer = csv.writer(f)

            #use gym variable action
            data = [row[0], row[1], action]
            writer.writerow(data)
            f.close()
            # use as counter
            print('State ID=', row[0], ' Station ID=', row[1], ' Action Rule=', action)
        else:
            # print('state_file not exist')
            time.sleep(0.15)
            #print("file is not open")


        with open('C:\\Facts\\Developer\\DQN-Dispatcher\\state_log.csv', "r") as scraped:
            state_final_line = scraped.readlines()[-1]
            #print(state_final_line)
            scraped.close()
        state_final_line = state_final_line.split(';')
        #print(state_final_line)

        #state = stationID
        state = state_final_line[1]
        LeadTime = state_final_line[-2]
        LeadTime = float(LeadTime)

        tardiness = state_final_line[-1]
        tardiness = float(tardiness)

        #normalize LeadTime
        LT_nor = - (LeadTime - 51435.777)

        reward = LT_nor


        # regardless of the action, game is done after a single step
        done = True

        info = {}

        return state, reward, done, info
    def reset(self):
        state = 0
        return state

    def render(self, mode='human'):
        pass

    def close(self):
        pass
#create custom gym environment
env = BasicEnv()

# set class base callback
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -numpy.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = numpy.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

#set log directory
log_dir = "/tmp/gym/facts/test1"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# monitor the environment
env = Monitor(env, log_dir)


model = DQN("MlpPolicy", env, verbose=1,tensorboard_log=log_dir,learning_rate= 0.2, learning_starts=10000)
callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)
timesteps = 40000
model.learn(total_timesteps=timesteps, callback=callback)

plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "facts_model_plot")
plt.savefig('test1')
plt.show()


#predict
obs = env.reset()

#check how the model runs

for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

