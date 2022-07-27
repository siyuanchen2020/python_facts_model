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
LeadTime_list = []
Tardiness_list = []


class BasicEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(0, 255, [1, 1, 6])


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
            time.sleep(0.2)
            #print("file is not open")


        with open('C:\\Facts\\Developer\\DQN-Dispatcher\\state_log.csv', "r") as scraped:
            state_final_line = scraped.readlines()[-1]
            #print(state_final_line)
            scraped.close()
        state_final_line = state_final_line.split(';')


        # normalize the input state within [0,1]
        # approach: MIN-MAX normalization
        """a = (float(state_final_line[1]) - 2)/(3-2)
        b = (float(state_final_line[2]) - 0)/(5-0)
        c = (float(state_final_line[3]) - 600)/(2000-600)
        d = (float(state_final_line[4]) - 2400)/(10000-2400)
        e = (float(state_final_line[5]) - 650)/(52500-650)
        f = (float(state_final_line[6]) - (-5600))/(44200 - (-5600))
        g = (float(state_final_line[7]) - (-0.6))/(32-(-0.6))
        h = (float(state_final_line[8]) - 2700)/(6700-2700)
        i = (float(state_final_line[9]) - 480)/(25100-480)
        j = (float(state_final_line[10]) - 5600)/(32000-5600)
        k = (float(state_final_line[11]) - 0)/(5-0)
        l = (float(state_final_line[12]) - 0)/(5-0)"""

        """state = numpy.array([state_final_line[11], state_final_line[12],
                             state_final_line[13], state_final_line[14], state_final_line[15]])"""
        state = numpy.array([state_final_line[1], state_final_line[11], state_final_line[12], state_final_line[13], state_final_line[14], state_final_line[15]])



        LeadTime = state_final_line[-3]
        LeadTime = float(LeadTime)

        tardiness = state_final_line[-1]
        tardiness = float(tardiness)

        #reward = - tardiness

        # LT_nor = - (LeadTime - 21000) test123

        #LT_nor = - (LeadTime - 21153.87)  simple normalization

        #new normalization for reward according to paper (Tt - T(q))/MAX(Tt, T(q))
        #Tt means the average value for reward before
        #T(q) means average value after the current timestep
        """if len(LeadTime_list) == 0:
            Tt = 0
        else:
            Tt = sum(LeadTime_list) / len(LeadTime_list)
        LeadTime_list.append(LeadTime)
        Tq = sum(LeadTime_list)/len(LeadTime_list)
        if Tq == 0:
            Nor_LT = Tq / 1
        else:
            Nor_LT = (Tt - Tq) / max(Tt, Tq)

        if len(Tardiness_list) == 0:
            Tp = 0
        else:
            Tp = sum(Tardiness_list) / len(Tardiness_list)
        Tardiness_list.append(tardiness)
        Tu = sum(Tardiness_list)/len(Tardiness_list)
        if Tu == 0:
            Nor_Ta = Tu / 1
        else:
            Nor_Ta = (Tp - Tu) / min(Tp, Tu)"""
        Nor_LT = -LeadTime/(51000)
        Nor_Ta = -tardiness/(2800)
        reward = 0.5*(Nor_LT) + 0.5*(Nor_Ta)

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
log_dir = "/tmp/gym/facts/6s/new/nor"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# monitor the environment
env = Monitor(env, log_dir)


model = DQN("MlpPolicy", env, verbose=1,tensorboard_log=log_dir,learning_rate= 0.1, learning_starts=5000)
callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)
timesteps = 10000
model.learn(total_timesteps=timesteps, callback=callback)

plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "facts_model_plot")
plt.savefig('test1')
#plt.show()


"""#predict
obs = env.reset()

#check how the model runs

for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()"""

