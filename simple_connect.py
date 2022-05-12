#import pandas as pd
import time
import os
import csv
import random
from random import randrange, uniform
from pathlib import Path

state_path = 'C:\\Facts\\Developer\\DQN-Dispatcher\\dqn_state_file.csv'
action_path = 'C:\\Facts\\Developer\\DQN-Dispatcher\\dqn_action_file.csv'
state_file = Path(state_path)
action_file = Path(action_path)

for i in range(50):
    if  state_file.is_file():
        csvfile = open(state_path, "r")
        line = csvfile.readline()
        row = line.split(";")
        #print('State ID=', row[0], ' Station ID=', row[1])
        csvfile.close()
        os.remove(state_path)
        f = open(action_path, 'w')
        writer = csv.writer(f)
        #now randonly generate a number between 0 and 5 to represent the action (Rule ID).
        irand = randrange(0, 5)
        data = [row[0], row[1], irand]
        writer.writerow(data)
        f.close()
        print('State ID=', row[0], ' Station ID=', row[1], ' Action Rule=', irand)
        print(i)
    else:
        print('state_file not exist')
        print(i)
        time.sleep(1)
