import csv


with open('C:\\Facts\\Developer\\DQN-Dispatcher\\state_log.csv', "r", encoding="utf-8", errors="ignore") as scraped:
    final_line = scraped.readlines()[-1]
    print(final_line)

word = final_line.split(';')
print(word)
print(word[-1])
print(word[-2])
"""
state_log_file = csv.reader(open('C:\\Facts\\Developer\\DQN-Dispatcher\\state_log.csv'))
for line in state_log_file:
    print(line)
"""

"""

        State ID

        Simulation Clock Time (sec.) ? 

        Station ID

        Current Rule ID

        Average required processing time waiting for this operation

        Average remaining processing time of the current tasks in the store

        Average lead-time so far of the current tasks in the current store

        Average slack time of the current tasks in the store

        Average critical ratio of the current tasks in the store

        Average remaining processing time of ALL current tasks

        Average lead-time so far of ALL tasks (a clear indication of the lead-time)

        Average slack time of ALL tasks

                NOT USED at this stage: Average critical ratio of ALL tasks

        Current Rule Matrix [Op2, Op3, Op4, Op5, Op6]

        Current Global Leadtime

        Current Global Tardiness



"""
