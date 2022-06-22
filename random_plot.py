import matplotlib.pyplot as plt
import csv
import pandas as pd

#fig, ax = plt.subplots()

random_data = pd.read_csv('random_state_log.csv',delimiter = ";", nrows=1893, usecols=[0,13])
op_data = pd.read_csv('state_log.csv',delimiter = ";", nrows=1893, usecols=[0,13])

random_data.set_index('StateID').plot()
op_data.set_index('StateID').plot()
plt.show()




"""with open("random_state_log.csv", 'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=';')
    for row in lines:
        x.append(row[0])
        y.append((row[-3]))

with open("state_log.csv", 'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=';')
    for row in lines:
        a.append(row[0])
        b.append((row[-2]))

plt.plot(a, b)
plt.show()"""

"""df = pd.read_csv('random_state_log.csv', delimiter = ";", usecols=[0,14])
df.set_index('StateID').plot()
plt.show()"""


"""x = []
y = []

with open("random_state_log.csv", 'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=';')
    for row in lines:
        x.append(row[0])
        y.append((row[-2]))
print(x)
print(y)"""



"""
length = len(random_csv)

timestep = list()
LeadTime = list()
Tardiness = list()

print(length)


for i in range(1, length):
    timestep.append(random_csv[i][0])
    #LeadTime.append(random_csv[i][-4])
    Tardiness.append(random_csv[i][-1])


#plt.plot(timestep, LeadTime)
plt.plot(timestep, Tardiness)
plt.savefig("plot1")
plt.show()
"""