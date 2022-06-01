import matplotlib.pyplot as plt
import csv


random_csv_file = open("state_log_test1.csv")
random_csv = csv.reader(random_csv_file)
random_csv = list(random_csv)
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
