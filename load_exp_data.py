import matplotlib.pyplot as plt
import pickle
import numpy as np

# datafile = "./exp1_results/2025-03-24-16-45-51/data.pickle"
# datafile = "./exp2_results/2025-03-24-19-24-26/data.pickle"
# datafile = "./exp3_results/2025-03-25-12-46-20/data.pickle"
datafile = "./exp5_results/smoothed/data.pickle"

with open(datafile, 'rb') as f:
    data = pickle.load(f)

t = np.array(data['timestamp'])
t = t - t[0]
q = np.array(data['q'])
time_per_loop = np.array(data['time_per_loop'])
dq = np.array(data['dq'])

plt.figure(0)
plt.plot(t, q)
plt.xlabel('Time (s)')
plt.ylabel('Joint angles (rad)')
plt.title('Joint angles vs time')
plt.grid()
plt.show()

plt.figure(1)
plt.hist(time_per_loop, bins=100)
plt.xlabel('Time (s)')
plt.ylabel('Frequency')
plt.title('Time per loop')
plt.grid()
plt.show()


plt.figure(1)
plt.plot(t, dq)
plt.xlabel('Time (s)')
plt.ylabel('Control input')
plt.title('Control input vs time')
plt.ylim([-1.0, 1.0])
plt.grid()
plt.show()