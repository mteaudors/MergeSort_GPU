import matplotlib.pyplot as plt
import numpy as np

x = [pow(2,i) for i in range(1,14)]

# with batch
y = []

#without bacth
y2 = []

with open("time_P1_batch.txt","r") as f:
    for k in range(len(x)):
        buffer = []
        for i in range(5):
            line = f.readline()
            buffer.append(float(line.split()[1]))
        y.append(np.mean(buffer))

with open("time_P1_no_batch.txt","r") as f:
    for k in range(len(x)):
        buffer = []
        for i in range(5):
            line = f.readline()
            buffer.append(float(line.split()[1]))
        y2.append(np.mean(buffer))

########################################################################

fig = plt.figure(1,figsize=(12,8))
plt.suptitle("Nvidia GeForce GTX 1050", fontsize=16)

ax = plt.subplot(1,2,1)
ax.plot(x, y, color='green')
ax.xaxis.set_ticklabels([0,1024,2048,4096, 8192])
ax.xaxis.set_ticks(np.arange(5)*2048)
plt.grid(True)
plt.xlabel("Array size")
plt.ylabel("Execution time (ms)")
plt.title("With Batch")

ax = plt.subplot(1,2,2)
ax.plot(x, y2, color='orange')
ax.xaxis.set_ticklabels([0,1024,2048,4096, 8192])
ax.xaxis.set_ticks(np.arange(5)*2048)
plt.grid(True)
plt.xlabel("Array size")
plt.ylabel("Execution time (ms)")
plt.title("Without Batch")

plt.show()