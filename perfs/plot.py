import matplotlib.pyplot as plt
import numpy as np

x = []
y = []

with open("time_record_merge_sort.txt","r") as f:
    line = f.readline()
    while line:
        line_array = line.split()
        x.append(int(line_array[0]))
        y.append(float(line_array[1]))
        line = f.readline()

print("x : ",x)
print("y : ",y)


fig, ax = plt.subplots(figsize=(8, 8))

plt.xlabel("Array size")
plt.ylabel("Execution time (ms)")
plt.grid(True)
line, = ax.plot(x[0:13], y[0:13], color='green')
ax.xaxis.set_ticks([0,1024,2048,4096, 8192])
plt.title("Merge Sort on Nvidia GeForce GTX 1050")
plt.show()