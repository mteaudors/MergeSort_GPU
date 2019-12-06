import matplotlib.pyplot as plt

x = []
y = []

with open("time_record_merge_sort.txt","r") as f:
    line = f.readline()
    while line:
        line_array = line.split()
        x.append(int(line_array[0]))
        y.append(float(line_array[1]))
        line = f.readline()

plt.plot(x,y)
