import numpy as np
import matplotlib.pyplot as pl

PCR = [12.081664,15.550240,18.315744,22.201504,25.856833,24.546623,27.076736,28.170143,30.936960,33.933281]
Thomas = [3.804032,6.424288,13.405312,17.203615,57.491871,67.187645,66.201218,91.869728,132.001022,141.114075]
Tailles = [2,4,8,16,32,64,128,256,512,1024]

n = 10
a = 1 + np.arange(n)
x = 2**a
"""
pl.xlabel('Tailles')
pl.ylabel('Temps (ms)')
pl.xticks(x, Tailles)
pl.plot(PCR, Thomas)
pl.show()
"""

fig = pl.figure(1, figsize=(7,7))
ax2  = fig.add_subplot(111)


pl.xlabel('Matrix size')
pl.ylabel('Time (ms)')
pl.grid(True)
line1, = ax2.plot(a, PCR) #we plot y as a function of a, which parametrizes x
line1.set_label('PCR')
line2, = ax2.plot(a, Thomas) #we plot y as a function of a, which parametrizes x
line2.set_label('Thomas')
ax2.xaxis.set_ticks(a) #set the ticks to be a
ax2.xaxis.set_ticklabels(x) # change the ticks' names to x
ax2.legend()
pl.title("Comparing the Thomas and PCR algorithms on a Nvidia GeForce 930MX")
pl.show()