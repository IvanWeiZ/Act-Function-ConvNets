from numpy import *
import math
import matplotlib.pyplot as plt
import numpy as np



x = np.linspace(-10,10,10000)
fig, ax = plt.subplots()

ax.plot(x, np.maximum(np.zeros(len(x)),x),label="relu",linewidth=3)
lrelu=x.copy()
for i in range(len(x)):
	if lrelu[i]<0:
		lrelu[i]=lrelu[i]*0.2
ax.plot(x, lrelu,label="1",linestyle="--")


ax.axhline(y=0, color='k',alpha=0.5)
ax.axvline(x=0, color='k',alpha=0.5)
legend = ax.legend(loc='upper right')
plt.xlabel("x")
plt.ylabel("y")
plt.title("w history")
plt.show()