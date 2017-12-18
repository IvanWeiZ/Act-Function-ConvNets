from numpy import *
import math
import matplotlib.pyplot as plt
import numpy as np


x = linspace(-5,5,1000)
fig, ax = plt.subplots()

ax.plot(x, x,label="1" )
ax.plot(x, 1+x,label="2" )

ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
legend = ax.legend(loc='upper right')
plt.xlabel("Time Steps")
plt.ylabel("w")
plt.title("w history")
plt.show()