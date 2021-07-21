import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
import numpy as np
fig = plt.figure()

x = np.linspace(1,100)
for i in range(5):
    if i==0:
        y=x**2
    elif i==1:
        y=x**3
    elif i==2:
        y = np.log(x)
    elif i==3:
        y= -x
    else:
        y=np.sqrt(x)
    plt.plot(x,y)
    plt.scatter(y,x)
    plt.pause(1)
    plt.cla()