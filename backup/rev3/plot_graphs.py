import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from sqlite_db import SqliteDB 

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    data_slot = SqliteDB().get_all_data()  
    xar, yar, _ = map(list, zip(*data_slot))
    ax1.clear()
    ax1.plot(yar, yar)

def main():    
    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.title('My first graph!') 
    plt.show()