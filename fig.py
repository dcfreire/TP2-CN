import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig  = plt.figure(figsize=(12,8))
def animate(frame):
    im = plt.imread(f"figs/{frame}.png")
    plt.axis('off')
    plt.imshow(im)

ani = animation.FuncAnimation(fig, animate, frames=600, interval=(50))
ani.save("test.mp4")