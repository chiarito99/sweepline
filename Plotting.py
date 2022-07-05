import imp
from pickle import TRUE
import numpy as np
import matplotlib.pyplot as plt
import time
class Plotting:
    def __init__(self, name, xlim=[-100,100], ylim=[-100,100], is_grid=True):
        self.name = name
        self.xlim = xlim
        self.ylim = ylim
        self.is_grid = is_grid

        self.ax = plt.axes(projection ='3d')
        self.ax.set_title(name)
        self.ax.grid(is_grid)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self.ax.axis('auto')
    
    def plot_path(self, path, label):
        path = np.array(path)
        self.ax.plot(path[:,0], path[:,1], label=label)
        plt.legend()

    def plot_animation(self, path1, path2, path3,x_s,y_s,x_start,y_start,x_end,y_end,radius=2.5):
        path1 = np.array(path1)
        path2 = np.array(path2)
        path3 = np.array(path3)
        # ref1 = np.array(ref1)
        # ref2 = np.array(ref2)
        start_time = time.time()

        length = len(path1)
        for i in range(length):
            plt.clf()
            # Reference
            # plt.plot(ref1[:i,0], ref1[:i,1], "-b")
            # plt.plot(ref2[:i,0], ref2[:i,1], "-b")

            plt.plot(path1[:i,0], path1[:i,1], "-g", label="Leader")
            self.draw_circle(path1[i,:2], radius, 'g')

            plt.plot(path2[:i,0], path2[:i,1], "-r", label="UAV 1")
            self.draw_circle(path2[i,:2], radius, 'r')

            plt.plot(path3[:i,0], path3[:i,1], "-b", label="UAV 2")
            self.draw_circle(path3[i,:2], radius, 'b')

            
            if i == length-1:
                for j in range(i):
                    self.draw_circle(path1[j,:2], radius, 'lightgray')
                    self.draw_circle(path2[j,:2], radius, 'lightgray')
                    self.draw_circle(path3[j,:2], radius, 'lightgray')

            plt.gcf().canvas.mpl_connect('key_release_event',
                                            lambda event:
                                            [exit(0) if event.key == 'escape' else None])
            # plt.title("Omni1: vx: {:.2f}, vy: {:.2f}, omega: {:.2f}\nOmni2: vx: {:.2f}, vy: {:.2f}, omega: {:.2f}".format(\
            #     path1[i,3], path1[i,4], path1[i,5], path2[i,3], path2[i,4], path2[i,5]))
            plt.xlim(self.xlim)
            plt.ylim(self.ylim)
            plt.grid(self.is_grid)
            end_time = time.time()
            elapsed_time = end_time - start_time
            plt.title("elapsed_time:{0}".format(elapsed_time)+ "[sec]")
            plt.plot(x_s,y_s,marker = "o", color = "red")
            plt.plot(x_start,y_start,marker= ">", color = "green")
            plt.plot(x_end,y_end,marker="*",color = "blue")
            plt.legend()
            plt.pause(0.001)
        

    def draw_circle(self, center, radius, color):
        q = np.arange(0, 2*np.pi+np.pi/6, np.pi/6)
        x = center[0] + radius*np.sin(q)
        y = center[1] + radius*np.cos(q)
        plt.plot(x, y, color=color)
