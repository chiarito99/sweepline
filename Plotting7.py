
from importlib.resources import path
from pickle import TRUE
from cv2 import sqrt
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from env import Env
from sympy import false
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

    def plot_animation(self, path1, path2, path3,path4,path5,path6,path7,x_s,y_s,x_start,y_start,x_end,y_end, length ,width,obs, radius = 2.5):
        path1 = np.array(path1)
        path2 = np.array(path2)
        path3 = np.array(path3)
        path4 = np.array(path4)
        path5 = np.array(path5)
        path6 = np.array(path6)
        path7 = np.array(path7)
        # ref1 = np.array(ref1)
        # ref2 = np.array(ref2)
        
        length_p = len(path1)
        yaw1=[]
        yaw2=[]
        yaw3=[]
        yaw4=[]
        yaw5=[]
        yaw6=[]
        yaw7=[]

        for i in range(length_p):
            #Yaw of Robot
            if i < length_p - 1:
                yaw_1 = math.atan2(path1[i+1,1]-path1[i,1],path1[i+1,0]-path1[i,0])
                yaw_2 = math.atan2(path2[i+1,1]-path2[i,1],path2[i+1,0]-path2[i,0])
                yaw_3 = math.atan2(path3[i+1,1]-path3[i,1],path3[i+1,0]-path3[i,0])
                yaw_4 = math.atan2(path4[i+1,1]-path4[i,1],path4[i+1,0]-path4[i,0])
                yaw_5 = math.atan2(path5[i+1,1]-path5[i,1],path5[i+1,0]-path5[i,0])
                yaw_6 = math.atan2(path6[i+1,1]-path6[i,1],path6[i+1,0]-path6[i,0])
                yaw_7 = math.atan2(path7[i+1,1]-path7[i,1],path7[i+1,0]-path7[i,0])
            else:
                yaw_1 = math.pi/2
                yaw_2 = math.pi/2
                yaw_3 = math.pi/2
                yaw_4 = math.pi/2
                yaw_5 = math.pi/2
                yaw_6 = math.pi/2
                yaw_7 = math.pi/2
            yaw1.append(yaw_1)
            yaw2.append(yaw_2)
            yaw3.append(yaw_3)
            yaw4.append(yaw_4)
            yaw5.append(yaw_5)
            yaw6.append(yaw_6)
            yaw7.append(yaw_7)

        start_time = time.time()
        for i in range(length_p):
            plt.clf()
            plt.plot(x_s,y_s,marker = "o", color = "red")
            plt.fill(x_s,y_s,facecolor='lightblue')
            plt.plot(x_start,y_start,marker= ">", color = "green")
            plt.plot(x_end,y_end,marker="*",color = "blue")
            for k in range(len(obs)):
                self.draw_circle(obs[k,:2],obs[k,2], 'r')
            # Reference
            # plt.plot(ref1[:i,0], ref1[:i,1], "-b")
            # plt.plot(ref2[:i,0], ref2[:i,1], "-b")
            
    
            # Plot Rectangle
            plt.plot(path1[:i,0], path1[:i,1], "-g", label="Leader")
            self.draw_rectangle(path1[i,:2], length, width,yaw1[i], 'b')

            plt.plot(path2[:i,0], path2[:i,1], "-b", label="Follower 1")
            self.draw_rectangle(path2[i,:2], length, width,yaw2[i], 'g')

            plt.plot(path3[:i,0], path3[:i,1], "-b", label="Follower 2")
            self.draw_rectangle(path3[i,:2], length, width,yaw3[i], 'g')

            plt.plot(path4[:i,0], path4[:i,1], "-b", label="Follower 3")
            self.draw_rectangle(path4[i,:2], length, width,yaw4[i], 'g')

            plt.plot(path5[:i,0], path5[:i,1], "-b", label="Follower 4")
            self.draw_rectangle(path5[i,:2], length, width,yaw5[i], 'g')
            
            plt.plot(path6[:i,0], path6[:i,1], "-b", label="Follower 5")
            self.draw_rectangle(path6[i,:2], length, width,yaw6[i], 'g')

            plt.plot(path7[:i,0], path7[:i,1], "-b", label="Follower 6")
            self.draw_rectangle(path7[i,:2], length, width,yaw7[i], 'g')
            # #Plot Circle
            # plt.plot(path1[:i,0], path1[:i,1], "-g", label="Leader")
            # self.draw_circle(path1[i,:2], radius, 'g')

            # plt.plot(path2[:i,0], path2[:i,1], "-r", label="UAV 1")
            # self.draw_circle(path2[i,:2], radius, 'r')

            # plt.plot(path3[:i,0], path3[:i,1], "-b", label="UAV 2")
            # self.draw_circle(path3[i,:2], radius, 'b')
            if i==length_p-1:
                for j in range(i):
                    self.draw_overlaid(path1[j,:2], length, width,yaw1[j], 'lightgray')
                    self.draw_overlaid(path2[j,:2], length, width,yaw2[j], 'lightgray')
                    self.draw_overlaid(path3[j,:2], length, width,yaw3[j], 'lightgray')
                    self.draw_overlaid(path4[j,:2], length, width,yaw4[j], 'lightgray')
                    self.draw_overlaid(path5[j,:2], length, width,yaw5[j], 'lightgray')
                    self.draw_overlaid(path6[j,:2], length, width,yaw6[j], 'lightgray')
                    self.draw_overlaid(path7[j,:2], length, width,yaw7[j], 'lightgray')
                # self.draw_circle(path1[j,:2], radius, 'lightgray')
                # self.draw_circle(path2[j,:2], radius, 'lightgray')
                # self.draw_circle(path3[j,:2], radius, 'lightgray')
                    
            plt.gcf().canvas.mpl_connect('key_release_event',
                                            lambda event:
                                            [exit(0) if event.key == 'escape' else None])
            # plt.title("Omni1: vx: {:.2f}, vy: {:.2f}, omega: {:.2f}\nOmni2: vx: {:.2f}, vy: {:.2f}, omega: {:.2f}".format(\
            #     path1[i,3], path1[i,4], path1[i,5], path2[i,3], path2[i,4], path2[i,5]))
            plt.xlim(self.xlim)
            plt.ylim(self.ylim)
            plt.grid(false)
            end_time = time.time()
            elapsed_time = end_time - start_time
            plt.title("elapsed_time:{0}".format(elapsed_time)+ "[sec]")
            # plt.plot(x_s,y_s,marker = "o", color = "red")
            # plt.fill(x_s,y_s,facecolor='lightblue')
            # plt.plot(x_start,y_start,marker= ">", color = "green")
            # plt.plot(x_end,y_end,marker="*",color = "blue")
            plt.legend()
            plt.pause(0.01)
        

    def draw_circle(self, center, radius, color):
        q = np.arange(0, 2*np.pi+np.pi/6, np.pi/6)
        x = center[0] + radius*np.sin(q)
        y = center[1] + radius*np.cos(q)
        plt.plot(x, y, color=color)

    def draw_rectangle(self ,center,length, width, yaw, color):
        x = []
        y = []
        x.append(center[0]+(+ width/2)*math.cos(yaw)-(+ length/2)*math.sin((yaw)))
        y.append(center[1]+(+ width/2)*math.sin(yaw)+(+ length/2)*math.cos((yaw)))
        x.append(center[0]+(+ width/2)*math.cos(yaw)-(- length/2)*math.sin((yaw)))
        y.append(center[1]+(+ width/2)*math.sin(yaw)+(- length/2)*math.cos((yaw)))
        x.append(center[0]+(- width/2)*math.cos((yaw))-(- length/2)*math.sin((yaw)))
        y.append(center[1]+(- width/2)*math.sin((yaw))+(- length/2)*math.cos((yaw)))
        x.append(center[0]+(- width/2)*math.cos((yaw))-(+ length/2)*math.sin((yaw)))
        y.append(center[1]+(- width/2)*math.sin((yaw))+(+ length/2)*math.cos((yaw)))
        x.append(center[0]+(+ width/2)*math.cos((yaw))-(+ length/2)*math.sin((yaw)))
        y.append(center[1]+(+ width/2)*math.sin((yaw))+(+ length/2)*math.cos((yaw)))
        plt.fill(x, y, facecolor='lightsalmon', edgecolor='orangered', linewidth=3)
        plt.plot(x,y,color)

    def draw_overlaid(self ,center,length, width, yaw, facecolor):
        x = []
        y = []
        x.append(center[0]+(+ width/2)*math.cos((yaw))-(+ length/2)*math.sin((yaw)))
        y.append(center[1]+(+ width/2)*math.sin((yaw))+(+ length/2)*math.cos((yaw)))
        x.append(center[0]+(+ width/2)*math.cos((yaw))-(- length/2)*math.sin((yaw)))
        y.append(center[1]+(+ width/2)*math.sin((yaw))+(- length/2)*math.cos((yaw)))
        x.append(center[0]+(- width/2)*math.cos((yaw))-(- length/2)*math.sin((yaw)))
        y.append(center[1]+(- width/2)*math.sin((yaw))+(- length/2)*math.cos((yaw)))
        x.append(center[0]+(- width/2)*math.cos((yaw))-(+ length/2)*math.sin((yaw)))
        y.append(center[1]+(- width/2)*math.sin((yaw))+(+ length/2)*math.cos((yaw)))
        x.append(center[0]+(+ width/2)*math.cos((yaw))-(+ length/2)*math.sin((yaw)))
        y.append(center[1]+(+ width/2)*math.sin((yaw))+(+ length/2)*math.cos((yaw)))
        plt.fill(x, y, facecolor)
