from cubic_spline import Spline2D
from grid_based_sweep import *

import numpy as np
import math
import matplotlib.pyplot as plt

class UAV:
    def __init__(self, pos=[0,0,0]):
        # Configuration
        self.pos = np.array(pos)
        self.heading = 0
        self.path = [self.pos]

        # Control parameters
        self.ao = 2.0
        self.bo = 3.0

    def avoid_obstacle(self, obs):
        v = np.zeros(np.size(self.pos))
        for i in range(len(obs)):
            do = math.sqrt((obs[i,0]-self.pos[0])**2 + (obs[i,1]-self.pos[1])**2) - obs[i,2]

            vao = (obs[i,:2]-self.pos[:2])/do          # Velocity avoiding obstacle
            sig1 = -np.sign(vao[0]*math.sin(self.heading)-vao[1]*math.cos(self.heading))
            sig2 = np.sign(vao[0]*math.cos(self.heading)-vao[1]*math.sin(self.heading))
            if sig2 == 0:
                sig2 = 1
            vao = np.array([vao[1]*sig1, vao[0]*sig2, 0])
            fao = 0
            if do <= self.bo:
                fao = self.ao*(1-do/self.bo)
            v = v + fao*vao + [1,1,1]
        return v

    def control_signal(self, obs):
        v2 = self.avoid_obstacle(obs)
        return  v2
    
    def update_position(self, vel, dt=0.02):
        self.pos = self.pos + vel*dt
        self.heading = np.arctan2(vel[1], vel[0])
        self.path.append(self.pos)

if __name__ == "__main__":
    obs = np.array([[5,5,1.5],[10,12,1.5]])

    leader = UAV()
    leader.heading = math.pi/4 

    dt = 0.01
    sim_time = 10
    iter = 0
    while sim_time-iter*dt > 0:
        lvel = leader.control_signal( obs)
        
        leader.update_position(lvel)
        iter += 1

    path = np.array(leader.path)
    plt.figure()
    ax = plt.axes()
    sp = np.arange(0,2*math.pi+math.pi/6,math.pi/6)
    for i in range(len(obs)):
        ax.plot(obs[i,0]+obs[i,2]*np.cos(sp), obs[i,1]+obs[i,2]*np.sin(sp), '-k')
    ax.plot(path[:,0], path[:,1], '-b', label='leader')
    
    ax.axis('equal')
    plt.show()