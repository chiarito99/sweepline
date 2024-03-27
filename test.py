from cubic_spline import Spline2D
from grid_based_sweep import *

import numpy as np
import math
import matplotlib.pyplot as plt

class LeaderUAV:
    def __init__(self, pos=[0,0,0]):
        # Configuration
        self.pos = np.array(pos)
        self.heading = 0
        self.path = [self.pos]

        # Control parameters
        self.am = 5.0
        self.bm = 5.0
        self.ao = 2.0
        self.bo = 3.0


    def move_to_goal(self, goal):
        dm = math.sqrt((goal[0]-self.pos[0])**2 + (goal[1]-self.pos[1])**2 + (goal[2]-self.pos[2])**2)

        vm2g = (goal-self.pos)/dm    # Velocity move to goal
        fm2g = self.am               # Control parameter of vm2g
        if dm <= self.bm:
            fm2g = self.am*dm/self.bm
        return fm2g*vm2g

    def avoid_obstacle(self, obs):
        v = np.zeros(np.size(self.pos))
        for i in range(len(obs)):
            do = math.sqrt((obs[i,0]-self.pos[0])**2 + (obs[i,1]-self.pos[1])**2) - obs[i,2]

            vao = (obs[i,:2]-self.pos[:2])/do          # Velocity avoiding obstacle
            sig = np.sign(math.sin(math.atan2(vao[1]*math.cos(self.heading)-vao[0]*math.sin(self.heading),
                                             vao[0]*math.cos(self.heading)+vao[1]*math.sin(self.heading))))
            vao = np.array([vao[1]*sig, vao[0], 0])
            fao = 0
            if do <= self.bo:
                fao = self.ao*(1-do/self.bo)
            v = v + fao*vao
        return v


    def control_signal(self, ref, obs):
        v1 = self.move_to_goal(ref)
        v2 = self.avoid_obstacle(obs)

        return v1 + v2
    
    def update_position(self, vel, dt=0.1):
        self.pos = self.pos + vel*dt
        self.heading = np.arctan2(vel[1], vel[0])
        self.path.append(self.pos)

class FollowerUAV:
    def __init__(self, pos=[0,0,0], leader=None, delta=[-2,-2]):
        # Configuration
        self.pos = np.array(pos)
        self.heading = 0
        if leader == None:
            self.leader = LeaderUAV()
        else:
            self.leader = leader
        self.delta = delta
        self.path = [self.pos]

        # Control parameters
        self.am = 5.0
        self.bm = 5.0
        self.ao = 3.0
        self.bo = 4.0

    def keep_formation(self, ref):
        # xr = np.cos(self.leader.heading)*self.delta[0] - np.sin(self.leader.heading)*self.delta[1] + ref[0]
        # yr = np.sin(self.leader.heading)*self.delta[0] + np.cos(self.leader.heading)*self.delta[1] + ref[1]
        xr = self.delta[0] + ref[0]
        yr = self.delta[1] + ref[1]
        zr = ref[2]
        pr = np.array([xr, yr, zr])
        
        dk = math.sqrt((xr-self.pos[0])**2 + (yr-self.pos[1])**2 + (zr-self.pos[2])**2)

        vkf = (pr-self.pos)/dk     # Velocity move to goal
        fkf = self.am              # Control parameter of vm2g
        if dk <= self.bm:
            fkf = self.am*dk/self.bm
        return fkf*vkf

    def avoid_obstacle(self, obs):
        v = np.zeros(np.size(self.pos))
        for i in range(len(obs)):
            do = math.sqrt((obs[i,0]-self.pos[0])**2 + (obs[i,1]-self.pos[1])**2) - obs[i,2]

            vao = (obs[i,:2]-self.pos[:2])/do          # Velocity avoiding obstacle
            sig = np.sign(math.sin(math.atan2(vao[1]*math.cos(self.heading)-vao[0]*math.sin(self.heading),
                                             vao[0]*math.cos(self.heading)+vao[1]*math.sin(self.heading))))
            vao = np.array([vao[1]*sig, vao[0], 0])
            fao = 0
            if do <= self.bo:
                fao = self.ao*(1-do/self.bo)
            v = v + fao*vao
        return v

    def control_signal(self, ref, obs):
        v1 = self.keep_formation(ref)
        v2 = self.avoid_obstacle(obs)
        return v1 + v2
    
    def update_position(self, vel, dt=0.1):
        self.pos = self.pos + vel*dt
        self.heading = np.arctan2(vel[1], vel[0])
        self.path.append(self.pos)

if __name__ == "__main__":
    obs = np.array([[5,5,1.5]])
    goal = np.array([10,10,5])

    leader = LeaderUAV()
    follower = FollowerUAV(pos=[1,0,0], leader=leader, delta=[-1,-1])    

    dt = 0.01
    sim_time = 10
    iter = 0
    while sim_time-iter*dt > 0:
        lvel = leader.control_signal(goal, obs)
        fvel = follower.control_signal(goal, obs)
        
        leader.update_position(lvel)
        follower.update_position(fvel)
        iter += 1

    path = np.array(leader.path)
    fpath = np.array(follower.path)
    plt.figure()
    ax = plt.axes()
    sp = np.arange(0,2*math.pi+math.pi/6,math.pi/6)
    for i in range(len(obs)):
        ax.plot(obs[i,0]+obs[i,2]*np.cos(sp), obs[i,1]+obs[i,2]*np.sin(sp), '-k')
    ax.plot(path[:,0], path[:,1], '-b', label='leader')
    ax.plot(fpath[:,0], fpath[:,1], '-r', label='follower')
    ax.axis('equal')
    plt.show()