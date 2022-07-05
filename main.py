import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from Plotting import Plotting
from env import Env
from sklearn.metrics import jaccard_score
from sweep_line import getConvexPolygon
import time

class LeaderUAV:
    def __init__(self, pos=[0,0,0]):
        # Configuration
        self.pos = np.array(pos)
        self.heading = 0
        self.path = [self.pos]

        # Control parameters
        self.am = 3.8
        self.bm = 1.0
        self.ao = 3.0
        self.bo = 4.0


    def move_to_goal(self, goal):
        dm = math.sqrt((goal[0]-self.pos[0])**2 + (goal[1]-self.pos[1])**2 + (goal[2]-self.pos[2])**2)

        vm2g = (goal-self.pos)/dm    # Velocity move to goal
        fm2g = self.am               # Control parameter of vm2g
        if dm <= self.bm:
            fm2g = self.am*dm/self.bm
        return 2*fm2g*vm2g

    # def avoid_obstacle(self, obs):
    #     v = np.zeros(np.size(self.pos))
    #     for i in range(len(obs)):
    #         do = math.sqrt((obs[i,0]-self.pos[0])**2 + (obs[i,1]-self.pos[1])**2) - obs[i,2]

    #         vao = (obs[i,:2]-self.pos[:2]) / do         # Velocity avoiding obstacle
    #         sig = np.sign(math.sin(math.atan2(vao[1]*math.cos(self.heading)-vao[0]*math.sin(self.heading),
    #                                          vao[0]*math.cos(self.heading)+vao[1]*math.sin(self.heading))))
    #         vao = np.array([vao[1]*sig, vao[0], 0])
    #         fao = 0
    #         if do <= self.bo + 0.1:
    #             fao = self.ao*(1-do/self.bo)
    #         v = v + fao*vao
    #     return v

    def avoid_obstacle(self,obs):
        v = np.zeros(np.size(self.pos))
        for i in range(len(obs)):
            do = math.sqrt((obs[i,0]-self.pos[0])**2 + (obs[i,1]-self.pos[1])**2) - obs[i,2]

            vao = (obs[i,:2]-self.pos[:2])/do  
            if self.heading > 0:     # Velocity avoiding obstacle
                sig1 = -np.sign(vao[0]*math.sin(self.heading)-vao[1]*math.cos(self.heading))
                sig2 = -np.sign(vao[0]*math.cos(self.heading)-vao[1]*math.sin(self.heading))
            else:
                sig1 = -np.sign(vao[0]*math.sin(self.heading)-vao[1]*math.cos(self.heading))
                sig2 = np.sign(vao[0]*math.cos(self.heading)-vao[1]*math.sin(self.heading))
            # if self.heading > 0 and self.heading <np.pi/2:     # Velocity avoiding obstacle
            #     sig1 = -np.sign(vao[0]*math.sin(self.heading)-vao[1]*math.cos(self.heading))
            #     sig2 = -np.sign(vao[0]*math.cos(self.heading)-vao[1]*math.sin(self.heading))
            # elif self.heading > np.pi/2 and self.heading <np.pi:
            #     sig1 = -np.sign( vao[0]*math.sin(self.heading)+ vao[1]*math.cos(self.heading))
            #     sig2 = np.sign(-vao[0]*math.cos(self.heading)-vao[1]*math.sin(self.heading))
            # elif self.heading > -np.pi/2 and self.heading <0:
            #     sig1 = -np.sign(-vao[0]*math.sin(self.heading)-vao[1]*math.cos(self.heading))
            #     sig2 = -np.sign(vao[0]*math.cos(self.heading)+vao[1]*math.sin(self.heading))
            # else:
            #     sig1 = -np.sign(-vao[0]*math.sin(self.heading)-vao[1]*math.cos(self.heading))
            #     sig2 = np.sign(vao[0]*math.cos(self.heading)- vao[1]*math.sin(self.heading))
            # if sig2 == 0:
            #     sig2 = 1
            vao = np.array([vao[1]*sig1, vao[0]*sig2, 0])
            e=0
            if do <= self.bo:
                e = 1.5*np.exp(-2.9*(do -2.0)/do)
                if abs(e)>3:
                    e= np.sign(e)
            # if do <= self.bo + 0.2:
            #     fao = self.ao*(1-do/self.bo)
            v = v + e*vao
        return v


    def control_signal(self, ref, obs):
        v1 = self.move_to_goal(ref)
        v2 = self.avoid_obstacle(obs)
        # print(v1+v2)
        return v1 + v2
    
    def update_position(self, vel, dt=0.1):
        self.pos = self.pos + vel*dt
        self.heading = (np.arctan2(vel[1], vel[0]) + np.pi) %(2*np.pi)-np.pi
        if self.heading < 0 and self.heading > -0.1:
            self.heading = abs(self.heading)
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
        self.am = 3.3
        self.bm = 1.0
        self.ao = 3.0
        self.bo = 4.0


    def avoid_Robot(self,rbt_pos):
        v = np.zeros(np.size(self.pos))
        for i in range(len(rbt_pos)):
            do = math.sqrt((rbt_pos[i,0]-self.pos[0])**2 + (rbt_pos[i,1]-self.pos[1])**2)
            e=0
            if do !=0 and do<0.1:
                e = -(rbt_pos[i,:2]-self.pos[:2])/do
                vao = np.array([e[0],e[1],0])
                v=v+vao
            else:
                v=v
        return v



    def keep_formation(self, ref):
        xr = np.cos(self.leader.heading)*self.delta[0] - np.sin(self.leader.heading)*self.delta[1] + ref[0]
        yr = np.sin(self.leader.heading)*self.delta[0] + np.cos(self.leader.heading)*self.delta[1] + ref[1]
        # xr = self.delta[0] + ref[0]
        # yr = self.delta[1] + ref[1]
        zr = ref[2]
        pr = np.array([xr, yr, zr])
        
        dk = math.sqrt((xr-self.pos[0])**2 + (yr-self.pos[1])**2 + (zr-self.pos[2])**2)

        vkf = (pr-self.pos)/dk     # Velocity move to goal
        fkf = self.am              # Control parameter of vm2g
        if dk <= self.bm:
            fkf = self.am*dk/self.bm
        return 2.2*fkf*vkf

    # def avoid_obstacle(self, obs):
    #     v = np.zeros(np.size(self.pos))
    #     for i in range(len(obs)):
    #         do = math.sqrt((obs[i,0]-self.pos[0])**2 + (obs[i,1]-self.pos[1])**2) - obs[i,2]

    #         vao = (obs[i,:2]-self.pos[:2])/do          # Velocity avoiding obstacle
    #         sig1 = -np.sign(vao[0]*math.sin(self.heading)-vao[1]*math.cos(self.heading))
    #         sig2 = -np.sign(vao[0]*math.cos(self.heading)-vao[1]*math.sin(self.heading))
    #         if sig2 == 0:
    #             sig2 = 1
    #         vao = np.array([vao[1]*sig1, vao[0]*sig2, 0])
    #         fao = 0
    #         if do <= self.bo + 0.2:
    #             fao = self.ao*(1-do/self.bo)
    #         v = v + fao*vao
    #     return v

    def avoid_obstacle(self,obs):
        v = np.zeros(np.size(self.pos))
        for i in range(len(obs)):
            do = math.sqrt((obs[i,0]-self.pos[0])**2 + (obs[i,1]-self.pos[1])**2) - obs[i,2]
            vao = (obs[i,:2]-self.pos[:2])/do          # Velocity avoiding obstacle
            if self.heading > 0 and self.heading <np.pi/2:     # Velocity avoiding obstacle
                sig1 = -np.sign(vao[0]*math.sin(self.heading)-vao[1]*math.cos(self.heading))
                sig2 = -np.sign(vao[0]*math.cos(self.heading)-vao[1]*math.sin(self.heading))
            elif self.heading > np.pi/2 and self.heading <np.pi:
                sig1 = -np.sign(vao[0]*math.sin(self.heading)+ vao[1]*math.cos(self.heading))
                sig2 = np.sign(-vao[0]*math.cos(self.heading)-vao[1]*math.sin(self.heading))
            elif self.heading > -np.pi/2 and self.heading <0:
                sig1 = -np.sign(-vao[0]*math.sin(self.heading)-vao[1]*math.cos(self.heading))
                sig2 = -np.sign(vao[0]*math.cos(self.heading)+vao[1]*math.sin(self.heading))
            else:
                sig1 = np.sign(-vao[0]*math.sin(self.heading)-vao[1]*math.cos(self.heading))
                sig2 = np.sign(vao[0]*math.cos(self.heading)- vao[1]*math.sin(self.heading))
            
            if sig2 == 0:
                sig2 = 1
            vao = np.array([vao[1]*sig1, vao[0]*sig2, 0])
            e=0
            if do <= self.bo:
                e = 1.5*np.exp(-2.9*(do -2.0)/do)
                if abs(e)>3:
                    e= np.sign(e)
            # if do <= self.bo + 0.2:
            #     fao = self.ao*(1-do/self.bo)
            v = v + e*vao
        return v
    

    def control_signal(self, ref, obs,rbt_pos):
        v1 = self.keep_formation(ref)
        v2 = self.avoid_obstacle(obs)
        v3 = self.avoid_Robot(rbt_pos)
        return v1 + v2+ v3
    
    def update_position(self, vel, dt=0.1):
        self.pos = self.pos + vel*dt
        self.heading = (np.arctan2(vel[1], vel[0]) + np.pi) %(2*np.pi)-np.pi
        self.path.append(self.pos)



def plot_obstacles(ox, oy, oz, r):
    z = np.linspace(0, oz, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = r*np.cos(theta_grid) + ox
    y_grid = r*np.sin(theta_grid) + oy
    return x_grid,y_grid,z_grid


if __name__ == "__main__":
    dt = 0.01  # time step
    n_vertices = 5
    polygon_radius = 40
    rad_var = 1
    ang_var = 1
    dx = 2
    transl_spd = 10
    rot_spd = np.pi/4

    samplingx0 = -80
    samplingx1 = 80
    samplingy0 = -80
    samplingy1 = 80

    x_start = (samplingx1- samplingx0) * np.random.rand() + samplingx0
    y_start = (samplingy1- samplingy0) * np.random.rand() + samplingy0
    x_end = (samplingx1- samplingx0) * np.random.rand() + samplingx0
    y_end = (samplingy1- samplingy0) * np.random.rand() + samplingy0

    M, Mshifted= getConvexPolygon(n_vertices,polygon_radius,rad_var,ang_var)
    K = M.tolist()
    # Map and reference path generation
    # ox = [0.0, 50.0, 50.0, 0.0, 0.0]
    # oy = [0.0, 0.0, 60.0, 60.0, 0.0]
    resolution = 15
    map = Env(K,x_start,y_start,x_end,y_end,resolution)

    K.append(K[0])
    ox, oy = zip(*K)
    
    # Formation processing
    leader = LeaderUAV(pos=[x_start,y_start,0])
    follower1 = FollowerUAV(pos=[x_start+5,y_start+5,0],leader=leader, delta=[-5,-5])
    follower2 = FollowerUAV(pos=[x_start-5,y_start-5,0],leader=leader, delta=[-5, 5])

    x_traj, y_traj = [], []
    for i in range(len(map.traj[0])):
        ref = map.traj[:,i]
        x_traj.append(ref[0])
        # print(x_traj)
        y_traj.append(ref[1])
        # print()
        # Check if the traj is colision
        is_colision = False
        for i in range(len(map.obs)):
            do = math.sqrt((ref[0]-map.obs[i,0])**2 + (ref[1]-map.obs[i,1])**2)
            bias = 2.0
            if do <= map.obs[i,2] + bias:
                is_colision = True
                break
        if is_colision:
            continue
        rbt_pos = np.array([follower1.pos,follower2.pos])
        # UAV processing
        lvel = leader.control_signal(ref, map.obs)
        f1vel = follower1.control_signal(ref, map.obs,rbt_pos)
        f2vel = follower2.control_signal(ref, map.obs,rbt_pos)

        # UAV update
        leader.update_position(lvel)
        follower1.update_position(f1vel)
        follower2.update_position(f2vel)

    # print(leader.path)
    plot= Plotting("formation")
    plot.plot_animation(leader.path,follower1.path,follower2.path,ox, oy,x_start,y_start,x_end,y_end)
    plt.show()
    
    # Plotting
    plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot(map.ox, map.oy, np.ones(len(map.ox))*map.altitude, '-xk', label='range')
    ax.plot(map.traj[0,:], map.traj[1,:], map.traj[2,:], '-b', label='reference')

    # plot obstacle
    for i in range(len(map.obs)):
        Xc, Yc, Zc = plot_obstacles(map.obs[i,0], map.obs[i,1], 1.2*map.altitude, map.obs[i,2])
        ax.plot_surface(Xc, Yc, Zc, alpha=0.5)

    leader.path = np.array(leader.path)
    follower1.path = np.array(follower1.path)
    follower2.path = np.array(follower2.path)
    # follower2.path = np.array(follower2.path)
    ax.plot(leader.path[:,0], leader.path[:,1], leader.path[:,2], '--r', label='leader UAV')
    ax.plot(follower1.path[:,0], follower1.path[:,1], follower1.path[:,2], '--c', label='follower 1 UAV')
    ax.plot(follower2.path[:,0], follower2.path[:,1], follower2.path[:,2], '--g', label='follower 2 UAV')
        
    ax.plot(map.traj[0,0], map.traj[1,0], map.traj[2,0], 'ks', label='start')    # start
    ax.plot(map.traj[0,-1], map.traj[1,-1], map.traj[2,-1], 'ko', label='end')    # end
    

    ax.set_title("Forest rangers")
    ax.grid(True)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.legend()
    plt.show()
    
