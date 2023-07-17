from cmath import nan
from ctypes import sizeof
from operator import le, length_hint
from tkinter import W
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from Plotting3 import Plotting
from env1 import Env1
from sklearn.metrics import jaccard_score
from sweep_line_has_bad_copy import getConvexPolygon,computeWLofCamera,getOpSweep
from intersect_Finding import *

global flag,err1,err2,cons,a,f1,f2,vleader
class LeaderUAV:
    def __init__(self, pos=[0,0]):
        # Configuration
        self.pos = np.array(pos)
        self.heading = 0
        self.path = [self.pos]

        # Control parameters
        self.am = 3.8
        self.bm = 1.0
        self.ao = 3.0
        self.bo = 4.0
        self.do = 1.5


    def move_to_goal(self, goal):
        dm = math.sqrt((goal[0]-self.pos[0])**2 + (goal[1]-self.pos[1])**2)
        vm2g = (goal-self.pos)/dm    # Velocity move to goal
        fm2g = self.am               # Control parameter of vm2g
        if dm <= self.bm:
            fm2g = self.am*dm/self.bm
        return 2.3*fm2g*vm2g

    def avoid_obstacle(self, obs):
        v = np.zeros(np.size(self.pos))
        for i in range(len(obs)):
            do = math.sqrt((obs[i,0]-self.pos[0])**2 + (obs[i,1]-self.pos[1])**2) - obs[i,2] - 0.5
            vao = (obs[i,:2]-self.pos[:2])/do          # Velocity avoiding obstacle
            sig = -np.sign(vao[0]*math.sin(self.heading)-vao[1]*math.cos(self.heading))
            rot = np.array([[0,-sig,0],[sig,0,0],[0,0,1]])
            fao = 0
            vao = np.array([vao[0],vao[1],0])
            if do <= self.bo :
                fao = self.ao*(1-do/self.bo)
            v = v + (fao*rot)@vao
        return v

    def control_signal(self, ref,flag,err1,err2):
        v1 = self.move_to_goal(ref)
        if flag == 1:
            v1= 1/(self.do*max(err1-cons,err2-cons)) * v1
            if max(err1-cons,err2-cons) < self.do:
                v1 =  self.bm*max(err1-cons,err2-cons)/self.am * v1
        # v2 = self.avoid_obstacle(obs)
        return v1
    
    def update_position(self, vel, dt=0.1):
        self.pos = self.pos + vel*dt
        self.heading = (np.arctan2(vel[1], vel[0]) + np.pi) %(2*np.pi)-np.pi
        self.path.append(self.pos)

class FollowerUAV:
    def __init__(self, pos=[0,0], leader=None, delta=[0,0], wp= None):
        # Configuration
        self.pos = np.array(pos)
        # print(self.pos)
        self.wp = wp
        self.heading = 0
        self.angle = []
        self.m = 0
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

    def keep_formation(self, ref,flag,a,f1,vleader,flag_p):
        # print(a)
        if flag_p == 0:
            xr = np.cos(self.leader.heading)*self.delta[0] - np.sin(self.leader.heading)*self.delta[1] + ref[0]
            yr = np.sin(self.leader.heading)*self.delta[0] + np.cos(self.leader.heading)*self.delta[1] + ref[1]
            pr = np.array([xr, yr])
        else:
            xr = ref[0]
            yr = ref[1]
            pr = np.array([xr, yr])
        phi1=0
        # print(self.pos)
        f1 = pr -self.pos
        # print(f1)
        phi1 = np.arccos((f1[0]*a[0]+f1[1]*a[1])/((np.hypot(f1[0],f1[1]))*(np.hypot(a[0],a[1]))))
        if phi1 == nan:
            phi1=0
        dk = math.sqrt((xr-self.pos[0])**2 + (yr-self.pos[1])**2)
        vkf = (pr-self.pos)/dk# Velocity move to goal
        if flag_p == 0:
            for i in range(1,len(self.wp)-1):
                wk = math.sqrt((xr-self.wp[i][0])**2 + (yr-self.wp[i][1])**2)
                if (3 <wk < 8.5) and  (phi1 > np.pi/2 or flag ==2) :
                    vkf = vleader/75
                else:
                    vkf= vkf
        fkf = self.am              # Control parameter of vm2g
        if dk <= self.bm:
            fkf = self.am*dk/self.bm
        return 2.2*fkf*vkf

    def avoid_obstacle(self, obs):
        v = np.zeros(np.size(self.pos))
        for i in range(len(obs)):
            do = math.sqrt((obs[i,0]-self.pos[0])**2 + (obs[i,1]-self.pos[1])**2) - obs[i,2]-0.5
            vao = (obs[i,:2]-self.pos[:2])/do          # Velocity avoiding obstacle
            sig = -np.sign(vao[0]*math.sin(self.heading)-vao[1]*math.cos(self.heading))
            rot = np.array([[0,-sig,0],[sig,0,0],[0,0,1]])
            fao = 0
            vao = np.array([vao[0],vao[1],0])
            if do <= self.bo :
                fao = self.ao*(1-do/self.bo)
            v = v + (fao*rot)@vao
        return v
    
    def control_signal(self, ref,flag_p):
        v1 = self.keep_formation(ref,flag,a,f1,vleader,flag_p)
        # print(flag_p)
        # v2 = 1.2*self.avoid_obstacle(obs)
        # v3 = self.avoid_Robot(rbt_pos)
        # if v2[1] == 0 and flag == 0 :
        if flag == 0:
            v1 = 1.3*v1
        else:
            v1 = v1
        return v1
    
    def update_position(self, vel, dt=0.1):
        self.heading =(np.arctan2(vel[1], vel[0]) + np.pi) %(2*np.pi)-np.pi
        self.angle.append(self.heading)
        self.pos = self.pos + vel*dt
        self.path.append(self.pos)



def plot_obstacles(ox, oy, oz, r):
    z = np.linspace(0, oz, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = r*np.cos(theta_grid) + ox
    y_grid = r*np.sin(theta_grid) + oy
    return x_grid,y_grid,z_grid

def giaiPTBac2(a, b, c):
    delta = b * b - 4 * a * c
    if (delta > 0):
        x1 = (float)((-b + math.sqrt(delta)) / (2 * a));
        x2 = (float)((-b - math.sqrt(delta)) / (2 * a));
    elif (delta == 0):
        x1 = (-b / (2 * a))
    else:
        print("Phương trình vô nghiệm!")
    return x1,x2

def overlap(width,length,percenoverlap):
    indicator = (percenoverlap *width *length)/(2*width *length*2)
    x1,x2 = giaiPTBac2(-1,1,-indicator)
    if x1<x2:
        return x1
    else:
        return x2

def calculateover(width,length,percen):
    width_s = width * percen
    length_s = length * percen
    offset_x = width - width_s
    offset_y = length - length_s
    resolution = 2*(offset_y + length/2)-length_s
    return offset_x,offset_y,resolution

def calculate_polygon_area(vertices):
    n = len(vertices)
    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i+1) % n]  # Lấy đỉnh kế tiếp (hoặc đỉnh đầu tiên nếu i là đỉnh cuối cùng)
        area += x1*y2 - x2*y1
    return abs(area) / 2.0

def duongthang(A,B):
    m = -(A[1]-B[1])/(B[0]-A[0])
    n = (A[1]-B[1])/(B[0]-A[0])*A[0]+A[1]
    return m,n

def getAngle(knee, hip, shoulder):
    ang = math.degrees(math.atan2(shoulder[1]-hip[1], shoulder[0]-hip[0]) - math.atan2(knee[1]-hip[1], knee[0]-hip[0]))
    return ang + 360 if ang < 0 else ang

def checkangle(K,point_angle):
    for i in range(len(K)-1):
        if i == len(K)-2:
            if getAngle(K[i],K[i+1],K[1]) > 180:
                point_angle.append(i+1)
        else:
            if getAngle(K[i],K[i+1],K[i+2]) > 180:
                point_angle.append(i+1)
    return point_angle

def giao(a,b,c,d): #giao của 2 đường thẳng
    x = (d-b)/(a-c)
    y = c*x+d
    return [x,y] 

def checkslide(K,point_angle):#hàm trả về giá trị điểm point non-convex
    # duongthnag y = mx+n
    Kc = K.copy()
    Kc.remove(Kc[point_angle[0]])
    Kc.remove(Kc[point_angle[0]-1])
    Kc.remove(Kc[point_angle[0]+1])
    m1,n1 = duongthang(K[point_angle[0]],K[point_angle[0]-1])
    m2,n2 = duongthang(K[point_angle[0]],K[point_angle[0]+1])
    for i in range(len(Kc)-1):
        if (m1*Kc[i][0]+n1-Kc[i][1])*(m1*Kc[i+1][0]+n1-Kc[i+1][1]) < 0:
            md,nd = duongthang(Kc[i],Kc[i+1])
            K.insert(point_angle[0],giao(m1,n1,md,nd))

def arange(K,point_angle):
    i = 0
    while i < len(point_angle):
        countam = 0
        countduong = 0
        indexam = []
        indexduong = []
        Kc = K.copy()
        Kc.pop()
        Kc.remove(K[point_angle[i]])
        Kc.remove(K[point_angle[i]+1])
        m3,n3 = duongthang(K[point_angle[i]],K[point_angle[i]+1])
        # print(Kc)
        for j in range(len(Kc)):
            if (m3*Kc[j][0]-Kc[j][1]+n3) < 0 :
                countam = countam + 1
                indexam.append(j)
            elif (m3*Kc[j][0]-Kc[j][1]+n3)> 0:
                countduong = countduong + 1
                indexduong.append(j)
        if countam > countduong:
            for i in range(len(indexam)):
                K1.append(Kc[indexam[i]])
            for i in range(len(indexduong)):
                K2.append(Kc[indexduong[i]])
        else:
            for i in range(len(indexam)):
                K2.append(Kc[indexam[i]])
            for i in range(len(indexduong)):
                K1.append(Kc[indexduong[i]])
        i = i+1

if __name__ == "__main__":
    dt = 0.5  # time step
    point_angle = []
    x_start = -80
    y_start = -80
    x_end = 80
    y_end = 80
    K = [[59, -40],[-10,-40],[-20, -79], [-63, 24], [-31, 61], [52, 26]]
    alpha = 0.15 # Góc máy  chiều rộng
    beta = 0.22 # GÓc máy chiều 
    altitude = 20
    percenoverlap = 0.3
    width , length = computeWLofCamera(altitude,alpha,beta)
    xxx = overlap(width,length,percenoverlap)
    offsetx,offsety,resolution = calculateover(width,length,xxx)
    disLF = np.hypot(offsetx,offsety)
    cons = disLF

    flag = 0
    err1 = 0
    err2 = 0
    a = [0]
    f1 = [0]
    f2 = [0]
    path1 = []
    path2 = []

    K.append(K[0])
    point_angle = checkangle(K,point_angle)
    checkslide(K,point_angle)

    ox, oy = zip(*K)
    K1 = []
    K1.append(K[point_angle[0]-1])
    K1.append(K[point_angle[0]])
    K2 = []
    K2.append(K[point_angle[0]+1])
    K2.append(K[point_angle[0]])
    arange(K,point_angle)

    K1.append(K1[0])
    K1.reverse()
    ox1,oy1= zip(*K1)
    a_path = planning(resolution,1,ox1,oy1)
    for i in range(len(a_path[0])):
        path1.append([a_path[0][i],a_path[1][i]])
    path1.insert(0,[x_start,y_start])
    map1 = Env1([x_start,y_start],[x_end,y_end],resolution,path1)

    K2.append(K2[0])
    ox2 ,oy2 = zip(*K2)
    b_path = planning(5,1,ox2,oy2)
    for i in range(len(b_path[0])):
        path2.append([b_path[0][i],b_path[1][i]])
    path2.append(K2[0])
    map2 = Env1([x_start,y_start],[x_end,y_end],5,path2)

    # Formation processing
    leader = LeaderUAV(pos=[x_start,y_start])
    follower1 = FollowerUAV(pos=[x_start,y_start],leader=leader, delta=[-offsetx,-offsety],wp = path2)
    follower2 = FollowerUAV(pos=[x_start,y_start],leader=leader, delta=[-offsetx, offsety],wp = path1)
    x_traj, y_traj = [], []
    min = 7
    flag_point1 = 0
    flag_point2 = 0
    var = 0
    for i in range(1,len(map1.traj[0])):
        ref1 = map1.traj[:,i]
        a = ref1-a
        lvel1 = leader.control_signal(ref1,flag,err1,err2)
        vleader = lvel1
        rbt_pos = np.array([follower1.pos,follower2.pos])
        disLP = leader.pos - map2.traj[:,0]
        if np.hypot(disLP[0],disLP[1]) < min:
            flag_point1 = 1
            # print(min)
        if flag_point1 == 1:
            # print("i:")
            # print(i)
            f1vel = follower1.control_signal(map2.traj[:,var], flag_point1)
            var = var + 1
            if var == len(map2.traj[0]):
                flag_point1 = 0
        else:
            f1vel = follower1.control_signal(ref1,flag_point1)

        f2vel = follower2.control_signal(ref1,flag_point2)
        # # UAV update
        leader.update_position(lvel1)
        a = leader.pos
        # vleader = lvel1
        follower1.update_position(f1vel)
        follower2.update_position(f2vel)
        err1= np.hypot((leader.pos[0]-follower1.pos[0]),(leader.pos[1]-follower1.pos[1]))
        err2 = np.hypot((leader.pos[0]-follower2.pos[0]),(leader.pos[1]-follower2.pos[1]))
        if err1+err2- 2*disLF > 0.7 and err1+err2- 2*disLF < 1.5:
            flag = 1
        elif (abs(leader.heading - follower1.heading) >np.pi/2 or abs(leader.heading - follower2.heading) >np.pi/2) :
            flag = 2
        else:
            flag=0           
    # plot= Plotting("formation")
    # plot.plot_animation(leader.path,follower1.path,follower2.path,ox, oy,x_start,y_start,x_end,y_end,length,width)
    # plt.show()
    
    # Plotting
    plt.figure()
    ax = plt.axes(projection ='rectilinear')
    # ax = plt.axes(projection ='3d')
    # ax.plot(ox1, oy1, '-xk', label='range')
    ax.plot(ox2, oy2, '-xk', label='range')
    ax.plot(ox1, oy1,'-xk',label = 'range')
    # ax.fill(ox1,oy1,facecolor='red')
    # ax.fill(ox2,oy2,facecolor='green')
    ax.plot(map1.traj[0,:], map1.traj[1,:], '-b', label='reference')
    ax.plot(map2.traj[0,:], map2.traj[1,:], '-b', label='reference')
    # ax.plot(a_path[0,:],a_path[1,:],'-b',label='reference')
    # plot obstacle
    # for i in range(len(map.obs)):
    #     Xc, Yc, Zc = plot_obstacles(map.obs[i,0], map.obs[i,1], 1.2*map.altitude, map.obs[i,2])
    #     ax.plot_surface(Xc, Yc, Zc, alpha=0.5)

    leader.path = np.array(leader.path)
    follower1.path = np.array(follower1.path)
    follower2.path = np.array(follower2.path)
    ax.plot(leader.path[:,0], leader.path[:,1], '--r', label='Leader ')
    ax.plot(follower1.path[:,0], follower1.path[:,1], '--c', label='Follower 1 ')
    ax.plot(follower2.path[:,0], follower2.path[:,1], '--g', label='Follower 2 ')

    ax.plot(x_start, y_start, 'ks', label='start')    # start
    ax.plot(x_end, y_end, 'ko', label='end')    # end
    
    ax.set_title("Forest rangers")
    # ax.grid(True)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    # ax.set_zlabel('z [m]')
    ax.legend()
    plt.show()