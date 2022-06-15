from cProfile import label
from calendar import leapdays
from select import select
from matplotlib import projections
from matplotlib.cbook import delete_masked_points
import numpy as np
import math
import matplotlib.pyplot as plt

from env import Env

show_animation = True


class Leader:
    def __init__(self, pos=[0, 0]):
        self.pos = np.array(pos)
        self.heading = 0
        self.path = [self.pos]

        # control paramaters
        self.am = 5.0
        self.bm = 1.0
        self.a0 = 2.0
        self.b0 = 3.0

    def move_to_goal(self, goal):
        diss = math.sqrt((goal[0] - self.pos[0]) ** 2 + (goal[0] - self.pos[0]) ** 2)
        vm2g = (goal - self.pos) / diss
        fm2g = self.am
        if diss < 0.3:
            fm2g = self.am * diss / self.bm
        return fm2g * vm2g

    def avoid_obs(self, obs):
        v = np.zeros(np.size(self.pos))
        for i in range(len(obs)):
            do = math.sqrt((obs[i, 0] - self.pos[0]) ** 2 + (obs[i, 1] - self.pos[1]) ** 2)
            vao = (obs[i, :2] - self.pos[:2]) / do
            sig = np.sign(math.sin(math.atan2(vao[1] * math.cos(self.heading) - vao[0] * math.sin(self.heading),
                                              vao[0] * math.cos(self.heading) + vao[1] * math.sin(self.heading))))

            vao = np.array([vao[1] * sig, vao[0]])
            fao = 0
            if do <= self.b0:
                fao = self.a0 * (1 - do / self.b0)
            v = v + fao * vao
        return v

    def control_signal(self, ref, obs):
        v1 = self.move_to_goal(ref)
        v2 = self.avoid_obs(obs)
        return v1 + v2

    def update_pos(self, vel, dt=0.1):
        self.heading = np.arctan2(vel[1], vel[0])
        self.pos = self.pos + vel * dt
        self.path.append(self.pos)


class Follower:
    def __init__(self, pos=[0, 0], leader=None, delta=[0, 0]):
        self.pos = np.array(pos)
        self.heading = 0
        if leader == None:
            self.leader = Leader()
        else:
            self.leader = leader
        self.delta = delta
        self.path = [self.pos]

        # control paramaters
        self.am = 5.0
        self.bm = 0.5
        self.a0 = 1.0
        self.b0 = 1.0

    def keep_fomation(self, ref):
        xr = self.delta[0] + ref[0]
        yr = self.delta[1] + ref[1]
        pr = np.array([xr, yr])

        dk = math.sqrt((xr - self.pos[0]) ** 2 + (yr - self.pos[1]) ** 2)
        vkf = (pr - self.pos) / dk
        fkf = self.am
        if dk <= self.bm:
            fkf = self.am * dk / self.bm
        return 2 * fkf * vkf

    def avoid_obs(self, obs):
        v = np.zeros(np.size(self.pos))
        for i in range(len(obs)):
            do = math.sqrt((obs[i, 0] - self.pos[0]) ** 2 + (obs[i, 1] - self.pos[1]) ** 2)
            vao = (obs[i, :2] - self.pos[:2]) / do
            sig = np.sign(math.sin(math.atan2(vao[1] * math.cos(self.heading) - vao[0] * math.sin(self.heading),
                                              vao[0] * math.cos(self.heading) + vao[1] * math.sin(self.heading))))

            vao = np.array([vao[1] * sig, vao[0]])
            fao = 0
            if do <= self.b0:
                fao = self.a0 * (1 - do / self.b0)
            v = v + fao * vao
        return v

    def control_signal(self, ref, obs):
        v1 = self.keep_fomation(ref)
        v2 = self.avoid_obs(obs)
        return v1 + v2

    def update_pos(self, vel, dt=0.1):
        self.pos = self.pos + vel * dt
        self.heading = np.arctan2(vel[1], vel[0])
        self.path.append(self.pos)


def plot_obs(ox, oy, r):
    theta_grip = 0
    xgrip = r * np.cos(theta_grip) + ox
    ygrip = r * np.cos(theta_grip) + oy
    return xgrip, ygrip


def plot_vehicle(x, y, theta, x_traj, y_traj):
    p1_i = np.array([0.5, 0, 1]).T
    p2_i = np.array([-0.5, 0.25, 1]).T
    p3_i = np.array([-0.5, -0.25, 1]).T

    T = transformation_matrix(x, y, theta)
    p1 = np.matmul(T, p1_i)
    p2 = np.matmul(T, p2_i)
    p3 = np.matmul(T, p3_i)

    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
    plt.plot([p2[0], p3[0]], [p2[1], p3[1]], 'k-')
    plt.plot([p3[0], p1[0]], [p3[1], p1[1]], 'k-')

    plt.plot(x_traj, y_traj, 'r--')

    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    plt.pause(dt)


def transformation_matrix(x, y, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0, 0, 1]
    ])


if __name__ == "__main__":
    dt = 0.1
    ox = [0.0, 50.0, 50.0, 0.0, 0.0]
    oy = [0.0, 0.0, 70.0, 60.0, 0.0]
    resolution = 6
    map = Env(ox, oy, resolution)

    leader = Leader(pos=[45, 5])
    follower1 = Follower(pos=[44, 3], leader=leader, delta=[-1, -1])
    follower2 = Follower(pos=[43, 1], leader=leader, delta=[-2, -2])
    follower3 = Follower(pos=[60, 3], leader=leader, delta=[1, -1])
    follower4 = Follower(pos=[47, 1], leader=leader, delta=[2, -2])

    x_traj, y_traj = [], []

    for i in range(len(map.traj[0])):
        ref = map.traj[:, i]
        x_traj.append(ref[0])
        y_traj.append(ref[1])

        is_colision = False
        for i in range(len(map.obs)):
            do = math.sqrt((ref[0] - map.obs[i, 0]) ** 2 + (ref[1] - map.obs[i, 1]) ** 2)
            bias = 2
            if do <= bias:
                is_colision = True
                break
        if is_colision:
            continue

        lvel = leader.control_signal(ref, map.obs)
        f1vel = follower1.control_signal(ref, map.obs)
        f2vel = follower2.control_signal(ref, map.obs)
        f3vel = follower3.control_signal(ref, map.obs)
        f4vel = follower4.control_signal(ref, map.obs)

        leader.update_pos(lvel)
        follower1.update_pos(f1vel)
        follower2.update_pos(f2vel)
        follower3.update_pos(f3vel)
        follower4.update_pos(f4vel)

        if show_animation:  # pragma: no cover
            plt.cla()
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            # plt.arrow(x_start, y_start, np.cos(theta_start),
            #           np.sin(theta_start), color='r', width=0.1)
            # plt.arrow(x_goal, y_goal, np.cos(theta_goal),
            #           np.sin(theta_goal), color='g', width=0.1)
            plot_vehicle(leader.pos[0], leader.pos[1], leader.heading, x_traj, y_traj)
            plot_vehicle(follower1.pos[0], follower1.pos[1], follower1.heading, x_traj, y_traj)
            plot_vehicle(follower2.pos[0], follower2.pos[1], follower2.heading, x_traj, y_traj)
            plot_vehicle(follower3.pos[0], follower3.pos[1], follower3.heading, x_traj, y_traj)
            plot_vehicle(follower4.pos[0], follower4.pos[1], follower4.heading, x_traj, y_traj)

    plt.cla()
    ax = plt.axes(projection='rectilinear')
    ax.plot(map.ox, map.oy, np.ones(len(map.ox)), label='range')
    ax.plot(map.traj[0, :], map.traj[1, :], '-b', label='reference')
    # Zc=2
    # for i in range (len(map.obs)):
    #     Xc,Yc = plot_obs(map.obs[i,0],map.obs[i,1],1.2*map.altitude)
    #     ax.plot(Xc,Yc)

    leader.path = np.array(leader.path)
    follower1.path = np.array(follower1.path)
    follower2.path = np.array(follower2.path)
    follower3.path = np.array(follower3.path)
    follower4.path = np.array(follower4.path)

    # if show_animation:  # pragma: no cover
    #     plot_vehicle(x, y, theta, x_traj, y_traj)

    ax.plot(leader.path[:, 0], leader.path[:, 1], '--r', label='Leader')
    ax.plot(follower1.path[:, 0], follower1.path[:, 1], '--c', label='follower1')
    ax.plot(follower2.path[:, 0], follower2.path[:, 1], '--c', label='follower2')
    ax.plot(follower3.path[:, 0], follower3.path[:, 1], '--c', label='follower3')
    ax.plot(follower4.path[:, 0], follower4.path[:, 1], '--c', label='follower4')

    ax.plot(map.traj[0, 0], map.traj[1, 0], 'ks', label='start')
    ax.plot(map.traj[0, -1], map.traj[1, -1], 'ko', label='end')
    ax.set_title('Forest rangers')
    ax.grid(True)
    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.legend()
    plt.show()
