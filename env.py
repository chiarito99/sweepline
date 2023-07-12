from importlib.util import set_loader
from cubic_spline import Spline2D
from grid_based_sweep import *
import numpy as np
from sweep_line_has_bad_copy import getOpSweep
from sweep_line_has_bad_copy import getBadSweep
from shapely.geometry import LineString
from shapely.ops import unary_union

class Env:
    def __init__(self,K ,x_start,y_start,x_end, y_end,altitude, resolution):
        # The range of the map
        self.K = K
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end
        self.altitude = altitude
        self.resolution= resolution

        
        path = getOpSweep(K, [x_start,y_start], [x_end,y_end],resolution)
        self.point = path
        K.append(K[0])
        ox, oy = zip(*K)
        self.ox = ox
        self.oy = oy
        path = np.array(path)
        px = []
        py = []
        si = path.shape[0] - 1
        for i in range(si):
            for k in range(0,30):
                xk = path[i][0]- k/34*(path[i][0]-path[i+1][0])
                yk = path[i][1]- k/34*(path[i][1]-path[i+1][1])
                px.append(xk)
                py.append(yk)
        ds = 0.5  # [m] distance of each intepolated points
        sp = Spline2D(px, py)
        s = np.arange(0, sp.s[-1], ds)
        self.s=s
        # print(len(s))

        rx, ry, ryaw, rk = [], [], [], []
        for i_s in s:
            ix, iy = sp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(sp.calc_yaw(i_s))
            rk.append(sp.calc_curvature(i_s))

        rz = np.ones(np.size(rx))*self.altitude

        # Desired trajectory
        self.traj = np.array([rx, ry, rz])
        self.ryaw = ryaw
        self.rk=rk

        # Obstacle
        # self.obs = np.array([[20,30,2],
        #                      [-25,10,2],
        #                      [10,20,2]])
        self.obs = []
        # # Obstacle
        # self.obs = np.array([[100,100,0],
        #                      [-100,100,0],
        #                      [90,90,0]])

if __name__ == "__main__":
    # ox = [0.0, 50.0, 50.0, 0.0, 0.0]
    # oy = [0.0, 0.0, 60.0, 60.0, 0.0]
    # resolution = 5
    x_start = -80
    y_start = -80
    x_end = 80
    y_end = 80

    # M, Mshifted= getConvexPolygon(n_vertices,polygon_radius,rad_var,ang_var)
    # K = M.tolist()
    # print(K)
    
    K = [[58.98295314305732, -40.46389776524755], [-19.5748849118947, -78.531936254165], [-62.674712335622026, 24.06481669719506], [-31.09947113556031, 61.08658069805723], [52.20911077098446, 26.412130624396212]]
    # Map and reference path generation
    # ox = [0.0, 50.0, 50.0, 0.0, 0.0]
    # oy = [0.0, 0.0, 60.0, 60.0, 0.0]
    altitude = 10
    overlap = 0.0 
    env = Env(K,x_start,y_start,x_end,y_end,altitude,overlap)
    
    plt.figure()
    plt.plot(env.ox, env.oy, '-xk', label='range')
    plt.plot(env.traj[0,:], env.traj[1,:], '-b', label='reference')
    plt.axis('scaled')
    plt.show()
    #
    # import scipy.io
    # scipy.io.savemat('ref.mat', dict(lm=np.array([ox, oy]),path=env.traj))
