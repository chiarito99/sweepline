from cubic_spline import Spline2D
from grid_based_sweep import *
import numpy as np
from sweep_line import getOpSweep
from shapely.geometry import LineString
from shapely.ops import unary_union

class Env:
    def __init__(self,K ,x_start,y_start,x_end, y_end, resolution=20):
        # The range of the map
        self.K = K
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end
        path = getOpSweep(K, [x_start,y_start], [x_end,y_end],resolution)

        path = np.array(path)
        px = []
        py = []
        si = path.shape[0] - 1
        for i in range(si):
            for k in range(0,100):
                xk = path[i][0]- k/100*(path[i][0]-path[i+1][0])
                yk = path[i][1]- k/100*(path[i][1]-path[i+1][1])
                px.append(xk)
                py.append(yk)

        ds = 0.5    # [m] distance of each intepolated points
        sp = Spline2D(px, py)
        s = np.arange(0, sp.s[-1], ds)

        rx, ry, ryaw, rk = [], [], [], []
        for i_s in s:
            ix, iy = sp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(sp.calc_yaw(i_s))
            rk.append(sp.calc_curvature(i_s))

        self.altitude = 5
        rz = np.ones(np.size(rx))*self.altitude

        # Desired trajectory
        self.traj = np.array([rx, ry, rz])

        # # Obstacle
        # self.obs = np.array([[20,30,2],
        #                      [30,40,2],
        #                      [10,20,2]])
        
        # Obstacle
        self.obs = np.array([[0,0,0],
                             [0,0,0],
                             [0,0,0]])

if __name__ == "__main__":
    # ox = [0.0, 50.0, 50.0, 0.0, 0.0]
    # oy = [0.0, 0.0, 60.0, 60.0, 0.0]
    # resolution = 5
    ox = [200.0, 800.0, 800.0, 200.0, 200.0]
    oy = [200.0, 200.0, 700.0, 700.0, 200.0]
    resolution = 30
    env = Env(ox, oy, resolution)

    plt.figure()
    plt.plot(env.ox, env.oy, '-xk', label='range')
    plt.plot(env.traj[0,:], env.traj[1,:], '-b', label='reference')
    plt.axis('scaled')
    plt.show()
    #
    # import scipy.io
    # scipy.io.savemat('ref.mat', dict(lm=np.array([ox, oy]),path=env.traj))
