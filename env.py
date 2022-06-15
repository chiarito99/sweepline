from cProfile import label
from os import O_EXCL

from matplotlib import pyplot as plt
from cubic_spline import Spline2D
from grid_base_sweep import *
import numpy as np

class Env:
    def __init__(self, ox ,oy, resolution =10.0):
        self.ox = ox
        self.oy = oy
        self.resolution =resolution
        px,py = planning(self.ox,self.oy,self.resolution)

        ds = 1.0
        sp = Spline2D(px,py)
        s=np.arange(0,sp.s[-1],ds)

        rx,ry,ryaw,rk = [], [], [], []
        for i_s in s:
            ix,iy  = sp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(sp.calc_yaw(i_s))
            rk.append(sp.calc_curvature(i_s))
        
        self.traj = np.array([rx,ry])

        self.obs = np.array([[0,0],[0,1]])
if __name__ == '__main__':
    # ox = [0.0, 50.0, 50.0, 0.0, 0.0]
    # ox = [0.0, 0.0, 60.0, 60.0, 0.0]
    # resolution = 5

    ox = [200.0, 800.0, 800.0, 200.0, 200.0]
    oy = [200.0, 200.0, 700.0, 700.0, 200.0]
    resolution = 20
    env = Env(ox,oy,resolution)

    plt.figure()
    plt.plot(env.ox,env.oy,'-xk',label= 'range')
    plt.plot(env.traj[0,:],env.traj[1,:],'-b',label = 'reference')
    plt.axis('equal')
    plt.show()

    import scipy.io
    scipy.io.savemat('ref.mat',dict(lm=np.array([ox,oy]),path=env.traj))