import numpy as np
sig = 1
rot = np.array([[0,-sig,0],
                [sig,0,0],
                [0,0,1]])
vao = np.array([2,3,0])
print(rot @ vao)
