import plotly.graph_objects as go
from sweep_line_has_bad import getConvexPolygon
import numpy as np


# Get convex polygon
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

x_start = -80
y_start = -80
x_end = 80
y_end = 80

M, Mshifted= getConvexPolygon(n_vertices,polygon_radius,rad_var,ang_var)
K = M.tolist()
x1,y1 = zip(*K)

K1 = [[58.98295314305732, -40.46389776524755], [-19.5748849118947, -78.531936254165], [-62.674712335622026, 24.06481669719506], [-31.09947113556031, 61.08658069805723], [52.20911077098446, 26.412130624396212]]
x2,y2 = zip(*K1)

fig = go.Figure()
fig.add_trace(go.Scatter(x1,y1, fill='tozeroy',
                    mode='none' # override default markers+lines
                    ))
fig.add_trace(go.Scatter(x2,y2, fill='tonexty',
                    mode= 'none'))

fig.show()