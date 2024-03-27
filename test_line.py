from audioop import mul
from turtle import color
import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import math


def increment(i,n):
    i_next = i%n + 1
    if i == n:
        i_next = 0
    return i_next;

def clockWiseDist(a,b):
    if a<0:
        if b<0:
            if(a<b):
                angle = 2*math.pi + (a-b)
            else:
                angle = a - b
        else:
            a = 2*math.pi + a
            angle = a-b
        
    else:
        if b<0:
            angle = a - b
        else:
            if(a<b):
                angle = 2*math.pi-(a-b)
            else:
                angle = a-b
    return angle

def angleAP(P,i,j):
    n = len(P)
    i_next = increment(i,n-1)
    y =  P[i_next][1] - P[i][1]
    x = P[i_next][0] - P[i][0]
    alpha_i = math.atan2(y,x)

    j_next = increment(j,n-1)
    y = P[j_next][1] - P[j][1]
    x = P[j_next][0] - P[j][0]
    alpha_j = math.atan2(y,x)
    angle = clockWiseDist(alpha_i, alpha_j)
    return angle;s

def isAConvexPolygon(M):
    m = M.shape[0]
    sum = 0
    isConvex = True
    for i in range(0,m):
        j = increment(i,m-1)
        angle_i = np.pi - angleAP(M, i,j)
        if (angle_i > np.pi) or (angle_i < 0):
            isConvex = False
    return isConvex;


def getPolygon(numVert, radius, radVar, angVar):
    x = np.zeros((numVert,1))
    y = np.zeros((numVert,1))
    circleAng = 2*np.pi
    angleSeparation = circleAng/(numVert)
    angleMatrix = np.arange(0,circleAng,angleSeparation)
    for k in range(0,numVert):
        x[k] = (radius + radius*np.random.rand()*radVar) * np.cos(angleMatrix[k] + angleSeparation*np.random.rand()*angVar)
        y[k] = (radius + radius*np.random.rand()*radVar) * np.sin(angleMatrix[k] + angleSeparation*np.random.rand()*angVar)

    Polygon_vertex = np.concatenate((x,y),1)
    Polygon_vertex = np.flipud(Polygon_vertex)
    shifted_polygon_vertex = np.roll(Polygon_vertex, -1,0)

    return Polygon_vertex, shifted_polygon_vertex;

def getConvexPolygon(numVert, radius, radVar, angVar):
    isConvex = False
    
    while not(isConvex):
        Polygon_vertex, shifted_polygon_vertex = getPolygon(numVert, radius, radVar, angVar)
        isConvex = isAConvexPolygon(Polygon_vertex)
    return Polygon_vertex, shifted_polygon_vertex

n_vertices = 5
polygon_radius = 100
rad_var = 1
ang_var = 1
dx = 20
transl_spd = 10
rot_spd = np.pi/4

samplingx0 = -300
samplingx1 = 300
samplingy0 = -300
samplingy1 = 300

x_start = (samplingx1- samplingx0) * np.random.rand() + samplingx0
y_start = (samplingy1- samplingy0) * np.random.rand() + samplingy0
x_end = (samplingx1- samplingx0) * np.random.rand() + samplingx0
y_end = (samplingy1- samplingy0) * np.random.rand() + samplingy0

M, Mshifted= getConvexPolygon(n_vertices,polygon_radius,rad_var,ang_var)
K = M.tolist()

coord = [[146.76306045999095, -2.0473334316286795], [-60.526906172934744, -109.76533421120082], [-132.6429327946019, 20.193654150405926], [-14.865406821485804, 101.09338703482929], [78.12772846254278, 117.07811942387043]]
coord = np.array(coord)
# si = coord.shape[0] - 1
# multipoint = []
# for i in range(si):
    
#     line = LineString((coord[i],coord[i+1]))

#     distance_delta = -5
#     distances = np.arange(0, line.length, distance_delta)
#     # or alternatively without NumPy:
#     # points_count = int(line.length // distance_delta) + 1
#     # distances = (distance_delta * i for i in range(points_count))
#     points = [line.interpolate(distance) for distance in distances] + [line.boundary[1]]
#     multi_point = unary_union(points)  # or new_line = LineString(points)
#     multi_point = np.array(multi_point)
#     multipoint= np.append(multipoint,multi_point)
    
# print(multipoint)
xPath = []
yPath = []
convec = 10
print(coord[0][0])
si = coord.shape[0] - 1
for i in range(si):
    for k in range(0,100):
        xk = coord[i][0]- k/100*(coord[i][0]-coord[i+1][0])
        yk = coord[i][1]- k/100*(coord[i][1]-coord[i+1][1])
        xPath.append(xk)
        yPath.append(yk)

xs, ys = zip(*coord)
plt.figure()
plt.plot(xPath,yPath, color = "blue")
# plt.plot(xs,ys,color ="red") 
plt.axis('scaled')
plt.show() 