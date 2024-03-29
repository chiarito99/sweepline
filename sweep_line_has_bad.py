from re import I
from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import math

def distance(A,B):
    dist = np.sqrt((A[0]-B[0])*(A[0]-B[0]) + (A[1]-B[1])*(A[1]-B[1]))
    return dist

def checkBetween(A,B,C):
    # C between AB
    dist_AC = distance(A,C)
    dist_BC = distance(B,C)
    dist_AB = distance(A,B)
    epsilon = 0.00000000001

    if (dist_AC + dist_BC - dist_AB) <= epsilon:
        return True
    return False

def getClockwiseAngle(vector1_,vector2_):
    Dir_C_to_A = np.arctan2(vector1_[1], vector1_[0])
    Dir_C_to_B = np.arctan2(vector2_[1], vector2_[0])
    angle = Dir_C_to_A - Dir_C_to_B

    if(angle < 0): angle += 2*np.pi
    return angle


def findIntersection(start_point,base, line2):

    if((base[1][0] == base[0][0]) and (line2[1][0] == line2[0][0])):
        intersection = []

    elif(base[1][0] == base[0][0]):
        m2 = (line2[1][1] - line2[0][1])/(line2[1][0] - line2[0][0])
        b2 = line2[0][1] - m2*line2[0][0]

        x = start_point[0]
        y = m2*x + b2
        intersection = [x,y]

    elif(line2[1][0] == line2[0][0]):
        m1 = (base[1][1] - base[0][1])/(base[1][0] - base[0][0])
        b1 = start_point[1] - m1*start_point[0]
        x = line2[1][0]
        y = m1*x + b1
        intersection = [x,y]
    else:
        m1 = (base[1][1] - base[0][1])/(base[1][0] - base[0][0])
        m2 = (line2[1][1] - line2[0][1])/(line2[1][0] - line2[0][0])
        b1 = start_point[1] - m1*start_point[0]
        b2 = line2[0][1] - m2*line2[0][0]
        m1 = -m1
        m2 = -m2

        if(m1 == m2):
            intersection = []
        elif(m2 == 0):
            y  = (m1*b2 - m2*b1)/(m1 - m2)
            x  = (b1 - y)/m1
            intersection = [x,y]
        else:
            y  = (m1*b2 - m2*b1)/(m1 - m2)
            x  = (b2 - y)/m2
            intersection = [x,y]
    return intersection

def nextVector(i,V, direction):
    print(i)
    if(i == len(V) - 1):
        a = V[i]
        b = V[0]
        if(direction==0):
            next = [b[0] - a[0], b[1] - a[1]]
        else:
            next = [a[0] - b[0], a[1] - b[1]]
    else:
        a = V[i]
        b = V[i+1]
        if(direction==0):
            next = [b[0] - a[0], b[1] - a[1]]
        else:
            next = [a[0] - b[0], a[1] - b[1]]
    return next

def findAntipodalPoints(V):
    i = 0
    j = 1
    A = []
    while(getClockwiseAngle(nextVector(i,V,0),nextVector(j,V,0)) < np.pi):
        j = j + 1
    antipodalPair = (i,j)
    A.append(antipodalPair)
    antipodalPair = (i+1,j)
    A.append(antipodalPair)
    
    current = i
    while(j != len(V)-1):
        if(current == i):
            if(getClockwiseAngle(nextVector(current,V,0),nextVector(current+1,V,0)) <= getClockwiseAngle(nextVector(current,V,1),nextVector(j,V,0))):
                i = i + 1
                current = i
                antipodalPair = (i+1,j)
                A.append(antipodalPair)
            else:
                i = i + 1
                current = j
                antipodalPair = (i,j+1)
                A.append(antipodalPair)
        else:
            if(getClockwiseAngle(nextVector(current,V,0),nextVector(current+1,V,0)) <= getClockwiseAngle(nextVector(current,V,1),nextVector(i,V,0))):
                j = j + 1
                current = j
                if(j < len(V) - 1):
                    antipodalPair = (i,j+1)
                    A.append(antipodalPair)
                else:
                    antipodalPair = (i,0)
                    A.append(antipodalPair)
            else:
                j = j + 1
                current = i       
                antipodalPair = (i+1,j)
                A.append(antipodalPair)
    return A

def getSweep(starting_point, base, V):
    print("base:")
    print(base)
    sweepIntersection = []
    for i in range(len(V)):
        if(i==len(V)-1):
            line = [V[i],V[0]]
            if not (base==line):
                intersection = findIntersection(starting_point,base,line)
                if (intersection) and (checkBetween(line[0],line[1],intersection)):
                    sweepIntersection.append(intersection)
        else:
            line = [V[i],V[i+1]]
            if not (base==line):
                intersection = findIntersection(starting_point,base,line)
                if (intersection) and (checkBetween(line[0],line[1],intersection)):
                    sweepIntersection.append(intersection)
    if(distance(starting_point,sweepIntersection[0]) > distance(starting_point,sweepIntersection[1])):
        return sweepIntersection[0]
    else:
        return sweepIntersection[1]


def checkSide(A,B,line):
    # True = same side
    if(line[0][0] == line[1][0]):
        if((A[0] < line[0][0]) and (B[0] < line[0][0])) or ((A[0] < line[0][0]) and (B[0] < line[0][0])):
            return True
        else:
            return False
        
    else:
        m = (line[1][1] - line[0][1])/(line[1][0] - line[0][0])
        b = line[0][1] - m*line[0][0]
        m = -m
        b = -b
        if((m*A[0] + A[1] + b)*(m*B[0] + B[1] + b) > 0):
            return True
        else:
            return False

def distPoint2line(point, line):
    if(line[0][0] == line[1][0]):
        dist = abs(line[0][0] - point[0])
    else:
        m = (line[1][1] - line[0][1])/(line[1][0] - line[0][0])
        b = line[0][1] - m*line[0][0]
        m = -m
        b = -b
        dist = abs(m*point[0] + point[1] + b)/np.sqrt(np.power(m,2) + 1)
    return dist

def getUp(start_point,polygon_end, dist , base):
    if(base[0][0] == base[1][0]):
        x1 = start_point[0] + dist
        x2 = start_point[0] - dist
        y1 = start_point[1]
        y2 = start_point[1]
        point1 = [x1,y1]
        point2 = [x2,y2]


        if(distance(point1,polygon_end) < distance(point2,polygon_end)):
            return point1
        else:
            return point2

    elif(base[1][1] == base[0][1]):
        x1 = start_point[0]
        x2 = start_point[0]
        y1 = start_point[1] + dist
        y2 = start_point[1] - dist
        point1 = [x1,y1]
        point2 = [x2,y2]

        if(distance(point1,polygon_end) < distance(point2,polygon_end)):
            return point1
        else:
            return point2
    else:
        m = (base[1][1] - base[0][1])/(base[1][0] - base[0][0])
        b = base[0][1] - m*base[0][0]
        a = np.sqrt(np.power(dist, 2)/(1+1/np.power(m,2)))
        x1 = start_point[0] + a
        x2 = start_point[0] - a
        y1 = start_point[1] - (1/m)*(x1 - start_point[0])
        y2 = start_point[1] - (1/m)*(x2 - start_point[0])
        point1 = [x1,y1]
        point2 = [x2,y2]

        if(distance(point1,polygon_end) < distance(point2,polygon_end)):
            return point1
        else:
            return point2

def getPath(base, polygon_end,offset, V, direction):
    # direction = 0 -> clockwise
    path = []
    point1 = getUp(base[0], polygon_end ,offset/2,base)
    point2 = getUp(base[1], polygon_end, offset/2,base)
    
    basePointOffset1 = getSweep(point2,base,V)
    basePointOffset2 = getSweep(point1,base,V)


    if(direction == 0):
        path.append(basePointOffset1)
        path.append(basePointOffset2)
    else:
        path.append(basePointOffset2)
        path.append(basePointOffset1)
    i = 0
    print(distPoint2line(polygon_end,[path[-1],path[-2]]))
    print(offset)
    while((distPoint2line(polygon_end,[path[-1],path[-2]]) > offset and i == 0) or (distPoint2line(polygon_end,[path[-2],path[-3]]) > offset and i == 1)):
        if(i == 0):
            next = getUp(path[-1], polygon_end, offset, base)
            path.append(next)
        if(i == 1):
            next = getSweep(path[-1],base,V)
            path.append(next)
        i = i + 1
        if(i > 1):
            i = 0
    # print(path)
    return path

def bestPath(antipodal_pair, offset, V):
    path = []
    if (getClockwiseAngle(nextVector(antipodal_pair[1],V,0),nextVector(antipodal_pair[0],V,1)) - np.pi < 0):
        b = antipodal_pair[1]
        a = antipodal_pair[0]
    else:
        b = antipodal_pair[0]
        a = antipodal_pair[1]

    phi = getClockwiseAngle(nextVector(b,V,0),nextVector(a,V,0)) - np.pi
    gammab = getClockwiseAngle(nextVector(b-1,V,0),nextVector(b,V,0))
    gammaa = getClockwiseAngle(nextVector(a-1,V,0),nextVector(a,V,0)) - phi

    if(gammab < gammaa):
        b2 = b-1
        a2 = a
    else:
        b2  = a-1
        a2 = b
    if(b+1 <= len(V)-1):
        if(distPoint2line(V[a],[V[b],V[b+1]]) < distPoint2line(V[a2],[V[b2],V[b2+1]])):
            base = [V[b],V[b+1]]
            polygon_end = V[a]
            direction = 0
            path = getPath(base, polygon_end, offset, V,direction)
        else:
            base = [V[b2],V[b2+1]]
            polygon_end = V[a2]
            direction = 1
            path = getPath(base, polygon_end, offset, V,direction)     
    else:
        if(distPoint2line(V[a],[V[b],V[0]]) < distPoint2line(V[a2],[V[b2],V[b2+1]])):
            base = [V[b],V[0]]
            polygon_end = V[a]
            direction = 0
            path = getPath(base, polygon_end, offset, V,direction)
        else:
            base = [V[b2],V[b2+1]]
            polygon_end = V[a2]
            direction = 1
            path = getPath(base, polygon_end, offset, V,direction)

    return path

def Cost(path):
    total_distance = 0
    for i in range(len(path)-1):
        total_distance = total_distance + distance(path[i], path[i+1])
    return total_distance

def flightLine(path):
    return len(path) - 1

def getOpSweep(V, starting_point, stopping_point, offset):

    A = findAntipodalPoints(V)
    c = np.inf

    for i in range(len(A)):
        p = bestPath(A[i], offset, V)
        p1 = p[::1]
        p1.insert(0,starting_point)
        p1.append(stopping_point)
        p2 = p[::-1]
        p2.insert(0,starting_point)
        p2.append(stopping_point)


        if(Cost(p1) < Cost(p2)):
            tau_aux = p1
        else:
            tau_aux = p2
        
        if(Cost(tau_aux) < c):
            tau = tau_aux
            c = Cost(tau)
    return tau

def worstPath(antipodal_pair, offset, V):
    path = []
    if (getClockwiseAngle(nextVector(antipodal_pair[1],V,0),nextVector(antipodal_pair[0],V,1)) - np.pi < 0):
        b = antipodal_pair[1]
        a = antipodal_pair[0]
    else:
        b = antipodal_pair[0]
        a = antipodal_pair[1]

    phi = getClockwiseAngle(nextVector(b,V,0),nextVector(a,V,0)) - np.pi
    gammab = getClockwiseAngle(nextVector(b-1,V,0),nextVector(b,V,0))
    gammaa = getClockwiseAngle(nextVector(a-1,V,0),nextVector(a,V,0)) - phi

    if(gammab < gammaa):
        b2 = b-1
        a2 = a
    else:
        b2  = a-1
        a2 = b
    if(b+1 <= len(V)-1):
        if(distPoint2line(V[a],[V[b],V[b+1]]) > distPoint2line(V[a2],[V[b2],V[b2+1]])):
            base = [V[b],V[b+1]]
            polygon_end = V[a]
            direction = 0
            path = getPath(base, polygon_end, offset, V,direction)
        else:
            base = [V[b2],V[b2+1]]
            polygon_end = V[a2]
            direction = 1
            path = getPath(base, polygon_end, offset, V,direction)     
    else:
        if(distPoint2line(V[a],[V[b],V[0]]) > distPoint2line(V[a2],[V[b2],V[b2+1]])):
            base = [V[b],V[0]]
            polygon_end = V[a]
            direction = 0
            path = getPath(base, polygon_end, offset, V,direction)
        else:
            base = [V[b2],V[b2+1]]
            polygon_end = V[a2]
            direction = 1
            path = getPath(base, polygon_end, offset, V,direction)

    return path


def getBadSweep(V, starting_point, stopping_point, offset):

    A = findAntipodalPoints(V)
    c = 0

    for i in range(len(A)):
        p = worstPath(A[i], offset, V)

        if(Cost(p) > c):

            c = Cost(p)
            tau_aux = p
            
    p1 = tau_aux[::1]
    p1.insert(0,starting_point)
    p1.append(stopping_point)
    p2 = tau_aux[::-1]
    p2.insert(0,starting_point)
    p2.append(stopping_point)

    if(Cost(p1) < Cost(p2)):
        tau = p1
    else:
        tau = p2

    return tau

def increment(i,n):
    i_next = i%n + 1
    if i == n:
        i_next = 0
    return i_next

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
    return isConvex


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

    return Polygon_vertex, shifted_polygon_vertex

def getConvexPolygon(numVert, radius, radVar, angVar):
    isConvex = False
    
    while not(isConvex):
        Polygon_vertex, shifted_polygon_vertex = getPolygon(numVert, radius, radVar, angVar)
        isConvex = isAConvexPolygon(Polygon_vertex)
    return Polygon_vertex, shifted_polygon_vertex

def computeWLofCamera(height, alpha, beta):
    w = 2*height*np.tan(alpha/2)
    l = 2*height*np.tan(beta/2)
    return w, l

# # coord = [[1,1], [2,1], [1.8,1.8], [0.8,2], [0.5,1.5]]
# coord = [[1,1], [3,1], [2,3], [0.8,2], [0.5,1.5]]
# coord = [[1,2], [3,2], [3,1]]
# coord = [[-10, -40], [-36.28155339805825, -40.0], [-36.28155339805825, -10]]

# # V = coord[::-1]
# # # coord = [[146.76306045999095, -2.0473334316286795], [-60.526906172934744, -109.76533421120082], [-132.6429327946019, 20.193654150405926], [-14.865406821485804, 101.09338703482929], [78.12772846254278, 117.07811942387043]]
# # # coord = [[93.66465192394718, -110.36956689901635], [-66.07872666446623, -178.26289333143518], [-120.0938559030976, -68.63167151822334], [-91.59810431321333, 106.98604310933723], [163.91984334377128, 8.73263368860047]]
# # coord = [[98.50962429288698, -144.1519054946347], [24.203143228386516, -163.44423722112364], [-125.85339761977067, 37.88119097590002], [-114.68711430151565, 153.07972833204173], [94.98896813047023, 104.89551120000584]]
# V = coord
# #  #repeat the first point to create a 'closed loop'

# # # coord = [[93.0, -88.0], [-85.0, -135.0], [-154.0, 45.0], [-84.0, 101.0], [126.0, 170.0]]

# # print(findAntipodalPoints(V))

# path = getOpSweep(V, [2,0], [0,2], 2)


# coord.append(coord[0])
# xPath,yPath = zip(*path)
# # print(path)
# xs, ys = zip(*coord) #create lists of x and y values

# plt.figure()
# plt.plot(xs,ys) 
# plt.plot(xPath,yPath)
# plt.axis('scaled')

# plt.show() # if you need...