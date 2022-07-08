from cmath import cos
import math
import matplotlib.pyplot as plt

def draw_rectangle(center,length, width, yaw, color):
        x = []
        y = []
        x.append(center[0]+(+ width/2)*math.cos(abs(yaw))-(+ length/2)*math.sin(abs(yaw)))
        y.append(center[1]+(+ width/2)*math.sin(abs(yaw))+(+ length/2)*math.cos(abs(yaw)))
        x.append(center[0]+(+ width/2)*math.cos(abs(yaw))-(- length/2)*math.sin(abs(yaw)))
        y.append(center[1]+(+ width/2)*math.sin(abs(yaw))+(- length/2)*math.cos(abs(yaw)))
        x.append(center[0]+(- width/2)*math.cos(abs(yaw))-(- length/2)*math.sin(abs(yaw)))
        y.append(center[1]+(- width/2)*math.sin(abs(yaw))+(- length/2)*math.cos(abs(yaw)))
        x.append(center[0]+(- width/2)*math.cos(abs(yaw))-(+ length/2)*math.sin(abs(yaw)))
        y.append(center[1]+(- width/2)*math.sin(abs(yaw))+(+ length/2)*math.cos(abs(yaw)))
        x.append(center[0]+(+ width/2)*math.cos(abs(yaw))-(+ length/2)*math.sin(abs(yaw)))
        y.append(center[1]+(+ width/2)*math.sin(abs(yaw))+(+ length/2)*math.cos(abs(yaw)))
        plt.plot(x,y,color)

draw_rectangle([0,0],8,4,0,'red')
plt.axis("scaled")
plt.show()