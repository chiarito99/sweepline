import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
class Find_intersect():

    def __init__(self, resolution, move_dr):
        self.resolution = resolution
        self.move_dr = move_dr
        # other para
        self.x_ori = 0
        self.y_ori = 0
        self.th = 0
        self.cali_resolution = resolution
        self.scale_para = 1
        self.limit_ratio = 1000

    #calibarate xo yo, resolution
    def cali_para(self, ox, oy):
        cali_ox = []
        cali_oy = []
        # print(ox)
        # print(oy)
        if (max(oy) - min(oy)) / self.resolution > self.limit_ratio:
            self.resolution = round(self.resolution)
            return ox, oy
        else:
            if self.resolution % 1 == 0:
                self.scale_para = 1
                return ox, oy
            elif (self.resolution * 10) % 1 ==0:
                self.scale_para = 10
                self.resolution = self.resolution*10
                cali_ox = list(np.array(ox)*10)
                cali_oy = list(np.array(oy)*10)
            else:
                self.scale_para = 100
                self.resolution = round(self.resolution*100)
                cali_ox = list(np.array(ox)*100)
                cali_oy = list(np.array(oy)*100)
        # print(self.scale_para)
        # print(self.resolution)
        # print(cali_ox)
        # print(cali_oy)
        return cali_ox, cali_oy

    #tim cac toa do cat voi resolution = 0.5
    def find_intersect1(self,ox, oy):
        x_grid =[]
        y_grid =[]
        # print(cox)
        # print(coy)
        # print(self.resolution)
        for k in range(len(ox) - 1):
            xA = ox[k]
            yA = oy[k]
            xB = ox[k + 1]
            yB = oy[k + 1]
            # truong hop y=a
            if oy[k] == oy[k + 1]:
                x = list(np.arange(min(ox[k], ox[k + 1]), max(ox[k], ox[k + 1]), 0.5))
                y = list(oy[k] * np.ones(len(x)))
                x_grid.extend(x)
                y_grid.extend(y)
                continue
            for i in np.arange(int(min(oy[k], oy[k + 1])), int(max(oy[k], oy[k + 1]))+1, 0.5):
                x = (i - yA) * (xB - xA) / (yB - yA) + xA
                x_grid.append(x)
                y_grid.append(i)
        # print(x_grid)
        # print(y_grid)
        return x_grid, y_grid
    # tìm các tọa do cat voi resolution bat ky
    def find_intersect(self, ox, oy):
        x_intersect = []
        y_intersect = []
        x_grid, y_grid = self.find_intersect1(ox, oy)
        offset = -self.resolution / 2
        begin = int(self.resolution / 2)
        ed = int(max(oy) - self.resolution / 2)
        # print(x_grid)
        # print(y_grid)
        # print(offset)
        for i in np.arange(begin, ed, 0.5):
            if (i + offset) % self.resolution == 0:
                if self.move_dr == 1:
                    x_intersect.extend([min(self.take_element(x_grid, self.indexes(y_grid, i))),
                                        max(self.take_element(x_grid, self.indexes(y_grid, i)))])
                    y_intersect.extend([i, i])
                else:
                    x_intersect.extend([max(self.take_element(x_grid, self.indexes(y_grid, i))),
                                        min(self.take_element(x_grid, self.indexes(y_grid, i)))])
                    y_intersect.extend([i, i])

            if (i + offset) % self.resolution == self.resolution - 1:
                self.swap_moving_direction()
        #print(y_intersect)
        return x_intersect, y_intersect
    # ham lay cac index cua list co gia tri ref
    def indexes(self, list, ref):
        result = []
        for i in range(len(list)):
            if list[i] == ref:
                result.append(i)
        return result
    # hàm lấy các gái trị trong list với index trong index_list
    def take_element(self, list, index_list):
        return [list[i] for i in index_list]
    def swap_moving_direction(self):
        self.move_dr *= -1
    def convert(self, ox, oy):
        #calibarate
        ox, oy = self.cali_para(ox, oy)

        self.find_base(ox, oy)
        cx = []
        cy = []
        T_matrix = np.matrix([[np.cos(self.th), -np.sin(self.th), 0, self.x_ori],
                              [np.sin(self.th),  np.cos(self.th), 0, self.y_ori],
                              [0,                0,               1,          0],
                              [0,                0,               0,          1]])
        T_matrix = linalg.inv(T_matrix)
        for i in range(len(ox)):
            temp = T_matrix * np.matrix([[ox[i]],
                                        [oy[i]],
                                        [0],
                                        [1]])
            cx.append(float(np.squeeze(np.array(temp[0]))))
            cy.append(float(np.squeeze(np.array(temp[1]))))
        # hiệu chỉnh co trường hợp xoay và dich xong vẫn có các tọa độ âm(x_point=0)
        # self.x_ori = self.x_ori + min(cx)
        # cx = list(np.array(cx) - min(cx))
        return cx, cy
    def in_convert(self,cx, cy):
     # Dịch và xoay, scale lại hình
        in_cx = []
        in_cy = []
        T_matrix = np.matrix([[np.cos(self.th), -np.sin(self.th), 0, self.x_ori],
                              [np.sin(self.th),  np.cos(self.th), 0, self.y_ori],
                              [0,                0,               1,          0],
                              [0,                0,               0,          1]])
        for i in range(len(cx)):
            temp = T_matrix * np.array([[cx[i]],
                                        [cy[i]],
                                        [0],
                                        [1]])
            in_cx.append(float(np.squeeze(np.array(temp[0])/self.scale_para)))
            in_cy.append(float(np.squeeze(np.array(temp[1])/self.scale_para)))
        return in_cx, in_cy
    # hàm tìm vector base,o_ori,y_ori góc xoay
    def find_base(self, ox, oy):
        x_new = []
        y_new = []
        max_norm = 0
        x_vector = []
        y_vector = []
        for i in range(len(ox) - 1):
            x_vector.append(ox[i+1] - ox[i])
            y_vector.append(oy[i+1] - oy[i])
            norm = np.sqrt(x_vector[i]**2 + y_vector[i]**2)
            if norm > max_norm:
                max_norm = norm
                self.x_ori = ox[i]
                self.y_ori = oy[i]
                self.th = np.arctan2(y_vector[i], x_vector[i])
def planning(resolution, dr_move, xo, yo):
    emp = Find_intersect(resolution, dr_move)
    ox, oy = emp.convert(xo, yo)
    cx, cy = emp.find_intersect(ox, oy)
    x, y = emp.in_convert(cx, cy)
    return x, y
# xo = [0, 150, 90,  60,  -20,  0 ]
# yo = [0,  0, 100, 120, 60,    0 ]
# x, y = planning(2.333, -1, xo, yo)
# print(x)
# plt.plot(xo, yo)
# plt.plot(x, y)
# plt.show()
