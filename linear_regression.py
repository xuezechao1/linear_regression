# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pylab as plt

def gradient_descent(x_data, y_data, k):
    # 梯度下降
    m = 2
    c = 3
    step = 0.001

    while k>0:
        divine_y = [m * i + c for i in x_data]
        D_value = [divine_y[i] - y_data[i] for i in range(len(divine_y))]
        D_value_m = step * np.dot(np.array(D_value), np.array(x_data))
        D_value_c = step * np.dot(np.array(D_value), np.ones(len(x_data)))

        m = m - D_value_m
        c = c - D_value_c

        if k % 100 == 0:
            # print(D_value_m, D_value_c, sum([i * i for i in D_value]))
            print('迭代次数:{0} m:{1} c:{2} 误差值:{3}'.format(k, D_value_m, D_value_c, sum([i * i for i in D_value])))

        k = k-1

    return m, c

def least_square_method(x_data, y_data):
    # 最小二乘法
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    xy = np.multiply(x_data, y_data)
    xx = [i * i for i in x_data]

    xy_mean = round(np.mean(xy), 2)
    x_mean = round(np.mean(x_data), 2)
    y_mean = round(np.mean(y_data), 2)
    xx_mean = round(np.mean(xx), 2)

    m = round((xy_mean - x_mean * y_mean)/(xx_mean - x_mean * x_mean), 2)
    c = round(y_mean - m * x_mean, 2)

    # print(xy_mean, x_mean, y_mean, xx_mean, m, c)

    return m, c

if __name__ == '__main__':
    y_data = np.random.randint(0, 10, size=10)
    y_data = sorted(list(y_data))
    print(y_data)

    x_data = []
    data = []
    for i in range(len(y_data)):
        x_data.append(i)
        data.append([i , y_data[i]])

    method = input('请选择平方损失函数计算方式[1:最小二乘法 2:梯度下降 3:1和2同时计算]:')
    method = int(method)
    if method == 1:
        m, c = least_square_method(x_data, y_data)
    elif method == 2:
        k = input('请输入迭代次数:')
        k = int(k)
        m, c = gradient_descent(x_data, y_data, k)
    elif method == 3:
        k = input('请输入迭代次数:')
        k = int(k)
        least_square_method_m, least_square_method_c = least_square_method(x_data, y_data)
        gradient_descent_m, gradient_descent_c = gradient_descent(x_data, y_data, k)

    if method == 1 or method == 2:
        linear_regression_y = []
        for i in range(len(x_data)):
            linear_regression_y.append(round( i * m + c ,2))

        plt.scatter(x_data, y_data, s=10, color='r')
        plt.plot(x_data,
                 linear_regression_y,
                 linewidth=2 ,
                 color='green',
                 linestyle='-',
                 marker='*',
                 label ='{0} {1}'.format(round(m, 2), round(c, 2)))
        plt.legend(loc='upper left')

        plt.title('linear regression', fontsize=24)
        plt.xlabel('X', fontsize=14)
        plt.ylabel('Y', fontsize=14)

        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.show()
    elif method == 3:
        least_square_method_y = []
        gradient_descent_y = []
        for i in range(len(x_data)):
            least_square_method_y.append(round( i * least_square_method_m + least_square_method_c ,2))
            gradient_descent_y.append(round(i * gradient_descent_m + gradient_descent_c, 2))

        plt.scatter(x_data, y_data, s=10, color='r')
        plt.plot(x_data,
                 least_square_method_y,
                 linewidth=2,
                 color='green',
                 linestyle='-',
                 marker='*',
                 label='{0} {1}'.format(round(least_square_method_m, 2), round(least_square_method_c, 2)))
        plt.legend(loc='upper left')

        plt.plot(x_data,
                 gradient_descent_y,
                 linewidth=2,
                 color='blue',
                 linestyle='-',
                 marker='*',
                 label='{0} {1}'.format(round(gradient_descent_m, 2), round(gradient_descent_c, 2)))
        plt.legend(loc='upper left')

        plt.title('linear regression', fontsize=24)
        plt.xlabel('X', fontsize=14)
        plt.ylabel('Y', fontsize=14)

        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.show()