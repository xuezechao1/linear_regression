#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf
from xzc_tools import tools
import matplotlib.pyplot as plt
from sklearn import datasets
from tensorflow.python.framework import ops

if __name__ == '__main__':
    try:
        # 导入数据
        iris = datasets.load_iris()
        x_vals = np.array([x[3] for x in iris.data])
        y_vals = np.array([y[0] for y in iris.data])

        # 声明学习率，占位符，模型，批量数据大小，循环次数
        maxCycle = 100
        learning_rate = 0.05
        batch_size = 25
        x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        A = tf.Variable(tf.random_normal(shape=[1, 1]))
        b = tf.Variable(tf.random_normal(shape=[1, 1]))

        # 增加线性模型 y = Ax + b
        model_output = tf.add(tf.matmul(x_data, A), b)

        # 声明L2损失函数，为批量损失的平均值
        loss = tf.reduce_mean(tf.square(y_target - model_output))

        # 初始化变量
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

            loss_vals = []
            for i in range(maxCycle):
                rand_index = np.random.choice(len(x_vals), size=batch_size)
                rand_x = np.transpose([x_vals[rand_index]])
                rand_y = np.transpose([y_vals[rand_index]])

                sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

                temp_loss = sess.run(loss, feed_dict={x_data: np.transpose([x_vals]), y_target: np.transpose([y_vals])})
                loss_vals.append(temp_loss)
                if i % 100 == 0:
                    print('step:{0},A:{1},b:{2},loss:{3}'.format(
                        str(i), str(sess.run(A)), str(sess.run(b)), str(temp_loss)
                    ))

            # 抽取系数，创建拟合直线
            [[A_result]] = sess.run(A)
            [[b_result]] = sess.run(b)
            pre_result = [A_result * i + b_result for i in x_vals]

            # 绘制结果图像
            plt.figure()
            plt.subplot(1,2,1)
            plt.plot(x_vals, y_vals, 'o', label='data_points')
            plt.plot(x_vals, pre_result, 'r-', label='predict points', linewidth=3)
            plt.legend(loc='upper left')
            plt.title('iris')
            plt.xlabel('x')
            plt.ylabel('y')

            plt.subplot(1,2,2)
            plt.plot(loss_vals, 'k-')
            plt.title('L2 loss pre generation')
            plt.xlabel('generation')
            plt.ylabel('L2 loss')
            plt.show()

    except Exception as msg:
        tools.printInfo(2, msg)
        sys.exit()