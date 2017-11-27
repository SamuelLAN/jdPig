#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.split(__file__)[0]
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)

import cv2
import numpy as np
from sklearn.cluster import KMeans


class Test:
    RESIZE_SIZE = (300, 300)

    def __init__(self):
        self.__channel = 3
        self.__k = 2


    def run(self):
        self.__im = self.__load(r'/Users/samuellin/Documents/GitHub/jdPigRecognition/Data/TrainImg/15_31.jpg')
        self.__im = self.__im[:, :425, :]

        import copy
        origin = copy.deepcopy(self.__im)

        w, h, c = self.__im.shape
        im_size = w * h

        real_size = 0
        for i, val_i in enumerate(self.__im):
            for j, val_j in enumerate(val_i):
                b = float(val_j[0])
                g = float(val_j[1])
                r = float(val_j[2])

                if (80 < r < 96 and 70 < g < 80 and 65 < b < 80) \
                        or (145 < r < 155 and 115 < g < 121 and 100 < b < 115) \
                        or (141 < r < 147 and 113 < g < 118 and 112 < b < 120) \
                        or (114 < r < 120 and 94 < g < 100 and 94 < b < 100) \
                        or (52 < r < 60 and 40 < g < 50 and 40 < b < 50) \
                        or (95 < r < 105 and 82 < g < 89 and 78 < b < 85) \
                        or (123 < r < 131 and 106 < g < 113 and 95 < b < 103) \
                        or b < 35 or g < 45 or r < 40 or r > 252 \
                        or g > 225 or b > 215 or g / b > 1.17 or g / b < 0.8 \
                        or r / g > 3.1 or 2 > r / g > 1.45 or r / g < 1.05:
                    self.__im[i][j] = np.array([255, 255, 255])
                else:
                    real_size += 1

        real_ratio = float(real_size) / im_size * 100
        print 'real ratio: %.2f%%' % (real_ratio,)

        cv2.imshow('test', self.__im)
        cv2.waitKey(1)
        cv2.imshow('test1', origin)
        cv2.waitKey(1)
        import time
        time.sleep(100)

        # self.__im, self.__cov = self.__normalize(self.__im)
        #
        # w = self.__neighborWeight(self.__im)
        #
        # L = self.__calLaplace(w)
        #
        # eigen_v = self.__calEigenVector(L)
        #
        # normalize_eigen_v = self.__normalizeEigen(eigen_v)
        #
        # print 'normalize_eigen_v:'
        # print normalize_eigen_v
        # print normalize_eigen_v.shape
        #
        # ret = KMeans(n_clusters=self.__k).fit_predict(normalize_eigen_v)
        # print ret.shape
        #
        # print ret



    def __load(self, file_name):
        im = cv2.imread(file_name)
        return im
        # return cv2.resize(im, self.RESIZE_SIZE)


    # def __gaussian(self, x1, x2):
    #     x_minus = (x1 - x2).reshape((1, -1))
    #     x_dot_cov = np.dot( x_minus, np.linalg.inv(self.__cov) )
    #     x_cov_x = np.dot( x_dot_cov, np.transpose(x_minus) )
    #     exp = np.exp(- 1.0 / 2 * x_cov_x)
    #     cons = pow(2 * np.pi, self.__channel / 2.0) * np.sqrt( np.linalg.det(self.__cov) )
    #     return (1.0 / cons * exp)[0][0]
    #
    #
    # def __sim(self, i1, j1, i2, j2, ar_img):
    #     x1 = ar_img[i1, j1, :]
    #     x2 = ar_img[i2, j2, :]
    #     return self.__gaussian(x1, x2)
    #
    #
    # def __normalize(self, ar_img):
    #     shape = ar_img.shape
    #     ar_img = ar_img.reshape((-1, 3))
    #     ar_img = ( ar_img - np.mean(ar_img, 0) ) / np.std(ar_img, 0)
    #     cov = np.cov( np.transpose(ar_img) )
    #     return ar_img.reshape(shape), cov
    #
    #
    # def __normalizeEigen(self, eigen_vector):
    #     eigen_vector = np.transpose(eigen_vector)
    #     eigen_vector = ( eigen_vector - np.mean(eigen_vector, 0) ) / np.std(eigen_vector, 0)
    #     return eigen_vector
    #
    # def __neighborWeight(self, ar_img):
    #     rows, cols, channel = ar_img.shape
    #     W = np.zeros((rows * cols, rows * cols))
    #
    #     for i1 in range(rows):
    #         i2 = i1 - 1
    #         i3 = i1 + 1
    #
    #         for j1 in range(cols):
    #             j2 = j1 - 1
    #             j3 = j1 + 1
    #
    #             index1 = i1 * rows + j1
    #
    #             if 0 <= j2:
    #                 index2 = i1 * rows + j2
    #                 W[index1, index2] = self.__sim(i1, j1, i1, j2, ar_img)
    #
    #             if j3 < cols:
    #                 index2 = i1 * rows + j3
    #                 W[index1, index2] = self.__sim(i1, j1, i1, j3, ar_img)
    #
    #             if 0 <= i2:
    #                 index2 = i2 * rows + j1
    #                 W[index1, index2] = self.__sim(i1, j1, i2, j1, ar_img)
    #                 if 0 <= j2:
    #                     index2 = i2 * rows + j2
    #                     W[index1, index2] = self.__sim(i1, j1, i2, j2, ar_img)
    #                 if j3 < cols:
    #                     index2 = i2 * rows + j3
    #                     W[index1, index2] = self.__sim(i1, j1, i2, j3, ar_img)
    #
    #                     if i3 < cols:
    #                         index2 = i3 * rows + j1
    #                         W[index1, index2] = self.__sim(i1, j1, i3, j1, ar_img)
    #                         if 0 <= j2:
    #                             index2 = i3 * rows + j2
    #                             W[index1, index2] = self.__sim(i1, j1, i3, j2, ar_img)
    #                         if j3 < cols:
    #                             index2 = i3 * rows + j3
    #                             W[index1, index2] = self.__sim(i1, j1, i3, j3, ar_img)
    #
    #     return W
    #
    #
    # def __calLaplace(self, W):
    #     tmp_d = sum(W)
    #     D = np.diag( tmp_d )
    #     D_m = np.diag( 1.0 / np.sqrt(tmp_d) )
    #     L = D - W
    #     return np.dot( np.dot(D_m, L), D_m )
    #
    #
    # def __calEigenVector(self, L):
    #     eigen_value, eigen_vector = np.linalg.eig(L)
    #     return eigen_vector[: self.__k]


o_sc = Test()
o_sc.run()
