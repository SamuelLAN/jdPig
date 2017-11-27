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


class SpectralClustering:
    RESIZE_SIZE = (50, 50)

    def __init__(self):
        self.__channel = 3
        self.__k = 2


    def run(self):
        self.__im = self.__load(r'/Users/samuellin/Desktop/691.JPG')
        self.__im, self.__cov = self.__normalize(self.__im)

        print 'cal neighborWeight'

        w = self.__neighborWeight(self.__im)

        print 'cal laplace'

        L = self.__calLaplace(w)

        print 'cal eigen vector'

        eigen_v = self.__calEigenVector(L)

        normalize_eigen_v = self.__normalizeEigen(eigen_v)

        print 'normalize_eigen_v:'
        print normalize_eigen_v
        print normalize_eigen_v.shape

        ret = KMeans(n_clusters=self.__k).fit_predict(normalize_eigen_v)
        print ret.shape

        print ret

        ret = ret.reshape(self.RESIZE_SIZE)

        for i, val_i in enumerate(self.__im):
            for j, val_j in enumerate(val_i):
                if ret[i][j] == 0:
                    self.__im[i][j] = np.array([0, 0, 0])

        cv2.imshow('test', self.__im)
        cv2.waitKey(1)

        import time
        time.sleep(100)


    def __load(self, file_name):
        im = cv2.imread(file_name)
        return cv2.resize(im, self.RESIZE_SIZE)


    def __gaussian(self, x1, x2):
        x_minus = (x1 - x2).reshape((1, -1))
        x_dot_cov = np.dot( x_minus, np.linalg.inv(self.__cov) )
        x_cov_x = np.dot( x_dot_cov, np.transpose(x_minus) )
        exp = np.exp(- 1.0 / 2 * x_cov_x)
        cons = pow(2 * np.pi, self.__channel / 2.0) * np.sqrt( np.linalg.det(self.__cov) )
        return (1.0 / cons * exp)[0][0]


    def __sim(self, i1, j1, i2, j2, ar_img):
        x1 = ar_img[i1, j1, :]
        x2 = ar_img[i2, j2, :]
        return self.__gaussian(x1, x2)


    def __normalize(self, ar_img):
        shape = ar_img.shape
        ar_img = ar_img.reshape((-1, 3))
        ar_img = ( ar_img - np.mean(ar_img, 0) ) / np.std(ar_img, 0)
        cov = np.cov( np.transpose(ar_img) )
        return ar_img.reshape(shape), cov


    def __normalizeEigen(self, eigen_vector):
        eigen_vector = np.transpose(eigen_vector)
        eigen_vector = ( eigen_vector - np.mean(eigen_vector, 0) ) / np.std(eigen_vector, 0)
        return eigen_vector

    def __neighborWeight(self, ar_img):
        rows, cols, channel = ar_img.shape
        W = np.zeros((rows * cols, rows * cols))

        for i1 in range(rows):
            i2 = i1 - 1
            i3 = i1 + 1

            for j1 in range(cols):
                j2 = j1 - 1
                j3 = j1 + 1

                index1 = i1 * rows + j1

                if 0 <= j2:
                    index2 = i1 * rows + j2
                    W[index1, index2] = self.__sim(i1, j1, i1, j2, ar_img)

                if j3 < cols:
                    index2 = i1 * rows + j3
                    W[index1, index2] = self.__sim(i1, j1, i1, j3, ar_img)

                if 0 <= i2:
                    index2 = i2 * rows + j1
                    W[index1, index2] = self.__sim(i1, j1, i2, j1, ar_img)
                    if 0 <= j2:
                        index2 = i2 * rows + j2
                        W[index1, index2] = self.__sim(i1, j1, i2, j2, ar_img)
                    if j3 < cols:
                        index2 = i2 * rows + j3
                        W[index1, index2] = self.__sim(i1, j1, i2, j3, ar_img)

                        if i3 < cols:
                            index2 = i3 * rows + j1
                            W[index1, index2] = self.__sim(i1, j1, i3, j1, ar_img)
                            if 0 <= j2:
                                index2 = i3 * rows + j2
                                W[index1, index2] = self.__sim(i1, j1, i3, j2, ar_img)
                            if j3 < cols:
                                index2 = i3 * rows + j3
                                W[index1, index2] = self.__sim(i1, j1, i3, j3, ar_img)

        return W


    def __calLaplace(self, W):
        tmp_d = sum(W)
        D = np.diag( tmp_d )
        D_m = np.diag( 1.0 / np.sqrt(tmp_d) )
        L = D - W
        return np.dot( np.dot(D_m, L), D_m )


    def __calEigenVector(self, L):
        eigen_value, eigen_vector = np.linalg.eig(L)
        return eigen_vector[:]


o_sc = SpectralClustering()
o_sc.run()
