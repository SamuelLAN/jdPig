#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from six.moves import cPickle as pickle
from sklearn.decomposition import PCA

class Data:
    DATA_ROOT = r'feature'
    PCA_DIMENSION = 420

    def __init__(self, prefix='train'):
        self.__prefix = prefix
        self.__deep_id_list = []
        self.__y_list = []

        self.__chang_dir()
        self.__load()


    def __load(self):
        for file_name in os.listdir(self.DATA_ROOT):
            if os.path.splitext(file_name)[1].lower() != '.pkl' or self.__prefix not in file_name:
                continue

            with open(os.path.join(self.DATA_ROOT, file_name), 'rb') as f:
                deep_id_list, y_list = zip(*pickle.load(f))
            
            self.__deep_id_list += deep_id_list
            self.__y_list += y_list

        self.__deep_id_list = np.array(self.__deep_id_list)
        self.__y_list = np.array(self.__y_list)

        pca = PCA(n_components=self.PCA_DIMENSION)
        self.__deep_id_list = pca.fit_transform(self.__deep_id_list)

        print 'deep_id_list.shape'
        print self.__deep_id_list.shape
        print 'y_list.shape'
        print self.__y_list.shape



    ''' 将运行路径切换到当前文件所在路径 '''
    @staticmethod
    def __chang_dir():
        cur_dir_path = os.path.split(__file__)[0]
        if cur_dir_path and os.path.abspath(os.path.curdir) != os.path.abspath(cur_dir_path):
            os.chdir(cur_dir_path)
            sys.path.append(cur_dir_path)


    ''' 输出展示 '''
    @staticmethod
    def echo(msg, crlf=True):
        if crlf:
            print msg
        else:
            sys.stdout.write(msg)
            sys.stdout.flush()


o_data = Data('train')
