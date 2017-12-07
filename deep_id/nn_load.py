#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys
import random
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
        self.__data_len = 0
        self.__index_list = []
        self.__cur_index = 0

        self.__chang_dir()
        self.__load()
        self.__generate_index()


    def __load(self):
        self.echo('\nLoading %s data ... ' % self.__prefix)

        for file_name in os.listdir(self.DATA_ROOT):
            if os.path.splitext(file_name)[1].lower() != '.pkl' or self.__prefix not in file_name:
                continue

            with open(os.path.join(self.DATA_ROOT, file_name), 'rb') as f:
                deep_id_list, y_list = zip(*pickle.load(f))
            
            self.__deep_id_list += deep_id_list
            self.__y_list += y_list

        self.__deep_id_list = np.array(self.__deep_id_list)
        self.__y_list = np.array(self.__y_list)

        # pca 降维
        self.echo(' pca %s data ... ' % self.__prefix)
        pca = PCA(n_components=self.PCA_DIMENSION)
        self.__deep_id_list = pca.fit_transform(self.__deep_id_list)

        self.echo('Finish loading ')


    ''' 生成数据索引 '''
    def __generate_index(self):
        self.echo('\n Generating %s data index ... ' % self.__prefix)
        for i in range(self.__data_len):
            for j in range(self.__data_len):
                self.__index_list.append([i, j])

        self.__data_len = len(self.__index_list)
        random.shuffle(self.__index_list)
        self.echo('Finish generating data index ')


    def __get_data_from_index(self, index_list):
        x_list = []
        y_list = []
        for i, j in index_list:
            x = np.hstack([self.__deep_id_list[i], self.__deep_id_list[j]])
            x_list.append(x)
        return np.array(x_list)


    ''' 获取下个 batch '''
    def next_batch(self, batch_size, loop = True):
        if not loop and self.__cur_index >= self.__data_len:
            return None, None

        start_index = self.__cur_index
        end_index = self.__cur_index + batch_size
        left_num = 0

        if end_index >= self.__data_len:
            left_num = end_index - self.__data_len
            end_index = self.__data_len

        index_list = self.__index_list[start_index: end_index]

        if not loop:
            self.__cur_index = end_index
            return self.__get_data_from_index(index_list)

        if not left_num:
            self.__cur_index = end_index if end_index < self.__data_len else 0
            return self.__get_data_from_index(index_list)

        while left_num:
            end_index = left_num
            if end_index > self.__data_len:
                left_num = end_index - self.__data_len
                end_index = self.__data_len
            else:
                left_num = 0

            left_index_list = self.__index_list[: end_index]
            index_list += left_index_list

        self.__cur_index = end_index if end_index < self.__data_len else 0
        return self.__get_data_from_index(index_list)


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
