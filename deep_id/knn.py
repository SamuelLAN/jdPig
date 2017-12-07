#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from six.moves import cPickle as pickle
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


class KNN:
    DATA_ROOT = r'feature'
    PCA_DIMENSION = 420
    K_NEIGHBORS_LIST = [10, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000]

    def __init__(self):
        self.__deep_id_list = []
        self.__y_list = []
        self.__data_len = 0
        self.__index_list = []
        self.__cur_index = 0

        self.__chang_dir()

        self.__x_train, self.__y_train, self.__train_size = self.__load('train')
        self.__x_val, self.__y_val, self.__val_size = self.__load('val')
        self.__x_test, self.__y_test, self.__test_size = self.__load('test')


    def __load(self, prefix='train'):
        self.echo('\nLoading %s data ... ' % prefix)
        deep_id_list = []
        y_list = []

        for file_name in os.listdir(self.DATA_ROOT):
            if os.path.splitext(file_name)[1].lower() != '.pkl' or prefix not in file_name:
                continue

            with open(os.path.join(self.DATA_ROOT, file_name), 'rb') as f:
                _deep_id_list, _y_list = zip(*pickle.load(f))
            
            deep_id_list += _deep_id_list
            y_list += _y_list

        deep_id_list = np.array(deep_id_list)
        y_list = np.array(y_list)
        data_len = y_list.shape[0]

        # pca 降维
        self.echo(' pca %s data ... ' % prefix)
        pca = PCA(n_components=self.PCA_DIMENSION)
        deep_id_list = pca.fit_transform(deep_id_list)

        self.echo('Finish loading ')

        return deep_id_list, y_list, data_len


    def __train(self, k):
        self.__classifier = KNeighborsClassifier(n_neighbors=k)
        self.echo('\nTraining knn classifier ... ')
        self.__classifier.fit(self.__x_train, self.__y_train)
        self.echo('Finish knn classifier ')


    def __get_accuracy(self, batch_x, batch_y, batch_size):
        output = self.__classifier.predict(batch_x)
        return np.sum( np.equal(output, batch_y) ) / float(batch_size) * 100.0


    def run(self):
        best_val_accuracy = 0
        best_val_k = 0
        best_test_accuracy = 0
        best_test_k = 0

        for k in self.K_NEIGHBORS_LIST:
            self.__train(k)

            self.echo('\nMeasuring val accuracy ... ')
            val_accuracy = self.__get_accuracy(self.__x_val, self.__y_val, self.__val_size)
            self.echo('Finish measuring ')
            self.echo('val accuracy: %.6f%% ' % val_accuracy)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_val_k = k
                self.echo('best k: %d ' % k)

            self.echo('\nMeasuring test accuracy ... ')
            test_accuracy = self.__get_accuracy(self.__x_test, self.__y_test, self.__test_size)
            self.echo('Finish measuring ')
            self.echo('test accuracy: %.6f%% ' % test_accuracy)

            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_test_k = k
                self.echo('best k: %d ' % k)

        self.echo('\nbest val accuracy: %.6f  k: %d ' % (best_val_accuracy, best_val_k))
        self.echo('best test accuracy: %.6f  k: %d ' % (best_test_accuracy, best_test_k))


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

        x_list = self.__deep_id_list[start_index: end_index]
        y_list = self.__deep_id_list[start_index: end_index]

        if not loop:
            self.__cur_index = end_index
            return np.array(x_list), np.array(y_list)

        if not left_num:
            self.__cur_index = end_index if end_index < self.__data_len else 0
            return np.array(x_list), np.array(y_list)

        while left_num:
            end_index = left_num
            if end_index > self.__data_len:
                left_num = end_index - self.__data_len
                end_index = self.__data_len
            else:
                left_num = 0

            left_x_list = self.__deep_id_list[: end_index]
            left_y_list = self.__deep_id_list[: end_index]
            x_list += left_x_list
            y_list += left_y_list

        self.__cur_index = end_index if end_index < self.__data_len else 0
        return np.array(x_list), np.array(y_list)


    def get_size(self):
        return self.__data_len


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


o_data = KNN()
o_data.run()
