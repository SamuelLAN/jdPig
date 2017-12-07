#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys
import random
import numpy as np
from six.moves import cPickle as pickle
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


class Hamming:
    DATA_ROOT = r'feature'
    PCA_DIMENSION = 420
    # K_NEIGHBORS_LIST = [10, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000]

    THRESHOLD = 0.7

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

    def __train(self):
        self.echo('\nTraining  ... ')

        self.echo('  generating label_index_dict ... ')
        label_index_dict = {}
        for i in range(self.__train_size):
            y = self.__y_train[i]
            if y not in label_index_dict:
                label_index_dict[y] = []
            label_index_dict[y].append(i)

        self.echo('  shuffling label_index_dict ... ')
        for k, v in label_index_dict.iteritems():
            random.shuffle(v)

        self.echo('  binary x ... ')
        threshold = self.THRESHOLD

        self.__x_train[self.__x_train >= threshold] = 1
        self.__x_train[self.__x_train < threshold] = 0

        self.__x_val[self.__x_val >= threshold] = 1
        self.__x_val[self.__x_val < threshold] = 0

        self.__x_test[self.__x_test >= threshold] = 1
        self.__x_test[self.__x_test < threshold] = 0

        same_hamming_list = []
        diff_hamming_list = []

        same_num = 500
        diff_pre_y_num = 20

        self.echo('  training hamming ... ')

        for i in range(self.__train_size):
            x = self.__x_train[i]
            y = self.__y_train[i]

            same_index_list = label_index_dict[y][:same_num]
            for index in same_index_list:
                hamming = np.mean( np.equal(x, self.__x_train[index]) )
                same_hamming_list.append(hamming)

            for k, v in label_index_dict.iteritems():
                if k == y:
                    continue

                for j in range(diff_pre_y_num):
                    rand_x = self.__x_train[ v[random.randrange(0, len(v))] ]
                    hamming = np.mean( np.equal(x, rand_x) )
                    diff_hamming_list.append(hamming)

        same_hamming_list = np.array(same_hamming_list)
        diff_hamming_list = np.array(diff_hamming_list)

        mean_same_hamming = np.mean(same_hamming_list)
        mean_diff_hamming = np.mean(diff_hamming_list)

        self.echo('mean_same_hamming: %.6f' % mean_same_hamming)
        self.echo('mean_diff_hamming: %.6f' % mean_diff_hamming)

        self.__threshold = (mean_same_hamming + mean_diff_hamming) / 2.0

        self.echo('Finish training ')


    def __test(self, x_list, y_list, _size, name = ''):

        self.echo('  generating %s label_index_dict ... ' % name)
        label_index_dict = {}
        for i in range(_size):
            y = y_list[i]
            if y not in label_index_dict:
                label_index_dict[y] = []
            label_index_dict[y].append(i)

        self.echo('  shuffling %s label_index_dict ... ' % name)
        for k, v in label_index_dict.iteritems():
            random.shuffle(v)

        result_list = []
        test_num = 5

        self.echo('  testing %s hamming distance ... ' % name)
        for i in range(_size):
            x = x_list[i]
            y = y_list[i]

            for k, v in label_index_dict.iteritems():
                for j in range(test_num):
                    index = v[random.randrange(0, len(v))]
                    rand_x = x_list[index]
                    rand_y = y_list[index]
                    hamming = np.mean(np.equal(x, rand_x))

                    predict_is_same = (hamming >= self.__threshold)
                    label_is_same = (y == rand_y)

                    result_list.append(predict_is_same == label_is_same)

        result_list = np.array(result_list)
        accuracy = float( np.mean(result_list) )

        self.echo('%s accuracy: %.6f ' % (name, accuracy))


    # def __get_accuracy(self, batch_x, batch_y, batch_size):
    #     output = self.__classifier.predict(batch_x)
    #     return np.sum(np.equal(output, batch_y)) / float(batch_size) * 100.0

    def run(self):
        # best_val_accuracy = 0
        # best_val_k = 0
        # best_test_accuracy = 0
        # best_test_k = 0

        self.__train()

        self.__test(self.__x_train, self.__y_train, self.__train_size, 'training')
        self.__test(self.__x_val, self.__y_val, self.__val_size, 'validation')
        self.__test(self.__x_test, self.__y_test, self.__test_size, 'test')

        # for k in self.K_NEIGHBORS_LIST:
        #     self.__train(k)
        #
        #     self.echo('\nMeasuring val accuracy ... ')
        #     val_accuracy = self.__get_accuracy(self.__x_val, self.__y_val, self.__val_size)
        #     self.echo('Finish measuring ')
        #     self.echo('val accuracy: %.6f%% ' % val_accuracy)
        #
        #     if val_accuracy > best_val_accuracy:
        #         best_val_accuracy = val_accuracy
        #         best_val_k = k
        #         self.echo('best k: %d ' % k)
        #
        #     self.echo('\nMeasuring test accuracy ... ')
        #     test_accuracy = self.__get_accuracy(self.__x_test, self.__y_test, self.__test_size)
        #     self.echo('Finish measuring ')
        #     self.echo('test accuracy: %.6f%% ' % test_accuracy)
        #
        #     if test_accuracy > best_test_accuracy:
        #         best_test_accuracy = test_accuracy
        #         best_test_k = k
        #         self.echo('best k: %d ' % k)
        #
        # self.echo('\nbest val accuracy: %.6f  k: %d ' % (best_val_accuracy, best_val_k))
        # self.echo('best test accuracy: %.6f  k: %d ' % (best_test_accuracy, best_test_k))

    ''' 获取下个 batch '''

    def next_batch(self, batch_size, loop=True):
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


o_data = Hamming()
o_data.run()
