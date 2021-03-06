#!/usr/bin/Python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import random
import numpy as np
from PIL import Image
import threading
if '2.7' in sys.version:
    import Queue as queue
else:
    import queue
import time
from sklearn.cluster import KMeans


class Data:
    DATA_ROOT = r'data/TrainImgMore'
    # RESIZE = [224, 224]
    RESIZE = [39, 39]
    RATIO = 1.0
    NUM_CLASSES = 30

    def __init__(self, start_ratio = 0.0, end_ratio = 1.0, name = '', sort_list = []):
        self.__chang_dir()

        # 初始化变量
        self.__name = name
        self.__data = []
        self.__sort_list = sort_list

        # 加载全部数据
        self.__load()
        self.__data_len = len(self.__data)

        # 检查输入参数
        start_ratio = min(max(0.0, start_ratio), 1.0)
        end_ratio = min(max(0.0, end_ratio), 1.0)

        # 根据比例计算数据的位置范围
        start_index = int(self.__data_len * start_ratio)
        end_index = int(self.__data_len * end_ratio)

        # 根据数据的位置范围 取数据
        self.__data = self.__data[start_index: end_index]
        self.__data_len = len(self.__data)
        random.shuffle(self.__data)

        self.__queue = queue.Queue()
        self.__stop_thread = False
        self.__thread = None

        self.__cur_index = 0


    @staticmethod
    def __chang_dir():
        # 将运行路径切换到当前文件所在路径
        cur_dir_path = os.path.split(__file__)[0]
        if cur_dir_path and os.path.abspath(os.path.curdir) != os.path.abspath(cur_dir_path):
            os.chdir(cur_dir_path)
            sys.path.append(cur_dir_path)


    ''' 加载数据 '''
    def __load(self):
        self.echo('Loading %s data ...' % self.__name)
        file_list = os.listdir(self.DATA_ROOT)
        file_len = len(file_list)

        for i, file_name in enumerate(file_list):
            progress = float(i + 1) / file_len * 100
            self.echo('\r >> progress: %.2f%% \t' % progress, False)

            split_file_name = os.path.splitext(file_name)
            no_list = split_file_name[0].split('_')

            if split_file_name[1].lower() != '.jpg' or int(no_list[-1]) == 1:
                continue

            self.__data.append( [split_file_name[0], os.path.join(self.DATA_ROOT, file_name)] )

        # self.echo(' sorting data ... ')
        # self.__data.sort(self.__sort)

        self.echo('\nFinish Loading\n')


    def __get_data(self):
        max_q_size = min(self.__data_len, 2000)
        while not self.__stop_thread:
            while self.__queue.qsize() <= max_q_size:
                file_name, img_path = self.__data[self.__cur_index]
                x, y = self.__get_x_y(img_path)

                self.__queue.put([x, y])
                self.__cur_index = (self.__cur_index + 1) % self.__data_len

            time.sleep(0.5)

        self.echo('\n*************************************\n Thread "get_%s_data" stop\n***********************\n' % self.__name)


    def start_thread(self):
        self.__thread = threading.Thread(target=self.__get_data, name=('get_%s_data' % self.__name))
        self.__thread.start()
        self.echo('Thread "get_%s_data" is running ... ' % self.__name)


    def stop(self):
        self.__stop_thread = True


    @staticmethod
    def __get_x_y(img_path):
        no_list = os.path.splitext(os.path.split(img_path)[1])[0].split('_')

        pig_no = int(no_list[0]) - 1
        label = np.zeros([Data.NUM_CLASSES])
        label[pig_no] = 1

        return Data.__get_patches(img_path), label


    @staticmethod
    def __get_patches(img_path):
        three_patches = Data.__get_three_patch(img_path)
        patch = Data.__add_padding(img_path)

        patch_list = []
        for i in range(3):
            for _patch in three_patches:
                patch_list.append(
                    _patch[:, :, i].reshape( np.hstack( [ _patch.shape[:-1], 1 ] ) )
                )

            patch_list.append(
                patch[:, :, i].reshape( np.hstack( [ patch.shape[:-1], 1 ] ) )
            )

        return np.array(patch_list)


    @staticmethod
    def __get_three_patch(img_path):
        np_image = np.array( Image.open(img_path) )
        h, w, c = np_image.shape

        if h > w:
            _size = w
            padding = int( (h - _size) / 2 )
            np_image_1 = np_image[:_size, :, :]
            np_image_2 = np_image[padding: padding + _size, :, :]
            np_image_3 = np_image[-_size:, :, :]

        else:
            _size = h
            padding = int( (w - _size) / 2 )
            np_image_1 = np_image[:, :_size, :]
            np_image_2 = np_image[:, padding: padding + _size, :]
            np_image_3 = np_image[:, -_size:, :]

        return [Data.__resize_np_img(np_image_1), Data.__resize_np_img(np_image_2), Data.__resize_np_img(np_image_3)]


    @staticmethod
    def __resize_np_img(np_image):
        return np.array( Image.fromarray(np_image).resize( Data.RESIZE ), dtype=np.float32 )


    # @staticmethod
    # def __get_kmeans_patch(img_path):
    #     image = Image.open(img_path)
    #     if image.size[0] > 320 or image.size[1] > 320:
    #         np_image = np.array( image.resize( np.cast['int32']( np.array(image.size) / 2 ) ) )
    #     else:
    #         np_image = np.array( image )
    #     np_tmp_image = ( np_image - (255.0 / 2) ) / 255.0
    #
    #     h, w, c = np_tmp_image.shape
    #
    #     if h < 60 or w < 60:
    #         return Data.__add_padding(img_path)
    #
    #     data = []
    #     for i in range(h):
    #         for j in range(w):
    #             pixel = np_tmp_image[i, j, :]
    #             data.append([ float(i - (h / 2.0)) / h, float(-j + (w / 2.0)) / w, pixel[0], pixel[1], pixel[2] ])
    #
    #     data = np.array(data)
    #
    #     k = 5
    #     estimator = KMeans(n_clusters=k)
    #     estimator.fit(data)
    #
    #     center = estimator.cluster_centers_
    #     center = center[:, :2]
    #
    #     center_x = np.cast['int32']( center[:, 0] * h + (h / 2.0) )
    #     center_y = np.cast['int32']( -center[:, 1] * w + (w / 2.0) )
    #
    #     patch_list = []
    #
    #     for i in range(k):
    #         x = center_x[i]
    #         y = center_y[i]
    #
    #         r_x_min = max(h / 10, 20)
    #         if x < r_x_min:
    #             x += int( r_x_min )
    #         elif h - x < max(h / 10, 20):
    #             x -= int( r_x_min )
    #
    #         r_y_min = max(w / 10, 20)
    #         if y < r_y_min:
    #             y += int( r_y_min )
    #         elif w - y < w / 10:
    #             y -= int( r_y_min )
    #
    #         r = int( min(x, y, h - x, w - y, h / 3, w / 3) ) - 1
    #
    #         np_new_img = np_image[x - r: x + r, y - r: y + r, :]
    #         new_img = Image.fromarray(np_new_img)
    #         # patch_list.append( np.array( new_img.resize(Data.RESIZE) ) )
    #
    #     # return patch_list

    
    @staticmethod
    def __add_padding(img_path):
        image = Image.open(img_path)
        w, h = image.size
        ratio = float(w) / h

        if abs(ratio - Data.RATIO) <= 0.1:
            return np.array( image.resize( Data.RESIZE ) )

        np_image = np.array(image)
        h, w, c = np_image.shape

        if ratio > Data.RATIO:
            new_h = int(float(w) / Data.RATIO)
            padding = int((new_h - h) / 2.0)

            np_new_image = np.zeros([new_h, w, c])
            np_new_image[padding: padding + h, :, :] = np_image

        else:
            new_w = int(float(h) * Data.RATIO)
            padding = int((new_w - w) / 2.0)

            np_new_image = np.zeros([h, new_w, c])
            np_new_image[:, padding: padding + w, :] = np_image

        new_image = Image.fromarray( np.cast['uint8'](np_new_image) )
        return np.array( new_image.resize( Data.RESIZE ) )


    def __sort(self, a, b):
        if self.__sort_list:
            index_a = self.__sort_list.index(a[0])
            index_b = self.__sort_list.index(b[0])
            if index_a < index_b:
                return -1
            elif index_a > index_b:
                return 1
            else:
                return 0

        if a[0] < b[0]:
            return -1
        elif a[0] > b[0]:
            return 1
        else:
            return 0


    @staticmethod
    def get_sort_list():
        img_no_set = set()
        for i, file_name in enumerate(os.listdir(Data.DATA_ROOT)):
            split_file_name = os.path.splitext(file_name)
            if split_file_name[1].lower() != '.jpg' or 'pig' in split_file_name[0]:
                continue

            img_no_set.add(split_file_name[0])

        img_no_list = list(img_no_set)
        random.shuffle(img_no_list)
        return img_no_list


    def next_batch(self, batch_size):
        X = []
        y = []
        for i in range(batch_size):
            while self.__queue.empty():
                time.sleep(0.2)
            if not self.__queue.empty():
                _x, _y = self.__queue.get()
                X.append(_x)
                y.append(_y)
        return np.array(X).transpose([1, 0, 2, 3, 4]), np.array(y)


    ''' 获取数据集大小 '''
    def get_size(self):
        return self.__data_len


    ''' 输出展示 '''
    @staticmethod
    def echo(msg, crlf=True):
        if crlf:
            print (msg)
        else:
            sys.stdout.write(msg)
            sys.stdout.flush()


# Download.run()
#
# train_data = Data(0.0, 0.64, 'train')
#
# print 'size:'
# print train_data.get_size()

# for i in range(10):
# batch_x, batch_y = train_data.next_batch(10)

# print '\n*************** %d *****************' % i
# print train_data.get_size()
# print batch_x.shape
# print batch_y.shape
#
#     tmp_x = batch_x[0]
#     o_tmp = Image.fromarray(tmp_x)
#     o_tmp.show()
#
#     time.sleep(1)
#
# train_data.stop()

# print 'y 0:'
# print batch_y[0]
#
# tmp_x_list = batch_x_list[0]
#
# print 'tmp_x_list'
# for i, x in enumerate(tmp_x_list):
#     print i
#     print x.shape
#     # o_tmp = Image.fromarray(x)
#     # o_tmp.show()
