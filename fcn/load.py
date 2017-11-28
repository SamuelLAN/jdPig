#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

import zipfile
import random
import numpy as np
from PIL import Image
from six.moves.urllib.request import urlretrieve


'''
    下载数据
'''
class Download:
    URL = 'http://www.lin-baobao.com/pig/data.zip'
    DATA_ROOT = r'data'
    FILE_NAME = 'data.zip'
    EXPECTED_BYTES = 49462122
    FILE_NUM = 375

    def __init__(self):
        pass


    ''' 将运行路径切换到当前文件所在路径 '''
    @staticmethod
    def __changDir():
        cur_dir_path = os.path.split(__file__)[0]
        if cur_dir_path:
            os.chdir(cur_dir_path)
            sys.path.append(cur_dir_path)


    ''' 下载进度 '''
    @staticmethod
    def __downloadProgressHook(count, block_size, total_size):
        sys.stdout.write('\r >> Downloading %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()


    ''' 判断是否需要下载；若需，下载数据压缩包 '''
    @staticmethod
    def __maybeDownload(force=False):
        """Download a file if not present, and make sure it's the right size."""
        if not os.path.isdir(Download.DATA_ROOT):
            os.mkdir(Download.DATA_ROOT)
        file_path = os.path.join(Download.DATA_ROOT, Download.FILE_NAME)
        if force or not os.path.exists(file_path):
            print 'Attempting to download: %s' % Download.FILE_NAME
            filename, _ = urlretrieve(Download.URL, file_path, reporthook=Download.__downloadProgressHook)
            print '\nDownload Complete!'
        stat_info = os.stat(file_path)
        if stat_info.st_size == Download.EXPECTED_BYTES:
            print 'Found and verified %s' % file_path
        else:
            raise Exception(
                'Failed to verify ' + file_path + '. Can you get to it with a browser?')


    @staticmethod
    def __checkFileNum():
        if not os.path.isdir(Download.DATA_ROOT):
            return False
        file_num = 0
        for file_name in os.listdir(Download.DATA_ROOT):
            if os.path.splitext(file_name)[1].lower() != '.jpg':
                continue
            file_num += 1
        if file_num != Download.FILE_NUM:
            return False
        return True


    @staticmethod
    def __maybeExtract(force=False):
        file_path = os.path.join(Download.DATA_ROOT, Download.FILE_NAME)

        zip_files = zipfile.ZipFile(file_path, 'r')
        for filename in zip_files.namelist():
            if '__MACOSX' in filename:
                continue
            print '\t extracting %s ...' % filename
            data = zip_files.read(filename)
            with open(os.path.join(Download.DATA_ROOT, filename), 'wb') as f:
                f.write(data)


    @staticmethod
    def run():
        Download.__changDir()   # 将路径切换到当前路径

        if Download.__checkFileNum():
            print 'data exist in %s' % Download.DATA_ROOT
            return

        Download.__maybeDownload()

        print 'Extracting data ...'

        Download.__maybeExtract()

        print 'Finish Extracting'

        print 'done'


'''
 Data: 取数据到基类
 对外提供接口:
    get_size()
    next_batch()
'''
class Data:
    DATA_ROOT = r'data'
    IMAGE_SCALE = 2

    def __init__(self, start_ratio = 0.0, end_ratio = 1.0, name = ''):
        # 初始化变量
        self.__name = name
        self.__data = []
        self.__y = {}

        # 加载全部数据
        self.__load()
        self.__dataLen = len(self.__data)

        # 检查输入参数
        start_ratio = min(max(0.0, start_ratio), 1.0)
        end_ratio = min(max(0.0, end_ratio), 1.0)

        # 根据比例计算数据的位置范围
        start_index = int(self.__dataLen * start_ratio)
        end_index = int(self.__dataLen * end_ratio)

        # 根据数据的位置范围 取数据
        self.__data = self.__data[start_index: end_index]
        self.__dataLen = len(self.__data)
        random.shuffle(self.__data)

        self.__curIndex = 0


    ''' 加载数据 '''
    def __load(self):
        self.echo('Loading %s data ...' % self.__name)
        file_list = os.listdir(self.DATA_ROOT)
        file_len = len(file_list)

        for i, file_name in enumerate(file_list):
            progress = float(i) / file_len * 100
            self.echo('\rprogress: %.2f%% \t' % progress, False)

            if os.path.splitext(file_name)[1].lower() != '.jpg':
                continue

            if 'mask' in file_name and file_name not in self.__y:
                self.__y[file_name] = self.__get_mask(file_name)

            if 'mask' not in file_name:
                img_no = file_name.split('_')[0]
                y_file_name = img_no + '_mask.jpg'

                if y_file_name not in self.__y:
                    self.__y[y_file_name] = self.__get_mask(y_file_name)

                image = Image.open(os.path.join(self.DATA_ROOT, file_name))
                image = np.array(image.resize( np.array(image.size) / self.IMAGE_SCALE ))
                self.__data.append([image, self.__y[y_file_name]])

        self.echo('\nFinish Loading\n')


    ''' 将 mask 图转为 0 1 像素 '''
    @staticmethod
    def __get_mask(file_name):
        mask = Image.open(os.path.join(Data.DATA_ROOT, file_name)).convert('L')
        mask = np.array(mask.resize( np.array(mask.size) / Data.IMAGE_SCALE ))
        mask[mask == 255] = 0
        mask[mask > 0] = 1
        mask = mask.reshape(list(mask.shape) + [1])
        return mask


    ''' 获取下个 batch '''
    def next_batch(self, batch_size, loop = True):
        if not loop and self.__curIndex >= self.__dataLen:
            return None, None

        start_index = self.__curIndex
        end_index = self.__curIndex + batch_size
        left_num = 0

        if end_index >= self.__dataLen:
            left_num = end_index - self.__dataLen
            end_index = self.__dataLen

        X, y = zip(*self.__data[start_index: end_index])
        if not left_num or not loop:
            self.__curIndex = end_index if end_index < self.__dataLen else 0
            return np.array(X), np.array(y)

        while left_num:
            end_index = left_num
            if end_index > self.__dataLen:
                left_num = end_index - self.__dataLen
                end_index = self.__dataLen
            else:
                left_num = 0

            left_x, left_y = zip(*self.__data[: end_index])
            X += left_x
            y += left_y

        self.__curIndex = end_index if end_index < self.__dataLen else 0
        return np.array(X), np.array(y)


    ''' 获取数据集大小 '''
    def getSize(self):
        return self.__dataLen


    ''' 输出展示 '''
    @staticmethod
    def echo(msg, crlf=True):
        if crlf:
            print msg
        else:
            sys.stdout.write(msg)
            sys.stdout.flush()


# Download.run()

# train_data = Data(0.6, 0.8)
# batch_x , batch_y = train_data.next_batch(10)

