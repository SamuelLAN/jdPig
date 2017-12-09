#!/usr/bin/Python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import numpy as np
from six.moves.urllib.request import urlretrieve


''' VGG 模型 (16层版) '''
class VGG:
    MODEL_DIR = r'model'
    MODEL_16 = r'model/vgg16.npy'
    MODEL_19 = r'model/vgg19.npy'
    MODEL_URL = r'http://www.lin-baobao.com/model/vgg16.npy'

    def __init__(self):
        pass


    ''' 加载模型 '''
    @staticmethod
    def load(model_19=False):
        '''
        Returns:
            vgg_mode (dict)
        '''
        if not os.path.isdir(VGG.MODEL_DIR):
            os.mkdir(VGG.MODEL_DIR)
        model = VGG.MODEL_16 if not model_19 else VGG.MODEL_19
        if not os.path.isfile(model):
            print ('Start downloading %s' % model)
            file_path, _ = urlretrieve(VGG.MODEL_URL, model, reporthook=VGG.__download_progress)
            stat_info = os.stat(file_path)
            print ('\nSuccesfully downloaded %s %d bytes' % (VGG.MODEL_URL, stat_info.st_size))
        return np.load(model, encoding='latin1').item()


    ''' 下载的进度 '''
    @staticmethod
    def __download_progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (os.path.split(VGG.MODEL)[1],
                                                         float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()


# model = VGG.load()
# print model.keys()
# print model['conv5_3'][0].shape
# print model['conv5_3'][1].shape
# print model['fc6'][0].shape
# print model['fc6'][1].shape
# print model['fc7'][0].shape
# print model['fc7'][1].shape
# print model['fc8'][0].shape
# print model['fc8'][1].shape
