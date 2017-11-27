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
import random
import numpy as np


class Patch:
    IMG_PATH = r'Data/TrainImg'
    PATCH_PATH = r'Data/TrainPatch'
    PATCH_PER_IMG = 20

    # 每个 ratio: 长 w, 宽 h, n 原图宽的 n 分之一
    RATIO_LIST = [[39, 31, 3], [31, 31, 3], [31, 39, 3], [31, 31, 3], [31, 31, 2], [31, 39, 2], [39, 31, 2], [31, 31, 1.5], [31, 39, 1.5], [39, 31, 1.5]]

    MIN_PIG_RATIO = 0.3         # 猪在图片至少占的比例
    MIN_RATIO_N = 3             # patch 的宽 至少是 原图的宽的 n 分之一
    MIN_CENTER_DIS = 10         # 不同 patch 中心之间的最小间距

    MAX_TRY_TIMES = 200

    def __init__(self):
        self.__imgList = []
        self.__alreadyList = {}
        self.__progressIndex = 0
        self.__progressLen = 0


    ''' 检查文件夹已经存在的 patch ，避免重复生成 '''
    def __getAlreadyExistList(self):
        already_list = {}
        for file_name in os.listdir(self.PATCH_PATH):
            if os.path.splitext(file_name)[1].lower() != '.jpg':
                continue
            file_name = os.path.splitext(file_name)[0]
            file_no = file_name.split('_')

            img_name = '%s_%s.jpg' % (file_no[0], file_no[1])
            if img_name not in already_list:
                already_list[img_name] = 0
            already_list[img_name] += 1

            if already_list[img_name] >= self.PATCH_PER_IMG:
                self.__alreadyList[img_name] = True


    ''' 获取图片列表 '''
    def __getImgList(self):
        only_list = []

        for file_name in os.listdir(self.IMG_PATH):
            if os.path.splitext(file_name)[1].lower() != '.jpg' or file_name in self.__alreadyList:
                continue

            if only_list:
                if file_name not in only_list:
                    continue

            self.__imgList.append( os.path.join(self.IMG_PATH, file_name) )

        self.__progressLen = len(self.__imgList) * self.PATCH_PER_IMG


    ''' 根据 ratio 获取 im 的 patch '''
    @staticmethod
    def __getPatch(im, org_w, org_h, ratio):
        w, h, n = ratio
        patch_w = org_w / n
        w_times = patch_w / w
        w *= w_times
        h *= w_times
        w_half = int(0.5 * w)
        h_half = int(0.5 * h)

        # patch 的中心
        w_center = random.randrange(w_half, org_w - w_half)
        h_center = random.randrange(h_half, org_h - h_half)

        return im[w_center - w_half: w_center + w_half, h_center - h_half: h_center + h_half, :], np.array([w_center, h_center])


    def __img2PatchGreedy(self):
        pass


    ''' 将 img 转化为 patch '''
    def __img2Patch(self, img_path):
        im_name = os.path.splitext(os.path.split(img_path)[1])[0]
        im = cv2.imread(img_path)                   # 读取图片
        org_w, org_h, c = im.shape

        ratio_len = len(self.RATIO_LIST)

        centers = []

        def __checkCenter(_center, _ratio, _try_times, max_try_times):
            if _try_times > max_try_times:
                print '%s has problem ' % im_name
                return True

            w, h, n = _ratio
            n_list = [n, self.MIN_RATIO_N]
            patch_w = org_w / n_list[random.randint(0, 1)]
            w_times = patch_w / w
            dis = self.MIN_CENTER_DIS * w_times
            for _c in centers:
                if np.sqrt( sum( pow(_c - _center, 2) ) ) < dis:
                    return False
            return True

        for i in range(self.PATCH_PER_IMG):
            ratio = self.RATIO_LIST[random.randint(0, ratio_len - 1)]
            patch, center = self.__getPatch(im, org_w, org_h, ratio)

            try_times = 0
            while self.calPigRatio(patch) < 0.35 or not __checkCenter(center, ratio, try_times, self.MAX_TRY_TIMES):
                try_times += 1
                ratio = self.RATIO_LIST[random.randint(0, ratio_len - 1)]
                patch, center = self.__getPatch(im, org_w, org_h, ratio)

            if try_times > self.MAX_TRY_TIMES:
                continue

            centers.append(center)
            self.__calProgress(im_name, i)

            cv2.imwrite(os.path.join(self.PATCH_PATH, '%s_%d.jpg' % (im_name, i)), patch)


    @staticmethod
    def calPigRatio(im):
        w, h, c = im.shape
        im_size = w * h

        real_size = 0
        for i, val_i in enumerate(im):
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
                        or g > 225 or b > 215 or g / b > 1.15 or g / b < 0.8 \
                        or r / g > 3.1 or 2 > r / g > 1.35 or r / g < 1.14:
                    pass
                else:
                    real_size += 1

        return float(real_size) / im_size


    ''' 获取进度 '''
    def __calProgress(self, im_name, patch_index):
        self.__progressIndex += 1
        progress = float(self.__progressIndex) / self.__progressLen * 100
        self.echo('progress: %.2f%% \t img_name: %s \t patch: %d \t \r' % (progress, im_name, patch_index), False)


    ''' 输出展示 '''
    @staticmethod
    def echo(msg, crlf=True):
        if crlf:
            print msg
        else:
            sys.stdout.write(msg)
            sys.stdout.flush()


    ''' 主函数 '''
    def run(self):
        self.__getAlreadyExistList()
        self.__getImgList()

        for i, img_path in enumerate(self.__imgList):
            self.__img2Patch(img_path)

        self.echo('\ndone')


o_patch = Patch()
o_patch.run()
