#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.split(__file__)[0]
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)

import math
import random
import numpy as np
from PIL import Image


'''
 9 3 * 3
 4 2 * 2
 1 2 * 2 center
 7 random
'''
class Patch:
    IMG_PATH = r'data/TrainImgMore'
    PATCH_PATH = r'data/TrainPatch'
    PATCH_PER_IMG = 20

    # 每个 ratio: 长 w, 宽 h, n 原图宽的 n 分之一
    RATIO_LIST = [[39, 31], [31, 31], [31, 39], [31, 31]]
    SCALE_LIST = [3, 2.5, 2, 1.5]

    MIN_CENTER_DIS = 10         # 不同 patch 中心之间的最小间距

    MAX_TRY_TIMES = 200

    def __init__(self):
        self.__img_list = []
        self.__already_list = {}
        self.__progress_len = 0


    def __check_dir(self):
        if not os.path.isdir(self.IMG_PATH):
            os.mkdir(self.IMG_PATH)

        if not os.path.isdir(self.PATCH_PATH):
            os.mkdir(self.PATCH_PATH)


    ''' 检查文件夹已经存在的 patch ，避免重复生成 '''
    def __get_already_exist_list(self):
        already_list = {}
        for file_name in os.listdir(self.PATCH_PATH):
            split_file_name = os.path.splitext(file_name)
            if split_file_name[1].lower() != '.jpg':
                continue
            file_name = split_file_name[0]
            file_no = file_name.split('_')

            img_name = '%s_%s_%s.jpg' % (file_no[0], file_no[1], file_no[2])
            if img_name not in already_list:
                already_list[img_name] = 0
            already_list[img_name] += 1

            if already_list[img_name] >= self.PATCH_PER_IMG:
                self.__already_list[img_name] = True


    ''' 获取图片列表 '''
    def __get_img_list(self):
        for file_name in os.listdir(self.IMG_PATH):
            split_file_name = os.path.splitext(file_name)
            if split_file_name[1].lower() != '.jpg' \
                    or file_name in self.__already_list:
                continue

            self.__img_list.append( os.path.join(self.IMG_PATH, file_name) )
        self.__progress_len = len(self.__img_list)


    ''' 获取每张图片所需的 patch '''
    def __get_patch(self, img_path):
        im_name = os.path.splitext( os.path.split(img_path)[1] )[0]
        
        image = Image.open(img_path)
        np_image = np.array( image )

        h, w, c = np_image.shape
        ratio_h, ratio_w = self.RATIO_LIST[random.randrange(0, len(self.RATIO_LIST))]

        if h < 2 * ratio_h and w < 2 * ratio_w:
            return

        trans_w = int( float(h) / ratio_h * ratio_w )
        num_trans_w = float(w) / trans_w

        patch_no = 0

        if num_trans_w < 1:
            trans_h = int( float(w) / ratio_w * ratio_h )
            num_trans_h = int( math.ceil( float(h) / trans_h ) )
            
            for i in range(num_trans_h):
                if i < num_trans_h - 1:
                    np_patch = np_image[i * trans_h: (i + 1) * trans_h, :, :]
                else:
                    np_patch = np_image[-trans_h:, :, :]

                patch = Image.fromarray(np_patch).resize([ratio_h, ratio_w])
                patch.save(os.path.join(self.PATCH_PATH, '%s_%d.jpg' % (im_name, patch_no)))
                patch_no += 1
                
                if patch_no >= self.PATCH_PER_IMG:
                    break
            
        else:
            num_trans_w = int( math.ceil(num_trans_w) )
            for i in range(num_trans_w):
                if i < num_trans_w - 1:
                    np_patch = np_image[:, i * trans_w: (i + 1) * trans_w, :]
                else:
                    np_patch = np_image[:, -trans_w:, :]
                
                patch = Image.fromarray( np_patch ).resize([ratio_h, ratio_w])
                patch.save( os.path.join( self.PATCH_PATH, '%s_%d.jpg' % (im_name, patch_no) ) )
                patch_no += 1

                if patch_no >= self.PATCH_PER_IMG:
                    break
        
        if patch_no >= self.PATCH_PER_IMG:
            return

        result, patch_no = self.__get_divide_patch(np_image, 3, im_name, patch_no)
        if result:
            result, patch_no = self.__get_divide_patch(np_image, 2, im_name, patch_no)

        if not result:
            return 

        for i in range(self.PATCH_PER_IMG - patch_no):
            self.__get_random_patch(np_image, im_name, patch_no)
            patch_no += 1 


    ''' 将图片的 w, h 分别等分为 k 份，然后在里面取 patch '''
    def __get_divide_patch(self, np_image, k, im_name, patch_no):
        h, w, c = np_image.shape

        divide_h = int(h / k)
        divide_w = int(w / k)
        for i in range(k):
            if i < k - 1:
                h_start = i * divide_h
            else:
                h_start = h - divide_h
            for j in range(k):
                if j < k - 1:
                    w_start = j * divide_w
                else:
                    w_start = w - divide_w
                np_tmp_image = np_image[h_start: h_start + divide_h, w_start: w_start + divide_w, :]
                self.__get_biggest_patch(np_tmp_image, im_name, patch_no)
                patch_no += 1
                if patch_no >= self.PATCH_PER_IMG:
                    return False, patch_no
        
        return True, patch_no


    ''' 在给定的 np_image 里取 size 尽可能大的 patch '''
    def __get_biggest_patch(self, np_image, im_name, patch_no):
        h, w, c = np_image.shape
        ratio_h, ratio_w = self.RATIO_LIST[random.randrange(0, len(self.RATIO_LIST))]
        
        if float(ratio_h) / ratio_w <= float(h) / w:
            patch_h = int( float(w) / ratio_w * ratio_h )
            patch_h_start = random.randrange(0, h - patch_h) if h - patch_h != 0 else 0
            np_patch = np_image[patch_h_start: patch_h_start + patch_h, :, :]

        else:
            patch_w = int( float(h) / ratio_h * ratio_w )
            patch_w_start = random.randrange(0, w - patch_w) if w - patch_w != 0 else 0
            np_patch = np_image[:, patch_w_start: patch_w_start + patch_w, :]

        patch = Image.fromarray(np_patch).resize([ratio_h, ratio_w])
        patch.save(os.path.join(self.PATCH_PATH, '%s_%d.jpg' % (im_name, patch_no)))
        

    ''' 在给定的 np_iamge 里随机取 patch '''
    def __get_random_patch(self, np_image, im_name, patch_no):
        h, w, c = np_image.shape
        ratio_h, ratio_w = self.RATIO_LIST[random.randrange(0, len(self.RATIO_LIST))]
        scale = self.SCALE_LIST[random.randrange(0, len(self.SCALE_LIST))]
        
        base_w = int(w / scale)
        w_times = int(base_w / ratio_w)
        
        patch_h = int( w_times * ratio_h )
        patch_w = int( w_times * ratio_w )

        if patch_h >= h:
            base_h = int(h / scale)
            h_times = int(base_h / ratio_h)

            patch_h = int( h_times * ratio_h )
            patch_w = int( h_times * ratio_w )

        patch_h_start = random.randrange(0, h - patch_h) if h - patch_h != 0 else 0
        patch_w_start = random.randrange(0, w - patch_w) if w - patch_w != 0 else 0
        
        np_patch = np_image[patch_h_start: patch_h_start + patch_h, patch_w_start: patch_w_start + patch_w, :]

        patch = Image.fromarray(np_patch).resize([ratio_h, ratio_w])
        patch.save(os.path.join(self.PATCH_PATH, '%s_%d.jpg' % (im_name, patch_no)))


    ''' 获取进度 '''
    def __cal_progress(self, i, img_patch):
        progress = float(i + 1) / self.__progress_len * 100
        self.echo('\r >> progress: %.2f%% \t img_patch: %s \t ' % (progress, img_patch), False)


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
        self.__check_dir()

        self.echo('\nGetting already exist list ... ')
        self.__get_already_exist_list()
        self.echo('Finish getting already exist list ')

        self.echo('\nGetting img list ... ')
        self.__get_img_list()
        self.echo('Finish getting img list ')

        self.echo('\nGetting patch ... ')
        for i, img_path in enumerate(self.__img_list):
            self.__cal_progress(i, img_path)        # 输出进度
            self.__get_patch(img_path)              # 获取每张图片的 patch
        self.echo('Finish getting patch ')

        self.echo('\ndone ')


o_patch = Patch()
o_patch.run()
