#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.split(__file__)[0]
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)

from PIL import Image
import numpy as np

data_dir = r'data/TrainImgMore'

ratio_list = []

for file_name in os.listdir(data_dir):
    if os.path.splitext(file_name)[1].lower() != '.jpg':
        continue

    img_path = os.path.join(data_dir, file_name)
    image = Image.open(img_path)
    np_image = np.array(image)

    h, w, c = np_image.shape

    ratio_list.append(float(w) / h)
    # if float(h) / w >= 1:
    #     ratio_list.append( float(w) / h )
    # else:
    #     ratio_list.append( float(h) / w )

ratio_list = np.array(ratio_list)

print 'mean:'
print np.mean(ratio_list)
print 'max:'
print max(ratio_list)
print 'min'
print min(ratio_list)
print 'std'
print np.std(ratio_list)

