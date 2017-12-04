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


def echo(msg, crlf=True):
    if crlf:
        print msg
    else:
        sys.stdout.write(msg)
        sys.stdout.flush()

echo('Loading data ...')
file_list = os.listdir(data_dir)
file_len = len(file_list)

for i, file_name in enumerate(file_list):
    progress = float(i + 1) / file_len * 100
    echo('\r >> progress: %.2f%% \t' % progress, False)

    split_file_name = os.path.splitext(file_name)
    no_list = split_file_name[0].split('_')

    if split_file_name[1].lower() != '.jpg' or int(no_list[-1]) > 0:
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

echo('Finish loading')

ratio_list = np.array(ratio_list)

print 'mean:'
print np.mean(ratio_list)
print 'max:'
print max(ratio_list)
print 'min'
print min(ratio_list)
print 'std'
print np.std(ratio_list)

