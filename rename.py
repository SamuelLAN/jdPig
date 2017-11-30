#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.split(__file__)[0]
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)

data_dir = r'/Users/samuellin/Documents/GitHub/jdPig/fcn/data'

img_no_set = set()

for file_name in os.listdir(data_dir):
    if os.path.splitext(file_name)[1].lower() != '.jpg':
        continue

    img_no = int(file_name.split('_')[0])
    img_no_set.add(img_no)

img_no_list = list(img_no_set)
max_img_no = max(img_no_list) + 1

rename_list = []
cp_name = ''

def cmd(command):
    return os.popen(command).read()

for file_name in os.listdir(data_dir):
    if os.path.splitext(file_name)[1].lower() != '.jpg':
        continue

    ga = os.path.splitext(file_name)[0].lower().split('_')
    if len(ga) > 2:
        rename_list.append(file_name)

        img_no = int(ga[2])
        if img_no == 0:
            cp_name = file_name
            cmd('cp %s %s' % ( os.path.join(data_dir, file_name), os.path.join(data_dir, str(max_img_no) + '_mask.jpg') ))

for i, file_name in enumerate(rename_list):
    file_path = os.path.join(data_dir, file_name)
    mv_path = os.path.join(data_dir, '%d_%d.jpg' % (max_img_no, i + 1))
    cmd('mv %s %s' % (file_path, mv_path))

print 'done'
import time
print time.time()




