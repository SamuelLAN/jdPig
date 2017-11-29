#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.split(__file__)[0]
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)

import numpy as np
from PIL import Image

img = Image.open(r'/Users/samuellin/Downloads/individualImage.png')
img.show()

np_img_color = np.array(img)

img = img.convert('L')

np_img = np.array(img)
# np_img[np_img > 0] = 1

h, w = np_img.shape

data = []
for i in range(h):
    for j in range(w):
        if np_img[i, j] != 0:
            data.append([i, j])

data = np.array(data)
center = np.cast['uint8'](np.mean(data, axis=0))

import Queue

s = set()

q = Queue.Queue()
q.put(center)

while not q.empty():
    x, y = q.get()

    if x + 1 < h:
        c = (x + 1, y)
        if np_img[c[0], c[1]] != 0:
            if c not in s:
                s.add(c)
                q.put(c)

    if x - 1 >= 0:
        c = (x - 1, y)
        if np_img[c[0], c[1]] != 0:
            if c not in s:
                s.add(c)
                q.put(c)

    if y - 1 >= 0:
        c = (x, y - 1)
        if np_img[c[0], c[1]] != 0:
            if c not in s:
                s.add(c)
                q.put(c)

    if y + 1 < w:
        c = (x, y + 1)
        if np_img[c[0], c[1]] != 0:
            if c not in s:
                s.add(c)
                q.put(c)


new_mask = np.zeros_like(np_img)

for c in s:
    new_mask[c[0], c[1]] = 1

new_mask = np.expand_dims(new_mask, axis=2)
new_img = np.cast['uint8'](new_mask * np_img_color)

o_new_img = Image.fromarray(new_img)
o_new_img.show()


