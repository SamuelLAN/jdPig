#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.split(__file__)[0]
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)

import base
import load
import vgg
import tensorflow as tf


''' 全卷积神经网络 '''
class FCN(base.NN):
    MODEL_NAME = 'fcn'  # 模型的名称

    BATCH_SIZE = 12     # 迭代的 epoch 次数
    EPOCH_TIMES = 1    # 随机梯度下降的 batch 大小

    IMAGE_SHAPE = [320, 180]
    IMAGE_PIXELS = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
    NUM_CHANNEL = 3     # 输入图片为 3 通道，彩色
    NUM_CLASSES = 1     # 输出的类别

    BASE_LEARNING_RATE = 0.01  # 初始 学习率
    DECAY_RATE = 0.9    # 学习率 的 下降速率

    REGULAR_BETA = 0.01 # 正则化的 beta 参数
    KEEP_PROB = 0.85    # dropout 的 keep_prob

    TENSORBOARD_SHOW_IMAGE = False  # 默认不将 image 显示到 TensorBoard，以免影响性能

    VGG_MODEL = vgg.VGG.load()  # 加载 VGG 模型

    ''' 模型的配置；采用了 VGG16 模型的 FCN '''
    MODEL = [
        {
            'name': 'conv1_1',
            'type': 'conv',
            'W': VGG_MODEL['conv1_1'][0],
            'b': VGG_MODEL['conv1_1'][1],
        },
        {
            'name': 'conv1_2',
            'type': 'conv',
            'W': VGG_MODEL['conv1_2'][0],
            'b': VGG_MODEL['conv1_2'][1],
        },
        {
            'name': 'pool_1',
            'type': 'pool',
            'k_size': [2, 2],
            'pool_type': 'avg',
        },
        {
            'name': 'conv2_1',
            'type': 'conv',
            'W': VGG_MODEL['conv2_1'][0],
            'b': VGG_MODEL['conv2_1'][1],
        },
        {
            'name': 'conv2_2',
            'type': 'conv',
            'W': VGG_MODEL['conv2_2'][0],
            'b': VGG_MODEL['conv2_2'][1],
        },
        {
            'name': 'pool_2',
            'type': 'pool',
            'k_size': [2, 2],
            'pool_type': 'avg',
        },
        {
            'name': 'conv3_1',
            'type': 'conv',
            'W': VGG_MODEL['conv3_1'][0],
            'b': VGG_MODEL['conv3_1'][1],
        },
        {
            'name': 'conv3_2',
            'type': 'conv',
            'W': VGG_MODEL['conv3_2'][0],
            'b': VGG_MODEL['conv3_2'][1],
        },
        {
            'name': 'conv3_3',
            'type': 'conv',
            'W': VGG_MODEL['conv3_3'][0],
            'b': VGG_MODEL['conv3_3'][1],
        },
        {
            'name': 'pool_3',
            'type': 'pool',
            'k_size': [2, 2],
            'pool_type': 'avg',
        },
        {
            'name': 'conv4_1',
            'type': 'conv',
            'W': VGG_MODEL['conv4_1'][0],
            'b': VGG_MODEL['conv4_1'][1],
        },
        {
            'name': 'conv4_2',
            'type': 'conv',
            'W': VGG_MODEL['conv4_2'][0],
            'b': VGG_MODEL['conv4_2'][1],
        },
        {
            'name': 'conv4_3',
            'type': 'conv',
            'W': VGG_MODEL['conv4_3'][0],
            'b': VGG_MODEL['conv4_3'][1],
        },
        {
            'name': 'pool_4',
            'type': 'pool',
            'k_size': [2, 2],
            'pool_type': 'avg',
        },
        {
            'name': 'conv5_1',
            'type': 'conv',
            'W': VGG_MODEL['conv5_1'][0],
            'b': VGG_MODEL['conv5_1'][1],
        },
        {
            'name': 'conv5_2',
            'type': 'conv',
            'W': VGG_MODEL['conv5_2'][0],
            'b': VGG_MODEL['conv5_2'][1],
        },
        {
            'name': 'conv5_3',
            'type': 'conv',
            'W': VGG_MODEL['conv5_3'][0],
            'b': VGG_MODEL['conv5_3'][1],
        },
        {
            'name': 'pool_5',
            'type': 'pool',
            'k_size': [2, 2],
            'pool_type': 'max',
        },
        {
            'name': 'conv6',
            'type': 'conv',
            'shape': [VGG_MODEL['conv5_3'][0].shape[3], 4096],
            'k_size': [7, 7],
        },
        {
            'name': 'dropout_6',
            'type': 'dropout',
        },
        {
            'name': 'conv7',
            'type': 'conv',
            'shape': [4096, 4096],
            'k_size': [1, 1],
        },
        {
            'name': 'dropout_7',
            'type': 'dropout',
        },
        {
            'name': 'conv8',
            'type': 'conv',
            'shape': [4096, NUM_CLASSES],
            'k_size': [1, 1],
            'activate': False,
        },
        {
            'name': 'tr_conv_1',
            'type': 'tr_conv',
            'shape': [VGG_MODEL['conv4_3'][0].shape[3], NUM_CLASSES],   # 对应 [ pool_4 层的 channel, NUM_CLASSES ]
            'k_size': [4, 4],
            'output_shape_index': 13,   # 对应 pool_4 层的 shape
        },
        {
            'name': 'add_1',
            'type': 'add',
            'layer_index': 13,          # 对应 pool_4 层
        },
        {
            'name': 'tr_conv_2',
            'type': 'tr_conv',
            'shape': [VGG_MODEL['conv3_3'][0].shape[3], VGG_MODEL['conv4_3'][0].shape[3]],  # 对应 [ pool_3 层的 channel, pool_4 层的 channel ]
            'k_size': [4, 4],
            'output_shape_index': 9,    # 对应 pool_3 层的 shape
        },
        {
            'name': 'add_2',
            'type': 'add',
            'layer_index': 9,           # 对应 pool_3 层
        },
        {
            'name': 'tr_conv_3',
            'type': 'tr_conv',
            'shape': [NUM_CLASSES, VGG_MODEL['conv3_3'][0].shape[3]],
            'k_size': [16, 16],
            'stride': 8,
            # 'output_shape': [12, 360, 640, NUM_CLASSES],  # 对应输入层的 shape
            'output_shape_x': [None, None, None, NUM_CLASSES],  # 对应输入层的 shape
        },
    ]

    ''' 自定义 初始化变量 过程 '''

    def init(self):
        # 加载数据
        self.load()

        # 常量
        self.__iter_per_epoch = int(self.__train_size // self.BATCH_SIZE)
        self.__steps = self.EPOCH_TIMES * self.__iter_per_epoch

        # 输入 与 label
        self.__image = tf.placeholder(tf.float32, [None, None, None, self.NUM_CHANNEL], name='X')
        self.__mask = tf.placeholder(tf.float32, [None, None, None, 1], name='y')

        self.__keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # # 用于预测
        # self.__preX = tf.placeholder(tf.float32, [None, self.IMAGE_PIXELS], name='preX')
        # self.__preY = tf.placeholder(tf.float32, [None, self.NUM_CLASSES], name='preY')
        # self.__preSize = tf.placeholder(tf.float32, name='preSize')

        # 随训练次数增多而衰减的学习率
        self.__learning_rate = self.get_learning_rate(
            self.BASE_LEARNING_RATE, self.globalStep, self.__steps, self.DECAY_RATE
        )

    ''' 加载数据 '''

    def load(self):
        self.__train_set = load.Data(0.0, 0.64, 'train')
        # self.__val_set = load.Data(0.64, 0.8, 'validation')
        # self.__test_set = load.Data(0.8, 1.0, 'test')

        self.__train_size = self.__train_set.getSize()
        # self.__val_size = self.__val_set.getSize()
        # self.__test_size = self.__test_set.getSize()

    ''' 模型 '''

    def model(self):
        self.__output = self.deep_model(self.__image, self.__keep_prob)
        with tf.name_scope('process_output'):
            self.__output_mask = tf.round(self.__output, name='output_mask')

    ''' 计算 loss '''

    def get_loss(self):
        with tf.name_scope('loss'):
            labels = tf.to_float(tf.reshape(self.__mask, (-1, self.NUM_CLASSES)), name='labels')
            self.__loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.__output,
                                                        labels=labels, name='entropy')
            )

    ''' 获取 train_op '''

    def get_train_op(self, loss, learning_rate, global_step):
        tf.summary.scalar('loss', loss)  # TensorBoard 记录 loss

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            return optimizer.minimize(loss, global_step=global_step)

    ''' 将图片输出到 tensorboard '''

    def __summary(self):
        with tf.name_scope('summary'):
            tf.summary.image('input_image', self.__image, max_outputs=2)                    # 输入图片
            tf.summary.image('mask', tf.cast(self.__mask * self.__image, tf.uint8), max_outputs=2)             # mask (ground truth)
            tf.summary.image('output_image', tf.cast(self.__output_mask * self.__image, tf.uint8), max_outputs=2)   # 输出图片

    ''' 主函数 '''

    def run(self):
        # 生成模型
        self.model()

        # 计算 loss
        self.get_loss()

        # 正则化
        # self.__loss = self.regularize(self.__loss, self.REGULAR_BETA)

        # 生成训练的 op
        train_op = self.get_train_op(self.__loss, self.__learning_rate, self.globalStep)

        # tensorboard 相关记录
        self.__summary()

        # 初始化所有变量
        self.init_variables()

        # TensorBoard merge summary
        self.merge_summary()
        
        self.echo('\nepoch:')
        
        for step in range(self.__steps):
            if step % 2 == 0:
                epoch_progress = float(step) % self.__iter_per_epoch / self.__iter_per_epoch * 100.0
                step_progress = float(step) / self.__steps * 100.0
                self.echo('\rstep: %d (%d|%.2f%%) / %d|%.2f%% \t' % (step, self.__iter_per_epoch, epoch_progress,
                                                                       self.__steps, step_progress), False)

            batch_x, batch_y = self.__train_set.next_batch(self.BATCH_SIZE)
            feed_dict = {self.__image: batch_x, self.__mask:batch_y, self.__keep_prob: self.KEEP_PROB}
            self.sess.run(train_op, feed_dict)

            if step % self.__iter_per_epoch == 0 and step != 0:
                epoch = step // self.__iter_per_epoch
                self.add_summary_train(feed_dict, epoch)

            if step == self.__steps - 1:
                output, image, output_mask = self.sess.run([self.__output, self.__image, self.__output_mask], feed_dict=feed_dict)
                # output_mask = self.sess.run(self.__output_mask, feed_dict=feed_dict)

                # print 'output_mask shape:'
                # print output_mask.shape

                print '\n************ output *******************'
                print output[0]
                print ''

                print '\n************ output *******************'
                print output_mask[0]
                print ''

                # tmp_mask = output_mask[0]
                # # tmp_mask
                # # tmp_mask[tmp_mask > 0] = 255
                # from PIL import Image
                # import numpy as np
                #
                # print 'tmp_mask:'
                # tmp_mask = np.cast['uint8'](tmp_mask)
                # tmp_mask = tmp_mask.reshape(tmp_mask.shape[:2])
                #
                # print tmp_mask.shape
                # print type(tmp_mask)
                #
                # tmp_mask_img = Image.fromarray(tmp_mask)
                # tmp_mask_img.show()


        self.close_summary()  # 关闭 TensorBoard

        # self.restore_model()  # 恢复模型


o_fcn = FCN()
o_fcn.run()

