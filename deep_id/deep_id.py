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
import tensorflow as tf


'''
    原图
    框猪图(带背景)
    
    
    
    padding 
    55 * 31
'''
class DeepId(base.NN):
    MODEL_NAME = 'deep_id'      # 模型的名称

    BATCH_SIZE = 4              # 迭代的 epoch 次数
    EPOCH_TIMES = 100           # 随机梯度下降的 batch 大小

    IMAGE_SHAPE = [39, 39]

    NUM_CLASSES = 30
    NUM_CHANNEL = 3

    BASE_LEARNING_RATE = 0.01   # 初始 学习率
    DECAY_RATE = 0.1            # 学习率 的 下降速率

    DEEP_ID_LAYER_INDEX = -2    # 倒数第二层为 deep_id 层

    SHOW_PROGRESS_FREQUENCY = 2  # 每 SHOW_PROGRESS_FREQUENCY 个 step show 一次进度 progress

    MODEL = [
        {   # 39 * 39 => 36 * 36
            'name': 'conv_1',
            'type': 'conv',
            'shape': [NUM_CHANNEL, 20],
            'k_size': [4, 4],
            'padding': 'VALID',
        },
        {   # 36 * 36 => 18 * 18
            'name': 'pool_1',
            'type': 'pool',
            'k_size': [2, 2],
        },
        {   # 18 * 18 => 16 * 16
            'name': 'conv_2',
            'type': 'conv',
            'shape': [20, 40],
            'k_size': [3, 3],
            'padding': 'VALID',
        },
        {   # 16 * 16 => 8 * 8
            'name': 'pool_2',
            'type': 'pool',
            'k_size': [2, 2],
        },
        {   # 8 * 8 => 6 * 6
            'name': 'conv_3',
            'type': 'pool_3',
            'shape': [40, 60],
            'k_size': [3, 3],
            'padding': 'VALID',
        },
        {   # 6 * 6 => 3 * 3
            'name': 'pool_3',
            'type': 'pool',
            'k_size': [2, 2],
        },
        {   # 3 * 3 => 2 * 2
            'name': 'conv_4',
            'type': 'conv',
            'shape': [60, 80],
            'k_size': [2, 2],
            'padding': 'VALID',
        },
        {   # 2 * 2 * 80 => 320 ; 与前一层全连接
            'name': 'fc_4',
            'type': 'fc',
            'shape': [320, 320],
        },
        {   # 3 * 3 * 60 => 320 与 pool_3 层全连接
            'name': 'fc_3',
            'type': 'fc_n',
            'shape': [540, 320],
            'layer_index': 5,
        },
        {   # 8 * 8 * 40 => 320 与 pool_2 层全连接
            'name': 'fc_2',
            'type': 'fc_n',
            'shape': [2560, 320],
            'layer_index': 3,
        },
        {   # 18 * 18 * 20 => 320 与 pool_1 层全连接
            'name': 'fc_1',
            'type': 'fc_n',
            'shape': [6480, 320],
            'layer_index': 1,
        },
        {   # softmax 层
            'name': 'softmax',
            'type': 'fc',
            'shape': [320, NUM_CLASSES],
        }
    ]


    ''' 自定义 初始化变量 过程 '''
    def init(self):
        # 加载数据
        self.load()

        # 常量
        self.__iter_per_epoch = int(self.__train_size // self.BATCH_SIZE)
        self.__steps = self.EPOCH_TIMES * self.__iter_per_epoch

        # 输入 与 label
        self.__image = tf.placeholder(tf.float32, [None, self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1],
                                                   self.NUM_CHANNEL], name='X')
        self.__label = tf.placeholder(tf.float32, [None, self.NUM_CLASSES], name='y')
        # dropout 的 keep_prob
        self.__keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # 随训练次数增多而衰减的学习率
        self.__learning_rate = self.get_learning_rate(
            self.BASE_LEARNING_RATE, self.global_step, self.__steps, self.DECAY_RATE, staircase=False
        )

        self.__has_rebuild = False


    ''' 加载数据 '''
    def load(self):
        pass
        # sort_list = load.Data.get_sort_list()
        # self.__train_set = load.Data(0.0, 0.9, 'train', sort_list)
        # self.__val_set = load.Data(0.9, 1.0, 'validation', sort_list)
        # # self.__test_set = load.Data(0.8, 1.0, 'test', sort_list)
        #
        # self.__train_size = self.__train_set.get_size()
        # self.__val_size = self.__val_set.get_size()
        # # self.__test_size = self.__test_set.get_size()


    ''' 模型 '''
    def model(self):
        self.__output = self.deep_model(self.__image, self.__keep_prob)


    ''' 重建模型 '''
    def rebuild_model(self):
        self.__output = self.deep_model_rebuild(self.__image)


    ''' 计算 loss '''
    def get_loss(self):
        with tf.name_scope('loss'):
            self.__loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.__output, labels=self.__label, name='entropy')
            )


    ''' 获取 train_op '''
    def get_train_op(self, loss, learning_rate, global_step):
        tf.summary.scalar('loss', loss)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            return optimizer.minimize(loss, global_step=global_step)


    ''' 将图片输出到 tensorboard '''
    def __summary(self):
        with tf.name_scope('summary'):
            # 记录 loss 到 tensorboard
            self.__loss_placeholder = tf.placeholder(tf.float32, name='loss')
            tf.summary.scalar('mean_loss', self.__loss_placeholder)


    ''' 主函数 '''
    def run(self):
        # 生成模型
        self.model()

        # 计算 loss
        self.get_loss()

        # 正则化
        # self.__loss = self.regularize_trainable(self.__loss, self.REGULAR_BETA)

        # 生成训练的 op
        train_op = self.get_train_op(self.__loss, self.__learning_rate, self.global_step)

        # # tensorboard 相关记录
        # self.__summary()

        # 初始化所有变量
        self.init_variables()

        # TensorBoard merge summary
        self.merge_summary()

        self.echo('\nepoch:')

        for step in range(self.__steps):
            if step % self.SHOW_PROGRESS_FREQUENCY == 0:
                epoch_progress = float(step) % self.__iter_per_epoch / self.__iter_per_epoch * 100.0
                step_progress = float(step) / self.__steps * 100.0
                self.echo('\rstep: %d (%d|%.2f%%) / %d|%.2f%% \t\t' % (step, self.__iter_per_epoch, epoch_progress,
                                                                       self.__steps, step_progress), False)


