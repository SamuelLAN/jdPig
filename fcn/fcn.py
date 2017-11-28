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
    EPOCH_TIMES = 100    # 随机梯度下降的 batch 大小

    IMAGE_SHAPE = [320, 180]
    IMAGE_PIXELS = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
    NUM_CHANNEL = 3     # 输入图片为 3 通道，彩色
    NUM_CLASSES = 2     # 输出的类别

    BASE_LEARNING_RATE = 0.01  # 初始 学习率
    DECAY_RATE = 0.1    # 学习率 的 下降速率

    REGULAR_BETA = 0.01 # 正则化的 beta 参数
    KEEP_PROB = 0.85    # dropout 的 keep_prob

    MAX_VAL_LOSS_INCR_TIMES = 100   # 校验集 val_loss 连续 100 次没有降低，则 early stop

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
        self.__mask = tf.placeholder(tf.float32, [None, None, None, self.NUM_CLASSES], name='y')

        self.__keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # # 用于预测
        # self.__preX = tf.placeholder(tf.float32, [None, self.IMAGE_PIXELS], name='preX')
        # self.__preY = tf.placeholder(tf.float32, [None, self.NUM_CLASSES], name='preY')
        # self.__preSize = tf.placeholder(tf.float32, name='preSize')

        # 随训练次数增多而衰减的学习率
        self.__learning_rate = self.get_learning_rate(
            self.BASE_LEARNING_RATE, self.globalStep, self.__steps / 10, self.DECAY_RATE, staircase=True
        )

    ''' 加载数据 '''

    def load(self):
        self.__train_set = load.Data(0.0, 0.64, 'train')
        self.__val_set = load.Data(0.64, 0.8, 'validation')
        self.__test_set = load.Data(0.8, 1.0, 'test')

        self.__train_size = self.__train_set.getSize()
        self.__val_size = self.__val_set.getSize()
        self.__test_size = self.__test_set.getSize()

    ''' 模型 '''

    def model(self):
        self.__output = self.deep_model(self.__image, self.__keep_prob)
        self.__output_mask = tf.argmax(self.__output, axis=3, name="output_mask")

    ''' 计算 loss '''

    def get_loss(self):
        with tf.name_scope('loss'):
            logits = tf.to_float( tf.reshape(self.__output, [-1, self.NUM_CLASSES]), name='logits' )
            labels = tf.to_float( tf.reshape(self.__mask, [-1, self.NUM_CLASSES]), name='labels' )

            self.__loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=labels, name='entropy')
            )

    ''' 获取 train_op '''

    def get_train_op(self, loss, learning_rate, global_step):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            return optimizer.minimize(loss, global_step=global_step)

    ''' 将图片输出到 tensorboard '''

    def __summary(self):
        with tf.name_scope('summary'):
            mask = tf.argmax(self.__mask, axis=3)

            # 转换 mask 和 output_mask 的 shape 成为 列向量
            mask = tf.to_float(tf.expand_dims(mask, dim=3), name='truth_mask')
            output_mask = tf.to_float(tf.expand_dims(self.__output_mask, dim=3), name='output_mask')

            # 输出图片到 tensorboard
            tf.summary.image('input_image', self.__image, max_outputs=2)
            tf.summary.image('truth_mask', tf.cast(mask * self.__image, tf.uint8), max_outputs=2)
            tf.summary.image('output_image', tf.cast(output_mask * self.__image, tf.uint8), max_outputs=2)

            # 记录 loss 到 tensorboard
            self.__loss_placeholder = tf.placeholder(tf.float32, name='loss')
            tf.summary.scalar('loss', self.__loss_placeholder)

    ''' 测量数据集的 loss '''

    def __measure_loss(self, data_set):
        mean_loss = 0
        count = 0
        batch_x, batch_y = data_set.next_batch(self.BATCH_SIZE, False)
        while batch_x and batch_y:
            feed_dict = {self.__image: batch_x, self.__mask: batch_y, self.__keep_prob: 1.0}
            loss = self.sess.run(self.__loss, feed_dict)
            mean_loss += loss
            batch_x, batch_y = data_set.next_batch(self.BATCH_SIZE, False)
            count += 1

        return mean_loss / count

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

        best_val_loss = 999999          # 校验集 loss 最好的情况
        increase_val_loss_times = 0     # 校验集 loss 连续上升次数
        mean_loss = 0

        self.echo('\nepoch:')
        
        for step in range(self.__steps):
            if step % 5 == 0:
                epoch_progress = float(step) % self.__iter_per_epoch / self.__iter_per_epoch * 100.0
                step_progress = float(step) / self.__steps * 100.0
                self.echo('\rstep: %d (%d|%.2f%%) / %d|%.2f%% \t' % (step, self.__iter_per_epoch, epoch_progress,
                                                                       self.__steps, step_progress), False)

            batch_x, batch_y = self.__train_set.next_batch(self.BATCH_SIZE)
            feed_dict = {self.__image: batch_x, self.__mask:batch_y, self.__keep_prob: self.KEEP_PROB}
            _, train_loss = self.sess.run([train_op, self.__loss], feed_dict)

            mean_loss += train_loss

            if step % self.__iter_per_epoch == 0 and step != 0:
                epoch = step // self.__iter_per_epoch

                feed_dict[self.__loss_placeholder] = mean_loss / self.__iter_per_epoch
                mean_loss = 0
                self.add_summary_train(feed_dict, epoch)

                # 测试 校验集 的 loss
                mean_val_loss = self.__measure_loss(self.__val_set)
                feed_dict[self.__loss_placeholder] = mean_val_loss
                self.add_summary_val(feed_dict, epoch)

                if best_val_loss > mean_val_loss:
                    best_val_loss = mean_val_loss
                    increase_val_loss_times = 0

                    self.save_model()   # 保存模型

                else:
                    increase_val_loss_times += 1
                    if increase_val_loss_times > self.MAX_VAL_LOSS_INCR_TIMES:
                        break

        self.close_summary()  # 关闭 TensorBoard

        self.restore_model()  # 恢复模型

        train_loss = self.__measure_loss(self.__train_set)
        val_loss = self.__measure_loss(self.__val_set)
        test_loss = self.__measure_loss(self.__test_set)

        self.echo('\ntrain mean loss: %.6f' % train_loss)
        self.echo('validation mean loss: %.6f' % val_loss)
        self.echo('test mean loss: %.6f' % test_loss)

        self.echo('\ndone')



o_fcn = FCN()
o_fcn.run()

