#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.split(__file__)[0]
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)

# import math
import base2 as base
import load
import vgg
# import Queue
# import numpy as np
# from PIL import Image
import tensorflow as tf

''' 全卷积神经网络 '''


class DeepId(base.NN):
    MODEL_NAME = 'deep_id_2'  # 模型的名称

    BATCH_SIZE = 10 # 迭代的 epoch 次数
    EPOCH_TIMES = 100  # 随机梯度下降的 batch 大小

    NUM_CHANNEL = 3  # 输入图片为 3 通道，彩色
    NUM_CLASSES = 30  # 输出的类别

    IMAGE_SHAPE = [224, 224]
    IMAGE_PIXELS = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
    IMAGE_PH_SHAPE = [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUM_CHANNEL]  # image 的 placeholder 的 shape

    BASE_LEARNING_RATE = 0.01  # 初始 学习率
    DECAY_RATE = 0.05  # 学习率 的 下降速率

    REGULAR_BETA = 0.01  # 正则化的 beta 参数
    KEEP_PROB = 0.5  # dropout 的 keep_prob

    SHOW_PROGRESS_FREQUENCY = 2  # 每 SHOW_PROGRESS_FREQUENCY 个 step show 一次进度 progress

    # MAX_VAL_LOSS_INCR_TIMES = 20  # 校验集 val_loss 连续 100 次没有降低，则 early stop

    TENSORBOARD_SHOW_IMAGE = False  # 默认不将 image 显示到 TensorBoard，以免影响性能

    VGG_MODEL = vgg.VGG.load()  # 加载 VGG 模型

    ''' 模型的配置；采用了 VGG16 模型的 FCN '''
    MODEL = [
        {
            'name': 'conv1_1',
            'type': 'conv',
            'W': VGG_MODEL['conv1_1'][0],
            'b': VGG_MODEL['conv1_1'][1],
            'trainable': False,
        },
        {
            'name': 'conv1_2',
            'type': 'conv',
            'W': VGG_MODEL['conv1_2'][0],
            'b': VGG_MODEL['conv1_2'][1],
            'trainable': False,
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
            'trainable': False,
        },
        {
            'name': 'conv2_2',
            'type': 'conv',
            'W': VGG_MODEL['conv2_2'][0],
            'b': VGG_MODEL['conv2_2'][1],
            'trainable': False,
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
            'trainable': False,
        },
        {
            'name': 'conv3_2',
            'type': 'conv',
            'W': VGG_MODEL['conv3_2'][0],
            'b': VGG_MODEL['conv3_2'][1],
            'trainable': False,
        },
        {
            'name': 'conv3_3',
            'type': 'conv',
            'W': VGG_MODEL['conv3_3'][0],
            'b': VGG_MODEL['conv3_3'][1],
            'trainable': False,
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
            'trainable': False,
        },
        {
            'name': 'conv4_2',
            'type': 'conv',
            'W': VGG_MODEL['conv4_2'][0],
            'b': VGG_MODEL['conv4_2'][1],
            'trainable': False,
        },
        {
            'name': 'conv4_3',
            'type': 'conv',
            'W': VGG_MODEL['conv4_3'][0],
            'b': VGG_MODEL['conv4_3'][1],
            'trainable': False,
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
            'trainable': False,
        },
        {
            'name': 'conv5_2',
            'type': 'conv',
            'W': VGG_MODEL['conv5_2'][0],
            'b': VGG_MODEL['conv5_2'][1],
            'trainable': False,
        },
        {
            'name': 'conv5_3',
            'type': 'conv',
            'W': VGG_MODEL['conv5_3'][0],
            'b': VGG_MODEL['conv5_3'][1],
            'trainable': False,
        },
        {
            'name': 'pool_5',
            'type': 'pool',
            'k_size': [2, 2],
            'pool_type': 'max',
        },
        {
            'name': 'fc6',
            'type': 'fc',
            'W': VGG_MODEL['fc6'][0],
            'b': VGG_MODEL['fc6'][1],
            'trainable': True,
        },
        {
            'name': 'dropout_6',
            'type': 'dropout',
        },
        {
            'name': 'fc7',
            'type': 'fc',
            'W': VGG_MODEL['fc7'][0],
            'b': VGG_MODEL['fc7'][1],
            'trainable': True,
        },
        {
            'name': 'dropout_7',
            'type': 'dropout',
        },
        {
            'name': 'fc8',
            'type': 'fc',
            'W': VGG_MODEL['fc8'][0],
            'b': VGG_MODEL['fc8'][1],
            'trainable': True,
        },
        {
            'name': 'softmax',
            'type': 'fc',
            'shape': [1000, NUM_CLASSES],
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
        self.__image = tf.placeholder(tf.float32, self.IMAGE_PH_SHAPE, name='X')
        self.__label = tf.placeholder(tf.float32, [None, self.NUM_CLASSES], name='y')
        self.__size = tf.placeholder(tf.float32, name='size')

        # dropout 的 keep_prob
        self.__keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # 随训练次数增多而衰减的学习率
        self.__learning_rate = self.get_learning_rate(
            self.BASE_LEARNING_RATE, self.global_step, self.__steps, self.DECAY_RATE, staircase=False
        )

        self.__has_rebuild = False


    ''' 加载数据 '''
    def load(self):
        sort_list = load.Data.get_sort_list()
        self.__train_set = load.Data(0.0, 0.64, 'train', sort_list)
        # self.__val_set = load.Data(0.9, 1.0, 'validation', sort_list)
        # self.__test_set = load.Data(0.8, 1.0, 'test', sort_list)

        self.__train_size = self.__train_set.get_size()
        # self.__val_size = self.__val_set.get_size()
        # self.__test_size = self.__test_set.get_size()


    ''' 模型 '''
    def model(self):
        self.__output = self.deep_model(self.__image, self.__keep_prob)

    # ''' 重建模型 '''
    # def rebuild_model(self):
    #     self.__output = self.deep_model_rebuild(self.__image)


    ''' 计算 loss '''
    def get_loss(self):
        with tf.name_scope('loss'):
            self.__loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.__output, labels=self.__label)
            )


    ''' 获取 train_op '''
    def get_train_op(self, loss, learning_rate, global_step):
        tf.summary.scalar('loss', loss)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            return optimizer.minimize(loss, global_step=global_step)


    # ''' 将图片输出到 tensorboard '''
    # def __summary(self):
    #     with tf.name_scope('summary'):
    #         # 记录 loss 到 tensorboard
    #         self.__loss_placeholder = tf.placeholder(tf.float32, name='loss')
    #         tf.summary.scalar('mean_loss', self.__loss_placeholder)


    def __get_accuracy(self, labels, predict, _size):
        with tf.name_scope('accuracy'):
            labels = tf.argmax(labels, 1)
            predict = tf.argmax(predict, 1)
            correct = tf.equal(labels, predict) # 返回 predict 与 labels 相匹配的结果

            accuracy = tf.divide( tf.reduce_sum( tf.cast(correct, tf.float32) ), _size ) # 计算准确率
            tf.summary.scalar('accuracy', accuracy)
            return accuracy


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

        ret_accuracy_train = self.__get_accuracy(self.__label, self.__output, self.__size)

        # # tensorboard 相关记录
        # self.__summary()

        # 初始化所有变量
        self.init_variables()

        # TensorBoard merge summary
        self.merge_summary()

        # best_val_loss = 999999  # 校验集 loss 最好的情况
        # increase_val_loss_times = 0  # 校验集 loss 连续上升次数
        # mean_loss = 0

        self.echo('\nepoch:')

        for step in range(self.__steps):
            if step % self.SHOW_PROGRESS_FREQUENCY == 0:
                epoch_progress = float(step) % self.__iter_per_epoch / self.__iter_per_epoch * 100.0
                step_progress = float(step) / self.__steps * 100.0
                self.echo('\r step: %d (%d|%.2f%%) / %d|%.2f%% \t\t' % (step, self.__iter_per_epoch, epoch_progress,
                                                                       self.__steps, step_progress), False)

            batch_x_list, batch_y = self.__train_set.next_batch(self.BATCH_SIZE)
            batch_x = batch_x_list[0]

            feed_dict = {self.__image: batch_x, self.__label: batch_y, self.__keep_prob: self.KEEP_PROB}
            _, train_loss = self.sess.run([train_op, self.__loss], feed_dict)

            if step % self.__iter_per_epoch == 0 and step != 0:
                epoch = int(step // self.__iter_per_epoch)

                feed_dict[self.__size] = batch_y.shape[0]
                train_accuracy = self.sess.run(ret_accuracy_train, feed_dict)

                self.echo('\n epoch: %d  train_loss: %.6f  train_accuracy: %.6f \t ' % (epoch, train_loss, train_accuracy))

                # feed_dict[self.__loss_placeholder] = mean_loss / self.__iter_per_epoch
                # mean_loss = 0
                self.add_summary_train(feed_dict, epoch)

                # # 测试 校验集 的 loss
                # mean_val_loss = self.__measure_loss(self.__val_set)
                # batch_val_x, batch_val_y = self.__val_set.next_batch(self.BATCH_SIZE)
                # feed_dict = {self.__image: batch_val_x, self.__mask: batch_val_y, self.__keep_prob: 1.0,
                #              self.__loss_placeholder: mean_val_loss}
                # self.add_summary_val(feed_dict, epoch)
                #
                # if best_val_loss > mean_val_loss:
                #     best_val_loss = mean_val_loss
                #     increase_val_loss_times = 0
                #
                #     self.echo('\n best_val_loss: %.2f \t ' % best_val_loss)
                #     self.save_model_w_b()
                #     # self.save_model()  # 保存模型
                #
                # else:
                #     increase_val_loss_times += 1
                #     if increase_val_loss_times > self.MAX_VAL_LOSS_INCR_TIMES:
                #         break

        self.close_summary()        # 关闭 TensorBoard

        # self.restore_model_w_b()    # 恢复模型
        # self.rebuild_model()        # 重建模型
        # self.get_loss()             # 重新 get loss
        #
        # self.init_variables()       # 重新初始化变量
        #
        # train_loss = self.__measure_loss(self.__train_set)
        # val_loss = self.__measure_loss(self.__val_set)
        # # test_loss = self.__measure_loss(self.__test_set)
        #
        # self.echo('\ntrain mean loss: %.6f' % train_loss)
        # self.echo('validation mean loss: %.6f' % val_loss)
        # # self.echo('test mean loss: %.6f' % test_loss)
        #
        # self.echo('\ndone')
        #
        # batch_x, batch_y = self.__val_set.next_batch(self.BATCH_SIZE)
        # feed_dict = {self.__image: batch_x, self.__keep_prob: 1.0}
        # output_mask = self.sess.run(self.__output_mask, feed_dict)
        #
        # output_mask = np.expand_dims(output_mask, axis=3)
        # for i in range(3):
        #     mask = output_mask[i]
        #     image = batch_x[i]
        #     new_image = np.cast['uint8'](mask * image)
        #
        #     o_image = Image.fromarray(np.cast['uint8'](image))
        #     o_image.show()
        #
        #     o_new_image = Image.fromarray(new_image)
        #     o_new_image.show()

    #
    # def use_model(self, np_image):
    #     if not self.__has_rebuild:
    #         self.restore_model_w_b()    # 恢复模型
    #         self.rebuild_model()        # 重建模型
    #
    #         self.init_variables()       # 初始化所有变量
    #         self.__has_rebuild = True
    #
    #     np_image = np.expand_dims(np_image, axis=0)
    #     feed_dict = {self.__image: np_image, self.__keep_prob: 1.0}
    #     output_mask = self.sess.run(self.__output_mask, feed_dict)
    #
    #     return self.__mask2img(output_mask[0], np_image[0])    # 将 mask 待人 image 并去掉外部的点点


o_deep_id = DeepId()
o_deep_id.run()
