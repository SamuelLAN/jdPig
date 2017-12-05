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
import math
import numpy as np
import tensorflow as tf


'''
    DeepID
    
    原图
    框猪图(带背景)
'''
class DeepId(base.NN):
    MODEL_NAME = 'deep_id'      # 模型的名称

    BATCH_SIZE = 100             # 迭代的 epoch 次数
    EPOCH_TIMES = 500           # 随机梯度下降的 batch 大小

    NUM_CLASSES = 30            # 总共分 NUM_CLASSES 类
    NUM_CHANNEL = 3             # 输入 channel

    IMAGE_SHAPE = [39, 39]      # 输入图片的大小
    IMAGE_PH_SHAPE = [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUM_CHANNEL]    # image 的 placeholder 的 shape

    X_LIST_LEN = 6              # 总共有 X_LIST_LEN 个输入，需要训练 X_LIST_LEN 个 CNN

    BASE_LEARNING_RATE = 0.03    # 初始 学习率
    DECAY_RATE = 0.1            # 学习率 的 下降速率

    KEEP_PROB = 0.75             # dropout 的 keep_prob

    PARAM_DIR = r'param'            # 动态参数目录地址
    LR_FILE_PATH = r'param/lr.tmp'  # 动态设置学习率的文件地址
    DROPOUT_FILE_PATH = r'param/dropout.tmp'  # 动态设置学习率的文件地址

    DEEP_ID_LAYER_INDEX = -2    # 倒数第二层为 deep_id 层

    SHOW_PROGRESS_FREQUENCY = 2         # 每 SHOW_PROGRESS_FREQUENCY 个 step show 一次进度 progress

    MAX_VAL_ACCURACY_DECR_TIMES = 20    # 校验集 val_accuracy 连续 100 次没有降低，则 early stop

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
            'type': 'conv',
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
        {
            'name': 'dropout',
            'type': 'dropout',
        },
        {   # softmax 层
            'name': 'softmax',
            'type': 'fc',
            'shape': [320, NUM_CLASSES],
            'activate': False,
        }
    ]


    ''' 自定义 初始化变量 过程 '''
    def init(self):
        # 加载数据
        self.load()

        # 常量
        self.__iter_per_epoch = int(self.__train_size // self.BATCH_SIZE)
        self.__steps = self.EPOCH_TIMES * self.__iter_per_epoch

        # 初始化变量
        # self.__x_list = []
        # self.__output_list = []
        # self.__loss_list = []
        # self.__train_op_list = []
        # self.__learning_rate_list = []
        # self.__global_step_list = []
        # self.__size_list = []
        # self.__accuracy_list = []

        self.__X = tf.placeholder(tf.float32, self.IMAGE_PH_SHAPE, name='X')

        self.__size = tf.placeholder(tf.float32, name='size')

        # for i in range(self.X_LIST_LEN):
        #     # 输入
        #     self.__x_list.append( tf.placeholder(tf.float32, self.IMAGE_PH_SHAPE, name='X_%d' % i) )
        #
        #     self.__size_list.append( tf.placeholder(tf.float32, name='size_%d' % i) )
        #
        #     # 记录训练 step
        #     global_step = self.get_global_step()
        #     self.__global_step_list.append( global_step )
        #
        #     # 随训练次数增多而衰减的学习率
        #     self.__learning_rate_list.append( self.get_learning_rate(
        #         self.BASE_LEARNING_RATE, global_step, self.__steps, self.DECAY_RATE, staircase=False
        #     ) )
        
        # self.__learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # self.__learning_rate_value = self.BASE_LEARNING_RATE

        # 随训练次数增多而衰减的学习率
        self.__learning_rate = self.get_learning_rate(
            self.BASE_LEARNING_RATE, self.global_step, self.__steps, self.DECAY_RATE, staircase=False
        )

        self.__label = tf.placeholder(tf.float32, [None, self.NUM_CLASSES], name='y')
        # dropout 的 keep_prob
        self.__keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.__keep_prob_value = 0.9

        self.__has_rebuild = False


    ''' 加载数据 '''
    def load(self):
        sort_list = load.Data.get_sort_list()
        self.__train_set = load.Data(0.0, 0.64, 'train', sort_list)
        self.__val_set = load.Data(0.64, 0.8, 'validation', sort_list)
        self.__test_set = load.Data(0.8, 1.0, 'test', sort_list)

        self.__train_size = self.__train_set.get_size()
        self.__val_size = self.__val_set.get_size()
        self.__test_size = self.__test_set.get_size()


    ''' 模型 '''
    def model(self):
        self.__output = self.deep_model(self.__X, self.__keep_prob)
        # for i in range(self.X_LIST_LEN):
        #     self.__output_list.append( self.deep_model(self.__x_list[i], self.__keep_prob) )


    ''' 重建模型 '''
    def rebuild_model(self):
        self.__output = self.deep_model_rebuild(self.__X, self.WList[0], self.bList[0])
        # self.__output_list = []
        # self.__loss_list = []
        #
        # for i in range(self.X_LIST_LEN):
        #     self.__output_list.append( self.deep_model_rebuild(self.__x_list[i]) )


    ''' 计算 loss '''
    def get_loss(self):
        with tf.name_scope('loss'):
            self.__loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=self.__output, labels=self.__label),
                    name='entropy'
                )

            # for i in range(self.X_LIST_LEN):
            #     loss = tf.reduce_mean(
            #         tf.nn.softmax_cross_entropy_with_logits(logits=self.__output_list[i], labels=self.__label),
            #         name='entropy'
            #     )
            #     self.__loss_list.append(loss)
            #     tf.summary.scalar('loss_%d' % i, loss)  # 记录 loss 到 tensorboard


    ''' 获取 train_op '''
    def get_train_op(self, loss, learning_rate, global_step):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            return optimizer.minimize(loss, global_step=global_step)


    # ''' 获取 train_op list '''
    # def __get_train_op_list(self):
    #     with tf.name_scope('optimizer'):
    #         for i in range(self.X_LIST_LEN):
    #             optimizer = tf.train.AdamOptimizer(self.__learning_rate_list[i])
    #             self.__train_op_list.append( optimizer.minimize(self.__loss_list[i],
    #                                                             global_step=self.__global_step_list[i]) )


    def __get_accuracy(self, labels, predict, _size):
        with tf.name_scope('accuracy'):
            labels = tf.argmax(labels, 1)
            predict = tf.argmax(predict, 1)
            correct = tf.equal(labels, predict) # 返回 predict 与 labels 相匹配的结果

            self.__accuracy = tf.divide( tf.reduce_sum( tf.cast(correct, tf.float32) ), _size ) # 计算准确率


    def __measure(self, data_set):
        times = int(math.ceil(float(data_set.get_size()) / self.BATCH_SIZE))

        mean_accuracy = 0
        mean_loss = 0
        for i in range(times):
            batch_x, batch_y = data_set.next_batch(self.BATCH_SIZE)
            feed_dict = {self.__X: batch_x, self.__label: batch_y,
                         self.__size: batch_y.shape[0], self.__keep_prob: 1.0}
            loss, accuracy = self.sess.run([self.__loss, self.__accuracy], feed_dict)
            mean_accuracy += accuracy
            mean_loss += loss

            progress = float(i + 1) / times * 100
            self.echo('\r >> measuring progress: %.2f%% | %d \t' % (progress, times), False)

        return mean_accuracy / times, mean_loss / times


    def __summary(self):
        with tf.name_scope('summary'):
            self.__mean_accuracy = tf.placeholder(tf.float32, name='mean_accuracy')
            self.__mean_loss = tf.placeholder(tf.float32, name='mean_loss')

        tf.summary.scalar('loss', self.__loss)
        tf.summary.scalar('accuracy', self.__accuracy)
        tf.summary.scalar('learning_rate', self.__learning_rate)
        tf.summary.scalar('mean_accuracy', self.__mean_accuracy)
        tf.summary.scalar('mean_loss', self.__mean_loss)


    # ''' 计算准确率 '''
    # def __get_accuracy(self):
    #     with tf.name_scope('accuracy'):
    #         labels = tf.argmax(self.__label, 1)
    #         for i in range(self.X_LIST_LEN):
    #             predict = tf.argmax(self.__output_list[i], 1)
    #             correct = tf.equal(labels, predict)
    #
    #             accuracy = tf.divide( tf.reduce_mean( tf.cast(correct, tf.float32) ), self.__size_list[i] )
    #             self.__accuracy_list.append(accuracy)
    #
    #             # 将 准确率 记录到 TensorBoard
    #             tf.summary.scalar('accuracy', accuracy)


    # ''' 将图片输出到 tensorboard '''
    # def __summary(self):
    #     with tf.name_scope('summary'):
    #         # 记录 loss 到 tensorboard
    #         self.__loss_placeholder = tf.placeholder(tf.float32, name='loss')
    #         tf.summary.scalar('mean_loss', self.__loss_placeholder)


    def __build_param_dir(self):
        if not os.path.isdir(self.PARAM_DIR):
            os.mkdir(self.PARAM_DIR)

        with open(self.LR_FILE_PATH, 'wb') as f:
            f.write('%.6f' % self.BASE_LEARNING_RATE)

        with open(self.DROPOUT_FILE_PATH, 'wb') as f:
            f.write('%.6f' % self.__keep_prob_value)


    def __update_param(self):
        with open(self.LR_FILE_PATH, 'rb') as f:
            self.__learning_rate_value = float(f.read())

        with open(self.DROPOUT_FILE_PATH, 'rb') as f:
            self.__keep_prob_value = float(f.read())


    ''' 主函数 '''
    def run(self):
        # 生成模型
        self.model()

        # 计算 loss
        self.get_loss()

        # 正则化
        # self.__loss = self.regularize_trainable(self.__loss, self.REGULAR_BETA)

        self.__get_accuracy(self.__label, self.__output, self.__size)

        # 生成训练的 op
        train_op = self.get_train_op(self.__loss, self.__learning_rate, self.global_step)
        # self.__get_train_op_list()

        # tensorboard 相关记录
        self.__summary()

        # self.__build_param_dir()

        # 初始化所有变量
        self.init_variables()

        # TensorBoard merge summary
        self.merge_summary()

        best_val_accuracy = 0
        decrease_val_accuracy_times = 0
        mean_train_accuracy = 0
        mean_train_loss = 0

        self.echo('\nStart training ... \nepoch:')

        for step in range(self.__steps):
            if step % self.SHOW_PROGRESS_FREQUENCY == 0:
                epoch_progress = float(step) % self.__iter_per_epoch / self.__iter_per_epoch * 100.0
                step_progress = float(step) / self.__steps * 100.0
                self.echo('\r step: %d (%d|%.2f%%) / %d|%.2f%% \t\t' % (step, self.__iter_per_epoch, epoch_progress,
                                                                       self.__steps, step_progress), False)

            batch_x, batch_y = self.__train_set.next_batch(self.BATCH_SIZE)

            # keep_prob = self.KEEP_PROB if step / self.__iter_per_epoch > 15 else 1.0

            feed_dict = {self.__X: batch_x, self.__label: batch_y, self.__size: batch_y.shape[0],
                         self.__keep_prob: self.KEEP_PROB,
                         # self.__learning_rate: self.__learning_rate_value, self.__keep_prob: self.__keep_prob_value,
                         }
            _, train_loss, train_accuracy = self.sess.run([train_op, self.__loss, self.__accuracy], feed_dict)
            
            mean_train_accuracy += train_accuracy
            mean_train_loss += train_loss

            if step % self.__iter_per_epoch == 0 and step != 0:
                # self.__update_param()

                epoch = int(step // self.__iter_per_epoch)
                mean_train_accuracy /= self.__iter_per_epoch
                mean_train_loss /= self.__iter_per_epoch

                feed_dict[self.__mean_accuracy] = mean_train_accuracy
                feed_dict[self.__mean_loss] = mean_train_loss
                self.add_summary_train(feed_dict, epoch)
                
                mean_val_accuracy, mean_val_loss = self.__measure(self.__val_set)

                batch_val_x, batch_val_y = self.__val_set.next_batch(self.BATCH_SIZE)
                feed_dict = {self.__X: batch_val_x, self.__label: batch_val_y,
                             self.__size: batch_val_y.shape[0], self.__keep_prob: 1.0,
                             self.__mean_accuracy: mean_val_accuracy, self.__mean_loss: mean_val_loss,
                             self.__learning_rate: self.__learning_rate_value}
                self.add_summary_val(feed_dict, epoch)

                self.echo('\n epoch: %d  mean_train_loss: %.6f  mean_train_accuracy: %.6f  mean_val_loss: %.6f  mean_val_accuracy: %.6f \t ' %
                          (epoch, mean_train_loss, mean_train_accuracy, mean_val_loss, mean_val_accuracy))

                mean_train_accuracy = 0
                mean_train_loss = 0

                if best_val_accuracy < mean_val_accuracy:
                    best_val_accuracy = mean_val_accuracy
                    decrease_val_accuracy_times = 0
                    
                    self.echo(' \t best_val_accuracy: %.6f \t ' % best_val_accuracy)

                    if best_val_accuracy > 0.8:
                        self.save_model_w_b()
                    
                else:
                    decrease_val_accuracy_times += 1
                    if decrease_val_accuracy_times > self.MAX_VAL_ACCURACY_DECR_TIMES:
                        break
                
                # batch_x_list, batch_y = self.__train_set.next_batch(self.BATCH_SIZE)
            #
            # for i in range(self.X_LIST_LEN):
            #     batch_x = batch_x_list[i]
            #     train_op = self.__train_op_list[i]
            #     loss = self.__loss_list[i]
            #
            #     feed_dict = {self.__x_list[i]: batch_x, self.__label: batch_y, self.__keep_prob: self.KEEP_PROB}
            #     _, train_loss = self.sess.run([train_op, loss], feed_dict)
            #
            #     if step % self.__iter_per_epoch == 0 and step != 0:
            #         accuracy = self.__accuracy_list[i]
            #         _size = self.__size_list[i]
            #
            #         epoch = int(step // self.__iter_per_epoch)
            #
            #         feed_dict[_size] = batch_y.shape[0]
            #         train_accuracy = self.sess.run(accuracy, feed_dict)
            #
            #         self.echo('\n epoch: %d  net: %d  loss: %.6f  accuracy: %.6f \t ' % (epoch, i, train_loss, train_accuracy))
            #         # self.add_summary_train(feed_dict, epoch)

        self.echo('\nFinish training ')

        self.close_summary()  # 关闭 TensorBoard

        self.restore_model_w_b()  # 恢复模型
        self.rebuild_model()  # 重建模型
        self.get_loss()  # 重新 get loss
        self.__get_accuracy(self.__label, self.__output, self.__size)


        self.init_variables()  # 重新初始化变量

        mean_train_accuracy, mean_train_loss = self.__measure(self.__train_set)
        mean_val_accuracy, mean_val_loss = self.__measure(self.__val_set)
        mean_test_accuracy, mean_test_loss = self.__measure(self.__test_set)

        self.echo('\nmean_train_accuracy: %.6f  mean_train_loss: %.6f ' % (mean_train_accuracy, mean_train_loss))
        self.echo('mean_val_accuracy: %.6f  mean_val_loss: %.6f ' % (mean_val_accuracy, mean_val_loss))
        self.echo('mean_test_accuracy: %.6f  mean_test_loss: %.6f ' % (mean_test_accuracy, mean_test_loss))

        self.__train_set.stop()  # 关闭获取数据线程
        self.__val_set.stop()  # 关闭获取数据线程
        self.__test_set.stop()  # 关闭获取数据线程

        self.echo('\ndone')


o_deep_id = DeepId()
o_deep_id.run()
