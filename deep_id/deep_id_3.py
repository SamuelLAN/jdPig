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
import load3 as load
import math
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle


'''
    DeepID
    
    原图
    框猪图(带背景)
'''
class DeepId(base.NN):
    MODEL_NAME = 'deep_id'      # 模型的名称

    BATCH_SIZE = 100            # 迭代的 epoch 次数
    EPOCH_TIMES = 100           # 随机梯度下降的 batch 大小

    NUM_CLASSES = 30            # 总共分 NUM_CLASSES 类
    NUM_CHANNEL = 1             # 输入 channel

    IMAGE_SHAPE = [39, 39]      # 输入图片的大小
    IMAGE_PH_SHAPE = [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUM_CHANNEL]    # image 的 placeholder 的 shape

    X_LIST_LEN = 12              # 总共有 X_LIST_LEN 个输入，需要训练 X_LIST_LEN 个 CNN

    BASE_LEARNING_RATE_LIST = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
    DECAY_RATE_LIST = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
    KEEP_PROB_LIST = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

    PARAM_DIR = r'param'            # 动态参数目录地址
    LR_FILE_PATH = r'param/lr.tmp'  # 动态设置学习率的文件地址
    DROPOUT_FILE_PATH = r'param/dropout.tmp'  # 动态设置学习率的文件地址

    FEATURE_DIR = r'feature'
    DEEP_ID_PER_FILE = 10000            # 每个文件存储 DEEP_ID_PER_FILE

    DEEP_ID_LAYER_INDEX = -3            # 倒数第二层为 deep_id 层

    SHOW_PROGRESS_FREQUENCY = 2         # 每 SHOW_PROGRESS_FREQUENCY 个 step show 一次进度 progress

    MAX_VAL_ACCURACY_DECR_TIMES = 30    # 校验集 val_accuracy 连续 100 次没有降低，则 early stop

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
        # deep_id 层
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
        self.__x_list = []
        self.__output_list = []
        self.__loss_list = []
        self.__train_op_list = []
        self.__learning_rate_list = []
        self.__global_step_list = []
        self.__size_list = []
        self.__accuracy_list = []
        self.__keep_prob_list = []

        for i in range(self.X_LIST_LEN):
            # 输入
            self.__x_list.append( tf.placeholder(tf.float32, self.IMAGE_PH_SHAPE, name='X_%d' % i) )

            # 计算 accuracy 时需要用到
            self.__size_list.append( tf.placeholder(tf.float32, name='size_%d' % i) )

            # 记录训练 step
            global_step = self.get_global_step()
            self.__global_step_list.append( global_step )

            # 随训练次数增多而衰减的学习率
            self.__learning_rate_list.append( self.get_learning_rate(
                self.BASE_LEARNING_RATE_LIST[i], global_step, self.__steps, self.DECAY_RATE_LIST[i], staircase=False
            ) )

            # dropout 的 keep_prob
            self.__keep_prob_list.append( tf.placeholder(tf.float32, name='keep_prob_%d' % i) )

        self.__label = tf.placeholder(tf.float32, [None, self.NUM_CLASSES], name='y')

        # self.__learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # self.__learning_rate_value = self.BASE_LEARNING_RATE

        # # 随训练次数增多而衰减的学习率
        # self.__learning_rate = self.get_learning_rate(
        #     self.BASE_LEARNING_RATE, self.global_step, self.__steps, self.DECAY_RATE, staircase=False
        # )

        # # dropout 的 keep_prob
        # self.__keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # self.__keep_prob_value = 0.9

        self.__has_rebuild = False


    ''' 加载数据 '''
    def load(self):
        self.__train_set = load.Data(0.0, 0.64, 'train')
        self.__val_set = load.Data(0.64, 0.8, 'validation')
        self.__test_set = load.Data(0.8, 1.0, 'test')

        self.__train_size = self.__train_set.get_size()
        self.__val_size = self.__val_set.get_size()
        self.__test_size = self.__test_set.get_size()


    ''' 模型 '''
    def model(self):
        for i in range(self.X_LIST_LEN):
            self.__output_list.append( self.deep_model(self.__x_list[i], self.__keep_prob_list[i]) )


    ''' 重建模型 '''
    def rebuild_model(self):
        self.__output_list = []
        self.__loss_list = []
        self.__accuracy_list = []
        self.net = []

        for i in range(self.X_LIST_LEN):
            net = []
            self.__output_list.append( self.deep_model_rebuild(self.__x_list[i], self.WList[i], self.bList[i], net) )
            self.net.append(net)


    ''' 计算 loss '''
    def get_loss(self):
        for i in range(self.X_LIST_LEN):
            with tf.name_scope('loss_%d' % i):
                self.__loss_list.append(tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=self.__output_list[i], labels=self.__label)
                ))


    ''' 获取 train_op list '''
    def __get_train_op_list(self):
        for i in range(self.X_LIST_LEN):
            with tf.name_scope('optimizer_%d' % i):
                optimizer = tf.train.AdamOptimizer(self.__learning_rate_list[i])
                self.__train_op_list.append( optimizer.minimize(self.__loss_list[i],
                                                                global_step=self.__global_step_list[i]) )


    def __get_accuracy(self):
        labels = tf.argmax(self.__label, 1)
        for i in range(self.X_LIST_LEN):
            with tf.name_scope('accuracy_%d' % i):
                predict = tf.argmax(self.__output_list[i], 1)
                correct = tf.equal(labels, predict)     # 返回 predict 与 labels 相匹配的结果

                self.__accuracy_list.append(            # 计算准确率
                    tf.divide( tf.reduce_sum( tf.cast(correct, tf.float32) ), self.__size_list[i], name='accuracy' )
                )


    def __measure(self, data_set, net_index, max_times=None):
        times = int(math.ceil(float(data_set.get_size()) / self.BATCH_SIZE))
        if max_times:
            times = min(max_times, times)

        mean_accuracy = 0
        mean_loss = 0
        for i in range(times):
            batch_x_list, batch_y = data_set.next_batch(self.BATCH_SIZE)
            feed_dict = {self.__x_list[net_index]: batch_x_list[net_index], self.__label: batch_y,
                         self.__size_list[net_index]: batch_y.shape[0], self.__keep_prob_list[net_index]: 1.0}
            loss, accuracy = self.sess.run([self.__loss_list[net_index], self.__accuracy_list[net_index]], feed_dict)
            mean_accuracy += accuracy
            mean_loss += loss

            progress = float(i + 1) / times * 100
            self.echo('\r >> measuring net_%d progress: %.2f%% | %d \t' % (net_index, progress, times), False)

        return mean_accuracy / times, mean_loss / times


    def __summary(self):
        self.__mean_accuracy_list = []
        self.__mean_loss_list = []

        for i in range(self.X_LIST_LEN):
            with tf.name_scope('summary_%d' % i):
                mean_accuracy = tf.placeholder(tf.float32, name='mean_accuracy')
                mean_loss = tf.placeholder(tf.float32, name='mean_loss')

                tf.summary.scalar('learning_rate', self.__learning_rate_list[i])
                tf.summary.scalar('keep_prob', self.__keep_prob_list[i])
                tf.summary.scalar('mean_accuracy', mean_accuracy)
                tf.summary.scalar('mean_loss', mean_loss)

                self.__mean_accuracy_list.append(mean_accuracy)
                self.__mean_loss_list.append(mean_loss)


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

        self.__get_accuracy()

        # 生成训练的 op
        self.__get_train_op_list()

        # tensorboard 相关记录
        self.__summary()

        # self.__build_param_dir()

        # 初始化所有变量
        self.init_variables()

        # TensorBoard merge summary
        self.merge_summary()

        self.echo('\nStart training ... \nepoch:')

        mean_train_loss_list = [0 for i in range(self.X_LIST_LEN)]
        mean_train_accuracy_list = [0 for i in range(self.X_LIST_LEN)]
        mean_val_loss_list = [0 for i in range(self.X_LIST_LEN)]
        mean_val_accuracy_list = [0 for i in range(self.X_LIST_LEN)]

        best_val_accuracy_list = [0 for i in range(self.X_LIST_LEN)]
        decr_val_accu_times_list = [0 for i in range(self.X_LIST_LEN)]
        stop_training_list = [False for i in range(self.X_LIST_LEN)]

        for step in range(self.__steps):
            if step % self.SHOW_PROGRESS_FREQUENCY == 0:
                epoch_progress = float(step) % self.__iter_per_epoch / self.__iter_per_epoch * 100.0
                step_progress = float(step) / self.__steps * 100.0
                self.echo('\r step: %d (%d|%.2f%%) / %d|%.2f%% \t\t' % (step, self.__iter_per_epoch, epoch_progress,
                                                                       self.__steps, step_progress), False)

            batch_x_list, batch_y = self.__train_set.next_batch(self.BATCH_SIZE)

            # 遍历每个网络；总共有 X_LIST_LEN 个网络需要训练
            for i in range(self.X_LIST_LEN):
                if stop_training_list[i]:
                    continue

                batch_x = batch_x_list[i]

                feed_dict = {self.__x_list[i]: batch_x, self.__label: batch_y,
                             self.__size_list[i]: batch_y.shape[0], self.__keep_prob_list[i]: self.KEEP_PROB_LIST[i]}
                _, train_loss, train_accuracy = self.sess.run([self.__train_op_list[i], self.__loss_list[i],
                                                               self.__accuracy_list[i]], feed_dict)

                mean_train_loss_list[i] += train_loss           # 计算 mean_train_loss
                mean_train_accuracy_list[i] += train_accuracy   # 计算 mean_train_accuracy

            if step % self.__iter_per_epoch == 0 and step != 0:
                # self.__update_param()

                epoch = int(step // self.__iter_per_epoch)
                self.echo('\n epoch: %d ' % epoch)

                summary_feed_dict = {self.__label: batch_y}

                for i in range(self.X_LIST_LEN):
                    mean_train_loss_list[i] /= self.__iter_per_epoch
                    mean_train_accuracy_list[i] /= self.__iter_per_epoch

                    summary_feed_dict[self.__mean_loss_list[i]] = mean_train_loss_list[i]
                    summary_feed_dict[self.__mean_accuracy_list[i]] = mean_train_accuracy_list[i]
                    summary_feed_dict[self.__x_list[i]] = batch_x
                    summary_feed_dict[self.__size_list[i]] = batch_y.shape[0]
                    summary_feed_dict[self.__keep_prob_list[i]] = self.KEEP_PROB_LIST[i]

                self.add_summary_train(summary_feed_dict, epoch)

                for i in range(self.X_LIST_LEN):
                    if not stop_training_list[i]:
                        mean_val_accuracy, mean_val_loss = self.__measure(self.__val_set, i, 5)
                        mean_val_loss_list[i] = mean_val_loss
                        mean_val_accuracy_list[i] = mean_val_accuracy

                    summary_feed_dict[self.__mean_loss_list[i]] = mean_val_loss_list[i]
                    summary_feed_dict[self.__mean_accuracy_list[i]] = mean_val_accuracy_list[i]

                self.add_summary_val(summary_feed_dict, epoch)

                for i in range(self.X_LIST_LEN):
                    echo_str = '\n\t net: %d  train_loss: %.6f  train_accuracy: %.6f ' \
                               ' val_loss: %.6f  val_accuracy: %.6f  has_stop: %s ' % (i,
                                mean_train_loss_list[i], mean_train_accuracy_list[i], mean_val_loss_list[i],
                                mean_val_accuracy_list[i], stop_training_list[i])

                    if best_val_accuracy_list[i] < mean_val_accuracy_list[i]:
                        best_val_accuracy_list[i] = mean_val_accuracy_list[i]
                        decr_val_accu_times_list[i] = 0

                        self.echo('%s  best \t ' % echo_str, False)
                        self.save_model_i(i)

                    else:
                        decr_val_accu_times_list[i] += 1
                        self.echo('%s  decr_times: %d ' % (echo_str, decr_val_accu_times_list[i]), False)

                        if decr_val_accu_times_list[i] > self.MAX_VAL_ACCURACY_DECR_TIMES:
                            stop_training_list[i] = True
                self.echo('')

                for i in range(self.X_LIST_LEN):
                    if stop_training_list[i]:
                        continue

                    mean_train_loss_list[i] = 0
                    mean_train_accuracy_list[i] = 0

                stop_all = True
                for is_stop in stop_training_list:
                    if not is_stop:
                        stop_all = False
                        break

                if stop_all:
                    break

        self.echo('\nFinish training ')

        self.close_summary()  # 关闭 TensorBoard

        self.WList = [None for i in range(self.X_LIST_LEN)]
        self.bList = [None for i in range(self.X_LIST_LEN)]

        for i in range(self.X_LIST_LEN):    # 恢复模型
            self.restore_model_i(i)

        self.rebuild_model()    # 重建模型
        self.get_loss()         # 重新 get loss
        self.__get_accuracy()   # 重新计算 accuracy

        self.init_variables()   # 重新初始化变量

        for i in range(self.X_LIST_LEN):
            self.echo('\n**********************************\nCalculating net_%d accuracy ... ' % i)

            mean_train_accuracy, mean_train_loss = self.__measure(self.__train_set, i)
            mean_val_accuracy, mean_val_loss = self.__measure(self.__val_set, i)
            mean_test_accuracy, mean_test_loss = self.__measure(self.__test_set, i)

            self.echo('train_accuracy: %.6f  train_loss: %.6f ' % (mean_train_accuracy, mean_train_loss))
            self.echo('val_accuracy: %.6f  val_loss: %.6f ' % (mean_val_accuracy, mean_val_loss))
            self.echo('test_accuracy: %.6f  test_loss: %.6f ' % (mean_test_accuracy, mean_test_loss))

        self.__train_set.stop()  # 关闭获取数据线程
        self.__val_set.stop()  # 关闭获取数据线程
        self.__test_set.stop()  # 关闭获取数据线程

        self.echo('\ndone')


    '''
     np_patch 需为灰度图片, shape : (patch_num, h, w, 1)
    '''
    def generate_deep_id(self, np_patch_list):
        if not self.__has_rebuild:
            self.WList = [None for i in range(self.X_LIST_LEN)]
            self.bList = [None for i in range(self.X_LIST_LEN)]

            for i in range(self.X_LIST_LEN):    # 恢复模型
                self.restore_model_i(i)

            self.rebuild_model()    # 重建模型

            self.init_variables()   # 重新初始化变量
            self.__has_rebuild = True

        deep_id_list = []
        for i in range(self.X_LIST_LEN):
            np_patch = np.expand_dims(np_patch_list[i], axis=0)
            deep_id_layer = self.net[i][self.DEEP_ID_LAYER_INDEX]

            feed_dict = {self.__x_list[i]: np_patch, self.__keep_prob_list[i]: 1.0}
            deep_id = self.sess.run(deep_id_layer, feed_dict)

            deep_id_list.append( deep_id.reshape([-1,]) )

        return np.hstack(deep_id_list)


    def __save_deep_id(self, data_set, name):
        if not os.path.isdir(self.FEATURE_DIR):
            os.mkdir(self.FEATURE_DIR)

        data = []
        file_no = 0

        self.echo('\nSaving %s deep_id ... ' % name)

        times = int(math.ceil(float(data_set.get_size()) / self.BATCH_SIZE))
        for i in range(times):
            progress = float(i + 1) / times * 100
            self.echo('\r >> saving progress: %.2f \t ' % progress, False)

            batch_x_list, batch_y = data_set.next_batch(self.BATCH_SIZE)
            batch_x_list = batch_x_list.transpose([1, 0, 2, 3, 4])

            for j, x_list in enumerate(batch_x_list):
                deep_id = self.generate_deep_id(x_list)
                data.append([deep_id, batch_y[j]])

                if len(data) >= self.DEEP_ID_PER_FILE:
                    with open( os.path.join(self.FEATURE_DIR, '%s_%d.pkl' % (name, file_no)), 'wb' ) as f:
                        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

                    data = []
                    file_no += 1

        if len(data):
            with open( os.path.join(self.FEATURE_DIR, '%s_%d.pkl' % (name, file_no)), 'wb' ) as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        self.echo('Finish saving %s deep_id ' % name)


    def save_deep_id(self):
        self.__save_deep_id(self.__test_set, 'train')
        self.__save_deep_id(self.__val_set, 'val')
        self.__save_deep_id(self.__test_set, 'test')


o_deep_id = DeepId()
# o_deep_id.run()
o_deep_id.save_deep_id()
