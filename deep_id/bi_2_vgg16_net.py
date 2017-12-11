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
import bi_2_base as base
import bi_load as load
import vgg
import csv
import math
# import Queue
import numpy as np
# from PIL import Image
import tensorflow as tf

''' 全卷积神经网络 '''


class VGG16(base.NN):
    MODEL_NAME = 'vgg_16_30'  # 模型的名称

    BATCH_SIZE = 16 # 迭代的 epoch 次数
    EPOCH_TIMES = 10  # 随机梯度下降的 batch 大小

    NUM_CHANNEL = 3  # 输入图片为 3 通道，彩色
    NUM_CLASSES = 2  # 输出的类别

    NUM_PIG = 30

    IMAGE_SHAPE = [224, 224]
    IMAGE_PIXELS = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
    IMAGE_PH_SHAPE = [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUM_CHANNEL]  # image 的 placeholder 的 shape

    BASE_LEARNING_RATE = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                          0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.00001, 0.0001, 0.0001, 0.0001, 0.0001,
                          0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]

    DECAY_RATE = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
                  0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
                  0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]

    REGULAR_BETA = [0.1, 0.1, 0.85, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                    0.8, 0.5, 0.5, 0.5, 0.5,  0.5] + [0.1] * 14  # 正则化的 beta 参数
    KEEP_PROB = 0.5  # dropout 的 keep_prob

    CORRECT_WEIGHT = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                      0.9, 0.9, 0.9, 0.9, 0.9,  0.9, 0.9, 0.9, 0.9, 0.9,
                      0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    INCORRECT_WEIGHT = [1.3, 1.3, 1.1, 1.3, 1.3, 1.3, 1.3, 1.2, 1.3, 1.3,
                        1.1, 1.3, 1.2, 1.3, 1.3,  1.3, 1.3, 1.2, 1.3, 1.3,
                        1.3, 1.3, 1.2, 1.3, 1.3, 1.3, 1.3, 1.2, 1.3, 1.3]

    EPLISION = 0.00001

    SHOW_PROGRESS_FREQUENCY = 2  # 每 SHOW_PROGRESS_FREQUENCY 个 step show 一次进度 progress

    # MAX_VAL_LOSS_INCR_TIMES = 20  # 校验集 val_loss 连续 100 次没有降低，则 early stop

    TENSORBOARD_SHOW_IMAGE = False  # 默认不将 image 显示到 TensorBoard，以免影响性能

    VGG_MODEL = vgg.VGG.load()  # 加载 VGG 模型

    MAX_VAL_ACCURACY_DECR_TIMES = 4  # 校验集 val_accuracy 连续 100 次没有降低，则 early stop

    RESULT_DIR = r'result'
    RESULT_FILE_PATH = r'result/test_B.csv'

    ''' 模型的配置；采用了 VGG16 模型的 FCN '''
    MODEL = [
        {
            'name': 'conv1_1',
            'type': 'conv',
            'W': VGG_MODEL['conv1_1'][0],
            'b': VGG_MODEL['conv1_1'][1],
            'bn': True,
            'trainable': False,
        },
        {
            'name': 'conv1_2',
            'type': 'conv',
            'W': VGG_MODEL['conv1_2'][0],
            'b': VGG_MODEL['conv1_2'][1],
            'bn': True,
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
            'bn': True,
            'trainable': False,
        },
        {
            'name': 'conv2_2',
            'type': 'conv',
            'W': VGG_MODEL['conv2_2'][0],
            'b': VGG_MODEL['conv2_2'][1],
            'bn': True,
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
            'bn': True,
            'trainable': False,
        },
        {
            'name': 'conv3_2',
            'type': 'conv',
            'W': VGG_MODEL['conv3_2'][0],
            'b': VGG_MODEL['conv3_2'][1],
            'bn': True,
            'trainable': False,
        },
        {
            'name': 'conv3_3',
            'type': 'conv',
            'W': VGG_MODEL['conv3_3'][0],
            'b': VGG_MODEL['conv3_3'][1],
            'bn': True,
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
            'bn': True,
            'trainable': False,
        },
        {
            'name': 'conv4_2',
            'type': 'conv',
            'W': VGG_MODEL['conv4_2'][0],
            'b': VGG_MODEL['conv4_2'][1],
            'bn': True,
            'trainable': False,
        },
        {
            'name': 'conv4_3',
            'type': 'conv',
            'W': VGG_MODEL['conv4_3'][0],
            'b': VGG_MODEL['conv4_3'][1],
            'bn': True,
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
            'bn': True,
            'trainable': False,
        },
        {
            'name': 'conv5_2',
            'type': 'conv',
            'W': VGG_MODEL['conv5_2'][0],
            'b': VGG_MODEL['conv5_2'][1],
            'bn': True,
            'trainable': False,
        },
        {
            'name': 'conv5_3',
            'type': 'conv',
            'W': VGG_MODEL['conv5_3'][0],
            'b': VGG_MODEL['conv5_3'][1],
            'bn': True,
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
            'shape': VGG_MODEL['fc6'][0].shape,
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
            'shape': VGG_MODEL['fc7'][0].shape,
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
            'shape': VGG_MODEL['fc8'][0].shape,
            'trainable': True,
        },
        {
            'name': 'dropout_8',
            'type': 'dropout',
        },
        {
            'name': 'softmax',
            'type': 'fc',
            'shape': [1000, NUM_CLASSES],
            'activate': False,
        },
    ]


    ''' 自定义 初始化变量 过程 '''
    def init(self):
        # 输入 与 label
        self.__image = tf.placeholder(tf.float32, self.IMAGE_PH_SHAPE, name='X')
        self.__label = tf.placeholder(tf.float32, [None, self.NUM_CLASSES], name='y')
        self.__size = tf.placeholder(tf.float32, name='size')

        # dropout 的 keep_prob
        self.__keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.__train_set_list = [None for i in range(self.NUM_PIG)]
        self.__val_set_list = [None for i in range(self.NUM_PIG)]

        self.__train_size_list = [0 for i in range(self.NUM_PIG)]
        self.__val_size_list = [0 for i in range(self.NUM_PIG)]


    def reinit(self, pig_id):
        self.net = []  # 存放每层网络的 feature map
        self.WList = []  # 存放权重矩阵的 list
        self.bList = []  # 存放偏置量的 list

        # 加载数据
        self.load_i(pig_id)

        # 常量
        self.__iter_per_epoch = int(self.__train_size_list[pig_id] // self.BATCH_SIZE)
        self.__steps = self.EPOCH_TIMES * self.__iter_per_epoch

        self.global_step = self.get_global_step()

        self.__learning_rate = self.get_learning_rate(
            self.BASE_LEARNING_RATE[pig_id], self.global_step, self.__steps, self.DECAY_RATE[pig_id], staircase=False
        )

        self.__has_rebuild = False

        self.sess = tf.Session()


    ''' 加载数据 '''
    def load_i(self, _id):
        self.__train_set_list[_id] = load.Data(_id, 0.0, 0.8, 'train')
        self.__val_set_list[_id] = load.Data(_id, 0.8, 1.0, 'validation')

        self.__train_size_list[_id] = self.__train_set_list[_id].get_size()
        self.__val_size_list[_id] = self.__val_set_list[_id].get_size()


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
                tf.nn.softmax_cross_entropy_with_logits(logits=self.__output, labels=self.__label)
            )


    ''' 获取 train_op '''
    def get_train_op(self, loss, learning_rate, global_step):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            optimizer_op = optimizer.minimize(loss, global_step=global_step)

            batch_norm_updates = tf.get_collection(self.UPDATE_OPS_COLLECTION)
            batch_norm_updates_op = tf.group(*batch_norm_updates)
            return tf.group(optimizer_op, batch_norm_updates_op)


    ''' 将图片输出到 tensorboard '''
    def __summary(self):
        with tf.name_scope('summary'):
            self.__mean_accuracy = tf.placeholder(tf.float32, name='mean_accuracy')
            self.__mean_loss = tf.placeholder(tf.float32, name='mean_loss')
            self.__mean_log_loss = tf.placeholder(tf.float32, name='mean_log_loss')
            # self.__mean_ch_log_loss = tf.placeholder(tf.float32, name='mean_ch_log_loss')

            tf.summary.scalar('learning_rate', self.__learning_rate)
            # tf.summary.scalar('keep_prob', self.__keep_prob)
            tf.summary.scalar('mean_accuracy', self.__mean_accuracy)
            tf.summary.scalar('mean_loss', self.__mean_loss)
            tf.summary.scalar('mean_log_loss', self.__mean_log_loss)
            # tf.summary.scalar('mean_ch_log_loss', self.__mean_ch_log_loss)


    def __get_accuracy(self):
        with tf.name_scope('accuracy'):
            labels = tf.argmax(self.__label, 1)
            predict = tf.argmax(self.__output, 1)
            correct = tf.equal(labels, predict) # 返回 predict 与 labels 相匹配的结果

            self.__accuracy = tf.divide( tf.reduce_sum( tf.cast(correct, tf.float32) ), self.__size ) # 计算准确率


    def __get_log_loss(self, pig_id):
        with tf.name_scope('log_loss'):
            labels = self.__label
            predict = tf.one_hot( tf.argmax(self.__output, 1), depth=self.NUM_CLASSES )

            correct = tf.cast( tf.equal(labels, predict), tf.float32 )
            incorrect = tf.cast( tf.not_equal(labels, predict), tf.float32 )

            # w = correct * 1.5 + incorrect * 0.8
            w = correct * self.CORRECT_WEIGHT[pig_id] + incorrect * self.INCORRECT_WEIGHT[pig_id]
            output = w * self.__output

            exp_x = tf.exp(self.__output)
            self.__prob = exp_x / tf.reduce_sum(exp_x, axis=0)
            # p = tf.maximum( tf.minimum( prob, 1 - 1e-15 ), 1e-15 )
            self.__log_loss = - tf.divide( tf.reduce_sum( tf.multiply(self.__label, tf.log(self.__prob)) ), self.__size )

            exp_x = tf.exp(output)
            p = exp_x / tf.reduce_sum(exp_x, axis=0)
            self.__ch_log_loss = - tf.divide(tf.reduce_sum(tf.multiply(self.__label, tf.log(p))), self.__size)


    def __measure(self, data_set, max_times=None):
        times = int(math.ceil(float(data_set.get_size()) / self.BATCH_SIZE))
        if max_times:
            times = min(max_times, times)

        mean_accuracy = 0.0
        mean_loss = 0.0
        mean_log_loss = 0.0
        # mean_ch_log_loss = 0.0
        for i in range(times):
            batch_x, batch_y = data_set.next_batch(self.BATCH_SIZE)

            batch_x = (batch_x - self.mean_x) / (self.std_x + self.EPLISION)
            feed_dict = {self.__image: batch_x, self.__label: batch_y,
                         self.__size: batch_y.shape[0], self.__keep_prob: 1.0}

            # loss, log_loss, ch_log_loss, accuracy = self.sess.run([self.__loss, self.__log_loss, self.__ch_log_loss, self.__accuracy], feed_dict)
            loss, log_loss, accuracy = self.sess.run([self.__loss, self.__log_loss, self.__accuracy], feed_dict)
            mean_loss += loss
            mean_log_loss += log_loss
            # mean_ch_log_loss += ch_log_loss
            mean_accuracy += accuracy

            del batch_x
            del batch_y

            progress = float(i + 1) / times * 100
            self.echo('\r >> measuring progress: %.2f%% | %d \t' % (progress, times), False)

        return mean_accuracy / times, mean_loss / times, mean_log_loss / times


    def __measure_prob(self, data_set):
        batch_size = 100
        times = int(math.ceil(float(data_set.get_size()) / batch_size))
        count = 0
        data_set.reset_cur_index()
        prob_list = []

        while True:
            batch_x, _ = data_set.next_batch(batch_size, False)
            if not batch_x:
                break

            batch_x = (batch_x - self.mean_x) / (self.std_x + self.EPLISION)
            feed_dict = {self.__image: batch_x, self.__keep_prob: 1.0}

            prob = self.sess.run(self.__prob, feed_dict)
            prob_list.append( prob[:, 1].transpose() )

            del batch_x

            count += 1
            progress = float(count) / times * 100
            self.echo('\r >> measuring progress: %.2f%% | %d \t' % (progress, times), False)

        return np.hstack(prob_list).transpose()


    ''' 主函数 '''
    def run_i(self, pig_id):
        self.echo('\nStart training %d net ... ' % pig_id)

        # self.get_summary_path(pig_id)
        self.reinit(pig_id)

        # 生成模型
        self.model()

        # 计算 loss
        self.get_loss()

        self.__get_log_loss(pig_id)

        # 正则化
        self.__ch_loss_regular = self.regularize_trainable(self.__ch_log_loss, self.REGULAR_BETA[pig_id])
        # self.__loss_regular = self.regularize_trainable(self.__loss, self.REGULAR_BETA)
        # self.__log_loss_regular = self.regularize_trainable(self.__log_loss, self.REGULAR_BETA)

        # 生成训练的 op
        train_op = self.get_train_op(self.__ch_loss_regular, self.__learning_rate, self.global_step)

        self.__get_accuracy()

        # # tensorboard 相关记录
        # self.__summary()

        # 初始化所有变量
        self.init_variables()

        # # TensorBoard merge summary
        # self.merge_summary()

        mean_train_loss = 0
        mean_train_log_loss = 0
        # mean_train_ch_log_loss = 0
        mean_train_accuracy = 0

        best_val_log_loss = 999999
        incr_val_log_loss_times = 0

        self.echo('\nepoch:')

        moment = 0.975
        self.__running_mean = None
        self.__running_std = None

        best_mean = 0
        best_std = 0.1

        self.__train_set_list[pig_id].start_thread()
        self.__val_set_list[pig_id].start_thread()

        for step in range(self.__steps):
            if step % self.SHOW_PROGRESS_FREQUENCY == 0:
                epoch_progress = float(step) % self.__iter_per_epoch / self.__iter_per_epoch * 100.0
                step_progress = float(step) / self.__steps * 100.0
                self.echo('\r step: %d (%d|%.2f%%) / %d|%.2f%% \t\t' % (step, self.__iter_per_epoch, epoch_progress,
                                                                       self.__steps, step_progress), False)

            batch_x, batch_y = self.__train_set_list[pig_id].next_batch(self.BATCH_SIZE)

            reduce_axis = tuple(range(len(batch_x.shape) - 1))
            _mean = np.mean(batch_x, axis=reduce_axis)
            _std = np.std(batch_x, axis=reduce_axis)
            self.__running_mean = moment * self.__running_mean + (1 - moment) * _mean if type(self.__running_mean) != type(None) else _mean
            self.__running_std = moment * self.__running_std + (1 - moment) * _std if type(self.__running_std) != type(None) else _std
            batch_x = ( batch_x - _mean ) / (_std + self.EPLISION)

            feed_dict = {self.__image: batch_x, self.__label: batch_y, self.__keep_prob: self.KEEP_PROB, self.__size: batch_y.shape[0]}
            _, train_loss, train_log_loss, train_accuracy = self.sess.run([train_op, self.__loss, self.__log_loss, self.__accuracy], feed_dict)

            mean_train_accuracy += train_accuracy
            mean_train_loss += train_loss
            mean_train_log_loss += train_log_loss
            # mean_train_ch_log_loss += train_ch_log_loss

            if step % self.__iter_per_epoch == 0 and step != 0:
                epoch = int(step // self.__iter_per_epoch)
                self.mean_x = self.__running_mean
                self.std_x = self.__running_std * (self.BATCH_SIZE / float(self.BATCH_SIZE - 1))

                mean_train_accuracy /= self.__iter_per_epoch
                mean_train_loss /= self.__iter_per_epoch
                mean_train_log_loss /= self.__iter_per_epoch
                # mean_train_ch_log_loss /= self.__iter_per_epoch

                # self.echo('\n epoch: %d  train_loss: %.6f  log_loss:    train_accuracy: %.6f \t ' % (epoch, mean_train_loss, mean_train_accuracy))
                #
                # feed_dict[self.__mean_accuracy] = mean_train_accuracy
                # feed_dict[self.__mean_loss] = mean_train_loss
                # feed_dict[self.__mean_log_loss] = mean_train_log_loss
                # # # feed_dict[self.__mean_ch_log_loss] = mean_train_ch_log_loss
                # self.add_summary_train(feed_dict, epoch)

                del batch_x
                del batch_y

                # 测试 校验集 的 loss
                mean_val_accuracy, mean_val_loss, mean_val_log_loss = self.__measure(self.__val_set_list[pig_id], 100)
                # batch_val_x, batch_val_y = self.__val_set_list[pig_id].next_batch(self.BATCH_SIZE)
                #
                # batch_val_x = (batch_val_x - self.mean_x) / (self.std_x + self.EPLISION)

                # feed_dict = {self.__mean_accuracy: mean_val_accuracy, self.__mean_loss: mean_val_loss,
                #              self.__mean_log_loss: mean_val_log_loss}
                # self.add_summary_val(feed_dict, epoch)

                # del batch_val_x
                # del batch_val_y

                echo_str = '\n\t net: %d  epoch: %d  train_loss: %.6f  train_log_loss: %.6f  train_accuracy: %.6f  ' \
                           'val_loss: %.6f val_log_loss: %.6f  val_accuracy: %.6f' % (pig_id, epoch, mean_train_loss,
                                                                    mean_train_log_loss, mean_train_accuracy,
                                                            mean_val_loss, mean_val_log_loss, mean_val_accuracy)

                mean_train_accuracy = 0
                mean_train_loss = 0
                mean_train_log_loss = 0

                if best_val_log_loss > mean_val_log_loss:
                    best_val_log_loss = mean_val_log_loss
                    incr_val_log_loss_times = 0

                    best_mean = self.mean_x
                    best_std = self.std_x

                    self.echo('%s  best  ' % echo_str, False)
                    self.save_model_w_b(pig_id)
                    # self.save_model()  # 保存模型

                else:
                    incr_val_log_loss_times += 1
                    self.echo('%s  incr_times: %d \n' % (echo_str, incr_val_log_loss_times), False)

                    if incr_val_log_loss_times > self.MAX_VAL_ACCURACY_DECR_TIMES:
                        break

            else:
                del batch_x
                del batch_y

        # self.close_summary()        # 关闭 TensorBoard

        # self.__test_set.start_thread()

        self.restore_model_w_b(pig_id)    # 恢复模型
        self.rebuild_model()        # 重建模型
        self.get_loss()             # 重新 get loss
        self.__get_accuracy()
        self.__get_log_loss(pig_id)

        self.init_variables()       # 重新初始化变量

        mean_train_accuracy, mean_train_loss, mean_train_log_loss = self.__measure(self.__train_set_list[pig_id])
        mean_val_accuracy, mean_val_loss, mean_val_log_loss = self.__measure(self.__val_set_list[pig_id])
        # mean_test_accuracy, mean_test_loss, mean_test_log_loss = self.__measure(self.__test_set)

        self.__result.append([pig_id, mean_train_accuracy, mean_train_loss, mean_train_log_loss,
                              mean_val_accuracy, mean_val_loss, mean_val_log_loss])

        self.__train_set_list[pig_id].stop()  # 关闭获取数据线程
        self.__val_set_list[pig_id].stop()  # 关闭获取数据线程
        # self.__test_set.stop()  # 关闭获取数据线程

        self.echo('\nFinish training %d net ' % pig_id)

        self.sess.close()

        # self.kill_tensorboard_if_runing()


    def run(self):
        self.__result = [
            # [0, 0.953552, 0.193088, 2.319002, 0.944293, 0.233021, 2.353769],
            # [1, 0.957576, 0.135455, 2.257838, 0.922619, 0.224127, 2.325161],
        ]

        except_list = []

        for i in range(self.NUM_PIG):
            # if i in except_list:
            #     continue
            # if '2.7' not in sys.version and i < 15:
            #     continue
            if '2.7' in sys.version and i < 17:
                continue
            elif '2.7' not in sys.version and (i >= 15 or i <= 3):
                continue
            self.run_i(i)

            for ret in self.__result:
                pig_id, mean_train_accuracy, mean_train_loss, mean_train_log_loss, \
                    mean_val_accuracy, mean_val_loss, mean_val_log_loss = ret
                self.echo('\n*************************************************')
                self.echo('net: %d  train_accuracy: %.6f  train_loss: %.6f  train_log_loss: %.6f  ' % (pig_id,
                                                    mean_train_accuracy, mean_train_loss, mean_train_log_loss))
                self.echo('net: %d  val_accuracy: %.6f  val_loss: %.6f  val_log_loss: %.6f  ' % (pig_id,
                                                    mean_val_accuracy, mean_val_loss, mean_val_log_loss))
                self.echo('*********************************')


    @staticmethod
    def np_softmax(x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=0)


    @staticmethod
    def np_log_loss(prob, label):
        prob = min( max(prob, 1e-15), 1 - 1e-15 )
        return - np.sum( np.multiply( label, np.log(prob) ) ) / label.shape[0]


    def test_i_tf(self, i):
        self.echo('\nTesting %d net ... ' % i)

        self.reinit(i)

        self.restore_model()

        self.__image = self.get_variable_by_name('X:0')
        self.__label = self.get_variable_by_name('y:0')
        self.__size = self.get_variable_by_name('size:0')
        self.__keep_prob = self.get_variable_by_name('keep_prob:0')

        self.__loss = self.get_variable_by_name('loss/Mean:0')
        self.__log_loss = self.get_variable_by_name('log_loss/Neg:0')
        self.__accuracy = self.get_variable_by_name('accuracy/truediv:0')

        self.sess.run(tf.global_variables_initializer())

        self.__train_set_list[i].start_thread()
        self.__val_set_list[i].start_thread()

        mean_train_accuracy, mean_train_loss, mean_train_log_loss = self.__measure(self.__train_set_list[i])
        mean_val_accuracy, mean_val_loss, mean_val_log_loss = self.__measure(self.__val_set_list[i])

        self.__train_set_list[i].stop()
        self.__val_set_list[i].stop()

        self.echo('\n*************************************************')
        self.echo('net: %d  train_accuracy: %.6f  train_loss: %.6f  train_log_loss: %.6f  ' % (i,
                                                                                               mean_train_accuracy,
                                                                                               mean_train_loss,
                                                                                               mean_train_log_loss))
        self.echo('net: %d  val_accuracy: %.6f  val_loss: %.6f  val_log_loss: %.6f  ' % (i,
                                                                                         mean_val_accuracy,
                                                                                         mean_val_loss,
                                                                                         mean_val_log_loss))
        self.echo('*********************************')

        self.sess.close()

        self.echo('Finish testing ')



    def test_i(self, i):
        self.echo('\nTesting %d net ... ' % i)

        self.reinit(i)

        self.restore_model_w_b(i)

        self.rebuild_model()

        self.get_loss()

        self.__get_accuracy()

        self.__get_log_loss(i)

        self.echo('\nInit variables ...')
        self.init_variables()

        # train_prob_list = self.__measure_prob(self.__train_data)
        # val_prob_list = self.__measure_prob(self.__val_data)

        self.__train_set_list[i].start_thread()
        self.__val_set_list[i].start_thread()

        self.echo('\nStart measuring ... ')
        mean_train_accuracy, mean_train_loss, mean_train_log_loss = self.__measure(self.__train_set_list[i])
        mean_val_accuracy, mean_val_loss, mean_val_log_loss = self.__measure(self.__val_set_list[i])

        self.echo('\n*************************************************')
        self.echo('net: %d  train_accuracy: %.6f  train_loss: %.6f  train_log_loss: %.6f  ' % (i,
                                                                                               mean_train_accuracy,
                                                                                               mean_train_loss,
                                                                                               mean_train_log_loss))
        self.echo('net: %d  val_accuracy: %.6f  val_loss: %.6f  val_log_loss: %.6f  ' % (i,
                                                                                         mean_val_accuracy,
                                                                                         mean_val_loss,
                                                                                         mean_val_log_loss))
        self.echo('*********************************')

        self.sess.close()

        self.__train_set_list[i].stop()
        self.__val_set_list[i].stop()

        self.echo('Finish testing ')



    def test(self):
        # self.__train_prob_list = []
        # self.__val_prob_list = []
        self.__prob_list = []

        self.__data = load.TestBData()

        # self.__train_data = load.TestData(0.0, 0.8)
        # self.__val_data = load.TestData(0.8, 1.0)

        self.echo('\nStart testing ... ')
        for i in range(self.NUM_PIG):
            self.echo('  testing %d net ... ' % i)

            self.reinit(i)

            self.restore_model_w_b(i)

            self.rebuild_model()

            self.get_loss()

            self.__get_accuracy()

            self.__get_log_loss(i)

            self.init_variables()

            prob_list = self.__measure_prob(self.__data)
            self.__prob_list.append(prob_list)

            # train_prob_list = self.__measure_prob(self.__train_data)
            # val_prob_list = self.__measure_prob(self.__val_data)

            # self.__train_prob_list.append(train_prob_list)
            # self.__val_prob_list.append(val_prob_list)

            self.sess.close()

        self.echo('Finish testing ')

        self.__prob_list = np.hstack(self.__prob_list)
        # self.__train_prob_list = np.hstack(self.__train_prob_list)
        # self.__val_prob_list = np.hstack(self.__val_prob_list)

        self.__prob_list = self.np_softmax(self.__prob_list)
        # self.__train_prob_list = self.np_softmax(self.__train_prob_list)
        # self.__val_prob_list = self.np_softmax(self.__val_prob_list)

        id_list = self.__data.get_label_list()
        # train_label_list = self.__train_data.get_label_list()
        # val_label_list = self.__val_data.get_label_list()

        data = []
        for i, pig_id in enumerate(id_list):
            tmp_prob_list = self.__prob_list[i]
            for class_no, prob in enumerate(tmp_prob_list.reshape(-1)):
                data.append([pig_id, class_no + 1, '%.10f' % prob])

        if not os.path.isdir(self.RESULT_DIR):
            os.mkdir(self.RESULT_DIR)

        self.echo('\nSaving result to %s ... ' % self.RESULT_FILE_PATH)
        data_len = len(data)

        with open(self.RESULT_FILE_PATH, 'w') as f:
            writer = csv.writer(f)

            for i, line in enumerate(data):
                progress = float(i + 1) / data_len * 100.0
                self.echo('\r  >> progress: %.6f ' % progress, False)

                writer.writerow(line)

        self.echo('Finish saving result ')

        # train_log_loss = self.np_log_loss(self.__train_prob_list, train_label_list)
        # val_log_loss = self.np_log_loss(self.__val_prob_list, val_label_list)

        # self.echo('\n****************************************')
        # self.echo('train_log_loss: %.8f' % train_log_loss)
        # self.echo('val_log_loss: %.8f' % val_log_loss)


    def use_model(self, np_image):
        if not self.__has_rebuild:
            self.restore_model_w_b()    # 恢复模型
            self.rebuild_model()        # 重建模型
            self.get_loss()  # 重新 get loss
            self.__get_accuracy()
            self.__get_log_loss()

            self.init_variables()       # 初始化所有变量
            self.__has_rebuild = True

        np_image = np.expand_dims(np_image, axis=0)

        np_image = (np_image - self.mean_x) / (self.std_x + self.EPLISION)

        feed_dict = {self.__image: np_image, self.__keep_prob: 1.0}
        output = self.sess.run(self.__output, feed_dict)

        return output[0]


    # def test(self):
    #
    #     self.restore_model_w_b()  # 恢复模型
    #     self.rebuild_model()  # 重建模型
    #
    #     self.get_loss()  # 重新 get loss
    #     self.__get_accuracy()
    #     self.__get_log_loss()
    #
    #     self.echo('init_variables ... ')
    #
    #     self.init_variables()  # 重新初始化变量
    #     self.__has_rebuild = True
    #
    #     # # ********************************************
    #     # self.echo('\ncalculating mean std ... ')
    #     #
    #     # moment = 0.975
    #     # self.__running_mean = None
    #     # self.__running_std = None
    #     #
    #     # times = int(math.ceil(float(self.__train_size) / self.BATCH_SIZE))
    #     # times = min(times, 1000)
    #     # for i in range(times):
    #     #     progress = float(i + 1) / times * 100
    #     #     self.echo('\r  >> progress: %.6f ' % progress, False)
    #     #
    #     #     batch_x, batch_y = self.__train_set.next_batch(self.BATCH_SIZE)
    #     #
    #     #     reduce_axis = tuple(range(len(batch_x.shape) - 1))
    #     #     _mean = np.mean(batch_x, axis=reduce_axis)
    #     #     _std = np.std(batch_x, axis=reduce_axis)
    #     #
    #     #     self.__running_mean = moment * self.__running_mean + (1 - moment) * _mean if type(
    #     #         self.__running_mean) != type(None) else _mean
    #     #     self.__running_std = moment * self.__running_std + (1 - moment) * _std if type(self.__running_std) != type(
    #     #         None) else _std
    #     #
    #     #     del batch_x
    #     #     del batch_y
    #     #
    #     # self.mean_x = self.__running_mean
    #     # self.std_x = self.__running_std * (self.BATCH_SIZE / float(self.BATCH_SIZE - 1))
    #     #
    #     # self.echo('\nFinish
    #     #  calculating \n')
    #     #
    #     # # ******************************************
    #
    #     mean_train_accuracy, mean_train_loss, mean_train_log_loss = self.__measure(self.__train_set, 100)
    #     mean_val_accuracy, mean_val_loss, mean_val_log_loss = self.__measure(self.__val_set, 100)
    #
    #     self.echo('\ntrain_accuracy: %.6f  train_loss: %.6f  train_log_loss: %.6f  \n' % (mean_train_accuracy,
    #                                                                                   mean_train_loss,
    #                                                                                   mean_train_log_loss))
    #     self.echo('\nval_accuracy: %.6f  val_loss: %.6f  val_log_loss: %.6f  ' % (mean_val_accuracy,
    #                                                                             mean_val_loss, mean_val_log_loss))
    #
    #     # batch_x, batch_y = self.__val_set.next_batch(3)
    #     #
    #     # for x in batch_x:
    #     #     self.echo('\n************************')
    #     #     self.echo(self.use_model(x))
    #
    #     self.__train_set.stop()
    #     self.__val_set.stop()
    #
    #     self.echo('\ndone')



o_vgg = VGG16()
o_vgg.run()
# o_vgg.test()
# o_vgg.test_i(0)
# o_vgg.test_i_tf(0)
