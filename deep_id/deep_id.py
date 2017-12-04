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
    DeepID
    
    原图
    框猪图(带背景)
'''
class DeepId(base.NN):
    MODEL_NAME = 'deep_id'      # 模型的名称

    BATCH_SIZE = 10             # 迭代的 epoch 次数
    EPOCH_TIMES = 100           # 随机梯度下降的 batch 大小

    NUM_CLASSES = 30            # 总共分 NUM_CLASSES 类
    NUM_CHANNEL = 3             # 输入 channel

    IMAGE_SHAPE = [39, 39]      # 输入图片的大小
    IMAGE_PH_SHAPE = [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUM_CHANNEL]    # image 的 placeholder 的 shape

    X_LIST_LEN = 6              # 总共有 X_LIST_LEN 个输入，需要训练 X_LIST_LEN 个 CNN

    BASE_LEARNING_RATE = 0.01   # 初始 学习率
    DECAY_RATE = 0.1            # 学习率 的 下降速率

    KEEP_PROB = 0.85            # dropout 的 keep_prob

    DEEP_ID_LAYER_INDEX = -2    # 倒数第二层为 deep_id 层

    SHOW_PROGRESS_FREQUENCY = 2         # 每 SHOW_PROGRESS_FREQUENCY 个 step show 一次进度 progress

    MAX_VAL_ACCURACY_INCR_TIMES = 20    # 校验集 val_accuracy 连续 100 次没有降低，则 early stop

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

        for i in range(self.X_LIST_LEN):
            # 输入
            self.__x_list.append( tf.placeholder(tf.float32, self.IMAGE_PH_SHAPE, name='X_%d' % i) )
            
            # 记录训练 step
            global_step = self.get_global_step()
            self.__global_step_list.append( global_step )

            # 随训练次数增多而衰减的学习率
            self.__learning_rate_list.append( self.get_learning_rate(
                self.BASE_LEARNING_RATE, global_step, self.__steps, self.DECAY_RATE, staircase=False
            ) )

        self.__label = tf.placeholder(tf.float32, [None, self.NUM_CLASSES], name='y')
        # dropout 的 keep_prob
        self.__keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.__has_rebuild = False


    ''' 加载数据 '''
    def load(self):
        sort_list = load.Data.get_sort_list()
        self.__train_set = load.Data(0.0, 0.64, 'train', sort_list)
        # self.__val_set = load.Data(0.64, 0.8, 'validation', sort_list)
        # self.__test_set = load.Data(0.8, 1.0, 'test', sort_list)

        self.__train_size = self.__train_set.get_size()
        # self.__val_size = self.__val_set.get_size()
        # self.__test_size = self.__test_set.get_size()


    ''' 模型 '''
    def model(self):
        for i in range(self.X_LIST_LEN):
            self.__output_list.append( self.deep_model(self.__x_list[i], self.__keep_prob) )


    ''' 重建模型 '''
    def rebuild_model(self):
        self.__output_list = []
        self.__loss_list = []
        
        for i in range(self.X_LIST_LEN):
            self.__output_list.append( self.deep_model_rebuild(self.__x_list[i]) )


    ''' 计算 loss '''
    def get_loss(self):
        with tf.name_scope('loss'):
            for i in range(self.X_LIST_LEN):
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=self.__output_list[i], labels=self.__label), 
                    name='entropy'
                )
                self.__loss_list.append(loss)
                tf.summary.scalar('loss_%d' % i, loss)  # 记录 loss 到 tensorboard


    ''' 获取 train_op list '''
    def __get_train_op_list(self):
        with tf.name_scope('optimizer'):
            for i in range(self.X_LIST_LEN):
                optimizer = tf.train.AdamOptimizer(self.__learning_rate_list[i])
                self.__train_op_list.append( optimizer.minimize(self.__loss_list[i], 
                                                                global_step=self.__global_step_list[i]) )


    # ''' 将图片输出到 tensorboard '''
    # def __summary(self):
    #     with tf.name_scope('summary'):
    #         # 记录 loss 到 tensorboard
    #         self.__loss_placeholder = tf.placeholder(tf.float32, name='loss')
    #         tf.summary.scalar('mean_loss', self.__loss_placeholder)


    ''' 主函数 '''
    def run(self):
        # 生成模型
        self.model()

        # 计算 loss
        self.get_loss()

        # 正则化
        # self.__loss = self.regularize_trainable(self.__loss, self.REGULAR_BETA)

        # 生成训练的 op
        self.__get_train_op_list()

        # # tensorboard 相关记录
        # self.__summary()

        # 初始化所有变量
        self.init_variables()

        # TensorBoard merge summary
        self.merge_summary()

        self.echo('\nStart training ... \nepoch:')

        for step in range(self.__steps):
            if step % self.SHOW_PROGRESS_FREQUENCY == 0:
                epoch_progress = float(step) % self.__iter_per_epoch / self.__iter_per_epoch * 100.0
                step_progress = float(step) / self.__steps * 100.0
                self.echo('\rstep: %d (%d|%.2f%%) / %d|%.2f%% \t\t' % (step, self.__iter_per_epoch, epoch_progress,
                                                                       self.__steps, step_progress), False)

            batch_x_list, batch_y = self.__train_set.next_batch(self.BATCH_SIZE)

            for i in range(self.X_LIST_LEN):
                batch_x = batch_x_list[i]
                train_op = self.__train_op_list[i]
                loss = self.__loss_list[i]

                print 'batch_x.shape:'
                print batch_x.shape

                feed_dict = {self.__x_list[i]: batch_x, self.__label: batch_y, self.__keep_prob: self.KEEP_PROB}
                _, train_loss = self.sess.run([train_op, loss], feed_dict)

                if step % self.__iter_per_epoch == 0 and step != 0:
                    epoch = int(step // self.__iter_per_epoch)

                    self.echo('\n epoch: %d \t net: %d \t loss: %d \r ' % (epoch, i, train_loss))

        self.echo('\nFinish training ')

        self.close_summary()  # 关闭 TensorBoard


o_deep_id = DeepId()
o_deep_id.run()
