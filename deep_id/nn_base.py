#!/usr/bin/Python
# -*- coding: utf-8 -*-
import tensorflow as tf
from math import sqrt
from numpy import hstack
import sys
import os

'''
 网络的基类
 提供了：
    基类默认初始化的操作
      子类请不要重载构造函数
      要初始化变量可以通过 重载 self.init 函数
    子类需要实现的接口
      init 初始化各种变量
      load 加载数据
      model 重载模型
      inference 前向传导 (若想使用 TensorBoard 训练时同时看到 训练集、测试集 的准确率，需要实现 inference)
      getLoss loss 函数
      run 主函数
    初始化变量
      权重矩阵的初始化
      偏置量的初始化
      学习率的初始化
      执行 tf 的所有变量的初始化
    保存模型
    TensorBoard summary
    常用模型
      全连接模型
      深度模型 CNN
    与 模型有关 的 常用函数
      计算 h
      激活函数
      卷积函数
      pooling 函数
    与 训练有关 的 常用函数
      get train op
    防止 overfitting 的 trick
      regularize
      dropout
    他常用函数 (与 DL 无关)
      echo
'''


class NN:
    MODEL_NAME = 'model_name'  # 模型的名称

    EPOCH_TIMES = 100  # 迭代的 epoch 次数
    BATCH_SIZE = 100  # 随机梯度下降的 batch 大小

    BASE_LEARNING_RATE = 0.1  # 初始 学习率
    DECAY_RATE = 0.95  # 学习率 的 下降速率

    DROPOUT_LIST = None  # 全连接模型 使用的 dropout 配置
    MODEL = []  # 深度模型的配置

    TENSORBOARD_SHOW_IMAGE = False  # 默认不将 image 显示到 TensorBoard，以免影响性能

    # ******************************** 基类默认初始化的操作 ****************************

    def __init__(self):
        self.__init()

    ''' 初始化 '''

    def __init(self):
        self.WList = []  # 存放权重矩阵的 list
        self.bList = []  # 存放偏置量的 list

        self.__modelPath = ''
        self.__getModelPath()  # 生成存放模型的文件夹 与 路径

        self.__summaryPath = ''
        self.__getSummaryPath()

        self.globalStep = self.getGlobalStep()  # 记录全局训练状态的 global step

        self.init()  # 执行定制化的 初始化操作

        self.saver = tf.train.Saver()
        self.sess = tf.Session()

    # ******************************* 子类需要实现的接口 *******************************

    ''' 初始化各种 变量 常量 '''

    def init(self):
        pass

    ''' 加载数据 '''

    def load(self):
        pass

    ''' 模型 '''

    def model(self):
        pass

    ''' 前向推导 '''

    def inference(self):
        pass

    ''' 计算 loss '''

    def getLoss(self):
        pass

    ''' 主函数 '''

    def run(self):
        pass

    # *************************** 初始化变量 ****************************

    ''' 初始化所有变量 '''

    def initVariables(self):
        self.sess.run(tf.global_variables_initializer())

    ''' 初始化权重矩阵 '''

    @staticmethod
    def initWeight(shape):
        if len(shape) == 4:
            input_nodes = shape[1] * shape[2]
        else:
            input_nodes = shape[0]

        return tf.Variable(
            tf.truncated_normal(
                shape,
                stddev=1.0 / sqrt(float(input_nodes)),
            ),
            name='weight'
        )

    ''' 初始化 bias '''

    @staticmethod
    def initBias(shape):
        if len(shape) == 4:
            nodes = shape[2]
        else:
            nodes = shape[-1]

        return tf.Variable(tf.zeros([nodes]), name='bias')

    ''' 获取全局的训练 step '''

    @staticmethod
    def getGlobalStep():
        return tf.Variable(0, name='global_step', trainable=False)

    ''' 获取随迭代次数下降的学习率 '''

    @staticmethod
    def getLearningRate(base_learning_rate, cur_step, batch_size, decay_times, decay_rate=0.95):
        learning_rate = tf.train.exponential_decay(
            base_learning_rate,  # Base learning rate.
            cur_step * batch_size,  # Current index into the dataset.
            decay_times,  # Decay step.
            decay_rate,  # Decay rate.
            staircase=True
        )
        tf.summary.scalar('learning_rate', learning_rate)
        return learning_rate

    # *************************** 保存模型 ***************************

    ''' 保存模型 '''

    def saveModel(self):
        self.saver.save(self.sess, self.__getModelPath())

    ''' 恢复模型 '''

    def restoreModel(self):
        self.saver.restore(self.sess, self.__getModelPath())

    ''' 获取存放模型的路径 '''

    def __getModelPath(self):
        if self.__modelPath:
            return self.__modelPath

        cur_dir = os.path.split(os.path.abspath(__file__))[0]
        model_dir = os.path.join(cur_dir, 'model')

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        model_dir = os.path.join(model_dir, self.MODEL_NAME.split('.')[0])
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        self.__modelPath = os.path.join(model_dir, self.MODEL_NAME)
        return self.__modelPath

    # ************************** TensorBoard summary ************************

    ''' TensorBoard merge summary '''

    def mergeSummary(self):
        self.__mergedSummaryOp = tf.summary.merge_all()
        if tf.gfile.Exists(self.__summaryPath):
            tf.gfile.DeleteRecursively(self.__summaryPath)
        self.__summaryWriter = tf.summary.FileWriter(self.__summaryPath, self.sess.graph)

    ''' TensorBoard add sumary '''

    def addSummary(self, feed_dict, step):
        summary_str = self.sess.run(self.__mergedSummaryOp, feed_dict)
        self.__summaryWriter.add_summary(summary_str, step)
        # self.__summaryWriter.add_graph(self.sess.graph) # @TODO 检验效果
        self.__summaryWriter.flush()

    ''' TensorBoard close '''

    def closeSummary(self):
        self.__summaryWriter.close()

    ''' 输出前 num 个节点的图像到 TensorBoard '''

    def imageSummary(self, tensor_4d, num, name, i):
        if not self.TENSORBOARD_SHOW_IMAGE:
            return

        with tf.name_scope('summary'):
            index = self.globalStep % self.BATCH_SIZE  # 让每次输出不同图片，取不同 index 的图片
            shape = list(hstack([-1, [int(j) for j in tensor_4d.shape[1: 3]], 1]))
            image = tf.concat([tensor_4d[index, :, :, j] for j in range(num)], 0)

            # 必须 reshape 成 [?, image_size, image_size, 1 或 3 或 4]
            image = tf.reshape(image, shape)
            tf.summary.image('%s_%d' % (name, i), image)

    ''' 获取 summary path '''

    def __getSummaryPath(self):
        if self.__summaryPath:
            return self.__summaryPath

        cur_dir = os.path.split(os.path.abspath(__file__))[0]
        summary_dir = os.path.join(cur_dir, 'summary')

        if not os.path.isdir(summary_dir):
            os.mkdir(summary_dir)

        summary_dir = os.path.join(summary_dir, self.MODEL_NAME.split('.')[0])
        if not os.path.isdir(summary_dir):
            os.mkdir(summary_dir)
        else:
            for file_name in os.listdir(summary_dir):
                file_path = os.path.join(summary_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        self.__summaryPath = summary_dir
        return self.__summaryPath

    # **************************** 常用模型 ***************************

    ''' 全连接模型 '''

    def fullConnectModel(self, X, shape_list, is_train=True):
        len_shape_list = len(shape_list)  # 获取 shape_list 的长度

        # 训练时，需要初始化变量：权重矩阵、偏置量
        if is_train:
            for i, shape in enumerate(shape_list):
                with tf.name_scope('layer_%d' % (i + 1,)):
                    W = self.initWeight(shape)  # 初始化权重矩阵
                    b = self.initBias(shape)  # 初始化偏置量

                    self.WList.append(W)  # 将权重矩阵存放到 公有变量 WList
                    self.bList.append(b)  # 将偏置量存放到 公有变量 BList

        # 正向推导
        for i, shape in enumerate(shape_list):
            with tf.name_scope('layer_%d' % (i + 1,)):
                W = self.WList[i]  # 获取权重矩阵
                b = self.bList[i]  # 获取偏置量

                if i == 0:  # 获取输入
                    a = X

                h = self.calH(a, W, b)  # 计算 h
                if i != len_shape_list - 1:  # 若不是最后一层，使用 激活函数
                    a = self.activate(h)

                    if is_train and self.DROPOUT_LIST and self.DROPOUT_LIST[i]:  # 若在训练，且使用 dropout
                        a = self.dropout(a, self.DROPOUT_LIST[i])
                else:
                    return h

    ''' 深度模型 '''

    def deepModel(self, X, image_shape, is_train=True):
        if is_train:
            for i, config in enumerate(self.MODEL):
                _type = config['type'].lower()

                # 卷积层
                if _type == 'conv':
                    with tf.name_scope('conv_%d' % (i + 1,)):
                        W = self.initWeight(config['k_size'] + config['shape'])
                        b = self.initBias(config['shape'])
                        self.WList.append(W)
                        self.bList.append(b)

                # 全连接层
                elif _type == 'fc':
                    with tf.name_scope('fc_%d' % (i + 1,)):
                        W = self.initWeight(config['shape'])
                        b = self.initBias(config['shape'])
                        self.WList.append(W)
                        self.bList.append(b)

                else:
                    self.WList.append(None)
                    self.bList.append(None)

        with tf.name_scope('reshape'):
            shape = list(hstack([-1, image_shape, 1]))
            a = tf.reshape(X, shape)

            if is_train:  # 将图像输出到 TensorBoard
                self.imageSummary(a, 1, 'input', 1)

        model_len = len(self.MODEL)
        for i, config in enumerate(self.MODEL):
            _type = config['type'].lower()

            # 卷积层
            if _type == 'conv':
                with tf.name_scope('conv_%d' % (i + 1,)):
                    h = tf.add(self.conv2D(a, self.WList[i]), self.bList[i])
                    a = self.activate(h)

                    if is_train:  # 将前 3 个节点的图像输出到 TensorBoard
                        self.imageSummary(a, 3, _type, i)

            # 池化层
            elif _type == 'pool':
                with tf.name_scope('pooling_%d' % (i + 1,)):
                    a = self.maxPool(a, config['k_size'])

                    if is_train:  # 将前 3 个节点的图像输出到 TensorBoard
                        self.imageSummary(a, 3, _type, i)

            # 全连接层
            elif _type == 'fc':
                with tf.name_scope('fc_%d' % (i + 1,)):
                    x = tf.reshape(a, [-1, config['shape'][0]])
                    h = tf.add(tf.matmul(x, self.WList[i]), self.bList[i])

                    if i < model_len - 1:
                        a = self.activate(h)
                    else:
                        return h

            # 训练的 dropout
            elif _type == 'dropout' and is_train:
                with tf.name_scope('dropout_%d' % (i + 1,)):
                    a = self.dropout(a, config['keep_prob'])

    # **************************** 与 模型有关 的 常用函数 *************************

    ''' 计算 h '''

    def calH(self, last_a, W, b):
        return tf.add(tf.matmul(last_a, W), b, name='h')

    ''' 激活函数 '''

    def activate(self, h):
        return tf.nn.relu(h, name='a')

    ''' 2D 卷积 '''

    @staticmethod
    def conv2D(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def maxPool(x, k_size):
        k_size = list(hstack([1, k_size, 1]))
        return tf.nn.max_pool(x, ksize=k_size, strides=k_size, padding='SAME')

    # *************************** 与 训练有关 的 常用函数 ***************************

    ''' 获取 train_op '''

    def getTrainOp(self, loss, learning_rate, global_step):
        tf.summary.scalar('loss', loss)  # 记录 loss 到 TensorBoard

        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            return optimizer.minimize(loss, global_step=global_step)

    # ************************* 防止 overfitting 的 trick *************************

    ''' 正则化，默认采用 l2_loss 正则化 '''

    def regularize(self, loss, beta):
        with tf.name_scope('regularize'):
            regularizer = 0.0
            for W in self.WList:
                if not W or len(W.shape) != 2:  # 若不是全连接层的权重矩阵，则不进行正则化
                    continue
                regularizer = tf.add(regularizer, tf.nn.l2_loss(W))
            return tf.reduce_mean(loss + beta * regularizer)

    ''' dropout '''

    def dropout(self, a, keep_prob):
        return tf.nn.dropout(a, tf.constant(keep_prob, dtype=tf.float32))

    # ********************** 其他常用函数 (与 DL 无关) *********************

    ''' 输出展示 '''

    @staticmethod
    def echo(msg, crlf=True):
        if crlf:
            print msg
        else:
            sys.stdout.write(msg)
            sys.stdout.flush()
