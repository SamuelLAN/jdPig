#!/usr/bin/Python
# -*- coding: utf-8 -*-
import tensorflow as tf
from math import sqrt
from numpy import hstack
import re
import sys
import os
from multiprocessing import Process


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
      get_loss loss 函数
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
    MODEL_NAME = 'model_name'           # 模型的名称

    EPOCH_TIMES = 100                   # 迭代的 epoch 次数
    BATCH_SIZE = 100                    # 随机梯度下降的 batch 大小

    BASE_LEARNING_RATE = 0.1            # 初始 学习率
    DECAY_RATE = 0.95                   # 学习率 的 下降速率

    DROPOUT_RATE = 0.8                  # 卷机神经网络 使用的 dropout 配置
    DROPOUT_LIST = None                 # 全连接模型 使用的 dropout 配置
    MODEL = []                          # 深度模型的配置

    TENSORBOARD_SHOW_IMAGE = False      # 默认不将 image 显示到 TensorBoard，以免影响性能

    DEBUG = True

    # ******************************** 基类默认初始化的操作 ****************************

    def __init__(self):
        self.__init()


    ''' 析构函数 '''
    def __del__(self):
        NN.kill_tensorboard_if_runing()
        # self.tbProcess.join(10)
        self.tbProcess.terminate()


    ''' 初始化 '''
    def __init(self):
        self.net = []                                       # 存放每层网络的 feature map
        self.WList = []                                     # 存放权重矩阵的 list
        self.bList = []                                     # 存放偏置量的 list

        self.__modelPath = ''
        self.__get_model_path()                               # 生成存放模型的文件夹 与 路径

        self.__summaryPath = ''
        self.__get_summary_path()

        self.globalStep = self.get_global_step()              # 记录全局训练状态的 global step

        self.init()                                         # 执行定制化的 初始化操作

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
    def get_loss(self):
        pass


    ''' 主函数 '''
    def run(self):
        pass

    # *************************** 初始化变量 ****************************

    ''' 初始化所有变量 '''
    def init_variables(self):
        self.sess.run(tf.global_variables_initializer())


    ''' 初始化权重矩阵 '''
    @staticmethod
    def init_weight(shape):
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
    def init_bias(shape):
        if len(shape) == 4:
            nodes = shape[2]
        else:
            nodes = shape[-1]

        return tf.Variable(tf.zeros([nodes]), name='bias')


    ''' 获取全局的训练 step '''
    @staticmethod
    def get_global_step():
        return tf.Variable(0, name='global_step', trainable=False)


    ''' 获取随迭代次数下降的学习率 '''
    @staticmethod
    def get_learning_rate(base_learning_rate, cur_step, decay_times, decay_rate=0.95, staircase=False):
        learning_rate = tf.train.exponential_decay(
            base_learning_rate,     # Base learning rate.
            cur_step,               # Current index into the dataset.
            decay_times,            # Decay step.
            decay_rate,             # Decay rate.
            staircase=staircase
        )
        tf.summary.scalar('learning_rate', learning_rate)
        return learning_rate

    # *************************** 保存模型 ***************************

    ''' 保存模型 '''
    def save_model(self):
        self.saver.save(self.sess, self.__get_model_path())
        
    
    ''' 恢复模型 '''
    def restore_model(self):
        self.saver.restore(self.sess, self.__get_model_path())


    ''' 获取存放模型的路径 '''
    def __get_model_path(self):
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

    def merge_summary(self):
        self.__mergedSummaryOp = tf.summary.merge_all()
        if tf.gfile.Exists(self.__summaryPath):
            tf.gfile.DeleteRecursively(self.__summaryPath)
        self.__summaryWriterTrain = tf.summary.FileWriter(
            os.path.join(self.__summaryPath, 'train'), self.sess.graph)
        self.__summaryWriterVal = tf.summary.FileWriter(
            os.path.join(self.__summaryPath, 'validation'), self.sess.graph)

    ''' TensorBoard add sumary training '''

    def add_summary_train(self, feed_dict, step):
        summary_str = self.sess.run(self.__mergedSummaryOp, feed_dict)
        self.__summaryWriterTrain.add_summary(summary_str, step)
        self.__summaryWriterTrain.flush()

    ''' TensorBoard add sumary validation '''

    def add_summary_val(self, feed_dict, step):
        summary_str = self.sess.run(self.__mergedSummaryOp, feed_dict)
        self.__summaryWriterVal.add_summary(summary_str, step)
        self.__summaryWriterVal.flush()

    ''' TensorBoard close '''

    def close_summary(self):
        self.__summaryWriterTrain.close()
        self.__summaryWriterVal.close()


    ''' 输出前 num 个节点的图像到 TensorBoard '''
    def image_summary(self, tensor_4d, num, name, i):
        if not self.TENSORBOARD_SHOW_IMAGE:
            return

        with tf.name_scope('summary'):
            index = self.globalStep % self.BATCH_SIZE       # 让每次输出不同图片，取不同 index 的图片
            shape = list(hstack([-1, [int(j) for j in tensor_4d.shape[1: 3]], 1]))  # 生成的 shape 为 [-1, image_width, image_height, 1]
            image = tf.concat([tensor_4d[index, :, :, j] for j in range(num)], 0)   # 将多幅图像合并在一起

            # 必须 reshape 成 [?, image_size, image_size, 1 或 3 或 4]
            image = tf.reshape(image, shape)
            tf.summary.image('%s_%d' % (name, i), image)


    ''' 获取 summary path '''

    def __get_summary_path(self):
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
            self.__remove_file_recursive(summary_dir)

        dirs = ['train', 'validation']
        for dir_name in dirs:
            dir_path = os.path.join(summary_dir, dir_name)
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)
            else:
                self.__remove_file_recursive(dir_path)

        self.__summaryPath = summary_dir

        # 异步在终端运行 tensorboard
        self.run_tensorboard(self.__summaryPath)
        return self.__summaryPath


    @staticmethod
    def __remove_file_recursive(dir_path):
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    @staticmethod
    def cmd(command):
        return os.popen(command).read()


    ''' 若 tensorboard 正在运行, kill tensorboard 进程 '''
    @staticmethod
    def kill_tensorboard_if_runing():
        try:
            # 检查 tensorboard 是否正在运行
            ps_cmd = NN.cmd('ps aux | grep tensorboard')
            ps_cmd = ps_cmd.replace('\r', '').split('\n')

            reg = re.compile(r'\bpython\s+')
            reg_space = re.compile(r'\s+')

            for line in ps_cmd:
                # 若 tensorboard 正在运行, kill 进程
                if reg.search(line):
                    pid = int(reg_space.split(line)[1])
                    NN.cmd('kill -9 %d' % pid)

        except Exception, ex:
            print ex


    '''
     同步状态 自动在终端打开 cmd (默认端口为 6006，port 参数可以自己指定端口)
    '''
    @staticmethod
    def __run_tensorboard_sync(path, port=6006):
        try:
            NN.cmd('tensorboard --logdir=%s --port=%d' % (path, port))
            # NN.cmd('source activate python27;tensorboard --logdir=%s --port=%d' % (path, port))
        except Exception, ex:
            print ex


    ''' 异步状态，自动在终端打开 cmd (默认端口为 6006，port 参数可以自己指定端口) '''
    def run_tensorboard(self, path, port=6006):
        NN.kill_tensorboard_if_runing()
        self.tbProcess = Process(target=NN.__run_tensorboard_sync, args=(path, port))
        self.tbProcess.start()

    # **************************** 常用模型 ***************************

    ''' 全连接模型 '''
    def full_connect_model(self, X, shape_list, is_train = True):
        len_shape_list = len(shape_list)            # 获取 shape_list 的长度

        # 训练时，需要初始化变量：权重矩阵、偏置量
        if is_train:
            for i, shape in enumerate(shape_list):
                with tf.name_scope('layer_%d' % (i + 1,)):
                    W = self.init_weight(shape)      # 初始化权重矩阵
                    b = self.init_bias(shape)        # 初始化偏置量

                    self.WList.append(W)            # 将权重矩阵存放到 公有变量 WList
                    self.bList.append(b)            # 将偏置量存放到 公有变量 BList

        # 正向推导
        for i, shape in enumerate(shape_list):
            with tf.name_scope('layer_%d' % (i + 1,)):
                W = self.WList[i]                   # 获取权重矩阵
                b = self.bList[i]                   # 获取偏置量

                if i == 0:                          # 获取输入
                    a = X

                h = self.cal_h(a, W, b)              # 计算 h
                if i != len_shape_list - 1:         # 若不是最后一层，使用 激活函数
                    a = self.activate(h)

                    if is_train and self.DROPOUT_LIST and self.DROPOUT_LIST[i]:     # 若在训练，且使用 dropout
                        a = self.dropout(a, self.DROPOUT_LIST[i])
                else:
                    return h


    ''' 
      深度模型
      在 self.MODEL 中配置即可
        self.MODEL 为 list，存放的元素为按顺序依次往下的网络的层
      配置支持的 type:
        conv：卷积
            for example:
            {
                'name': 'conv_3',   # name_scope，默认为 'type_第i层'; 所有 type 都有 name，下面不再详写
                'type': 'conv',
                'shape': [NUM_CHANNEL, 32], # 若有 'W' 和 'b'，可以没有该值
                'k_size': [5, 5],   # 若有 'W'，可以没有该值
                'activate': True,   # 默认为 True
                'W': W,             # kernel；若没有该值，会自动根据 k_size 以及 shape 初始化
                'b': b,             # bias; 若没有该值，会自动根据 shape 初始化
            }
        tr_conv: 反卷积(上采样)
            for example:
            {
                'type': 'tr_conv',
                'shape': [NUM_CHANNEL, 32],
                'k_size': [16, 16],     # 一般为 stride 的两倍
                'output_shape': [1, 256, 256, NUM_CHANNEL], # 若指定了 output_shape_index 或 output_shape_x，
                                                            # 无需设置该项；该项可空
                'output_shape_index': 3,    # output_shape 跟第 3 层网络的 shape 一致；层数从 0 开始
                                            # 若指定了 output_shape 或 output_shape_x，无需设置该项；该项可空
                'output_shape_x': [None, None, None, NUM_CHANNEL] # 使用输入层的 shape 作为 output_shape, 
                                            # 并且在此基础上，根据提供的 'output_shape_x' 更改 shape (若元素不为 None)
                                            # 若指定了 output_shape 或 output_shape_index，无需设置该项；该项可空
                'stride': 8,            # 默认为 2；stride 相当于上采样的倍数
            }
        pool: 池化
            for example:
            {
                'type': 'pool',
                'k_size': [2, 2],
                'pool_type': 'max',     # 只支持 'max' 或 'avg'; 若没有设置该项，默认为 max_pool
            }
        fc: 全连接
            for example:
            {
                'type': 'fc',
                'shape': [1024, NUM_CLASSES]
                'activate': False,
            }
        dropout: dropout
            for example:
            {
                'type': 'dropout',
                'keep_prob': 0.5,
            }
        add: 将上一层网络的输出 与 第 layer_index 层网络 sum
            for example:
            {
                'type': 'add',
                'layer_index': 4, # layer_index 是从 0 开始的, 且必须小于当前 index
            }
             
    '''
    def deep_model(self, X, dropout = None, is_train = True):
        if is_train:
            for i, config in enumerate(self.MODEL):
                _type = config['type'].lower()
                name = '%s_%d' % (_type, i + 1) if 'name' not in config else config['name']

                # 卷积层
                if _type == 'conv':
                    with tf.name_scope(name):
                        W = self.init_weight(config['k_size'] + config['shape']) if not 'W' in config else config['W']
                        b = self.init_bias(config['shape']) if not 'b' in config else config['b']
                        self.WList.append(W)
                        self.bList.append(b)

                # 反卷积层 (上采样 transpose conv)
                elif _type == 'tr_conv':
                    with tf.name_scope(name):
                        W = self.init_weight(config['k_size'] + config['shape']) if not 'W' in config else config['W']
                        b = self.init_bias(config['shape'][:-2] + [config['shape'][-1], config['shape'][-2]]) if not 'b' in config else config['b']
                        self.WList.append(W)
                        self.bList.append(b)

                # 全连接层
                elif _type == 'fc':
                    with tf.name_scope(name):
                        W = self.init_weight(config['shape'])
                        b = self.init_bias(config['shape'])
                        self.WList.append(W)
                        self.bList.append(b)

                else:
                    self.WList.append(None)
                    self.bList.append(None)



        a = X
        if is_train:    # 将图像输出到 TensorBoard
            self.image_summary(a, 1, 'input', 1)

        self.echo('\nStart building model ...')

        model_len = len(self.MODEL)
        for i, config in enumerate(self.MODEL):
            _type = config['type'].lower()
            name = '%s_%d' % (_type, i + 1) if 'name' not in config else config['name']
            # self.echo('building %s layer ...' % name)

            # 卷积层
            if _type == 'conv':
                with tf.name_scope(name):
                    a = tf.add(self.conv2d(a, self.WList[i]), self.bList[i])
                    if not 'activate' in config or config['activate']:
                        a = self.activate(a)

                    if is_train:    # 将前 3 个节点的图像输出到 TensorBoard
                        self.image_summary(a, 3, _type, i)

            # 池化层
            elif _type == 'pool':
                with tf.name_scope(name):
                    if not 'pool_type' or config['pool_type'] == 'max':
                        a = self.max_pool(a, config['k_size'])
                    else:
                        a = self.avg_pool(a, config['k_size'])

                    if is_train:    # 将前 3 个节点的图像输出到 TensorBoard
                        self.image_summary(a, 3, _type, i)

            # 全连接层
            elif _type == 'fc':
                with tf.name_scope(name):
                    x = tf.reshape(a, [-1, config['shape'][0]])
                    a = tf.add(tf.matmul(x, self.WList[i]), self.bList[i])

                    if ('activate' not in config and i < model_len - 1) or config['activate']:
                        a = self.activate(a)

            # 训练的 dropout
            elif _type == 'dropout':
                with tf.name_scope(name):
                    a = self.dropout(a, dropout)

            # 反卷积层(上采样层)
            elif _type == 'tr_conv':
                with tf.name_scope(name):
                    if 'output_shape' in config:
                        output_shape = config['output_shape']
                    elif 'output_shape_index' in config:
                        output_shape = tf.shape( self.net[ config['output_shape_index'] ] )
                    elif 'output_shape_x' in config:
                        output_shape = tf.shape(X)
                        for j, val_j in enumerate(config['output_shape_x']):
                            if not val_j:
                                continue
                            tmp = tf.Variable([1 if k != j else 0 for k in range(4)], tf.int8)
                            output_shape *= tmp
                            tmp = tf.Variable([0 if k != j else val_j for k in range(4)], tf.int8)
                            output_shape += tmp
                    else:
                        output_shape = None

                    stride = config['stride'] if 'stride' in config else 2
                    a = self.conv2d_transpose_stride(a, self.WList[i], self.bList[i], output_shape, stride)

            # 将上一层的输出 与 第 layer_index 层的网络相加
            elif _type == 'add':
                with tf.name_scope(name):
                    a = tf.add(a, self.net[config['layer_index']])

            self.net.append(a)

        self.echo('Finish building model')

        return a


    # **************************** 与 模型有关 的 常用函数 *************************

    ''' 计算 h '''
    def cal_h(self, last_a, W, b):
        return tf.add(tf.matmul(last_a, W), b, name='h')


    ''' 激活函数 '''
    def activate(self, h):
        return tf.nn.relu(h, name='a')


    ''' 2D 卷积 '''
    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


    ''' 2D 卷积 并加上 bias '''
    @staticmethod
    def conv2d_bias(x, W, b):
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.bias_add(conv, b)


    ''' 2D 反卷积(transpose conv) 并 加上 bias '''
    @staticmethod
    def conv2d_transpose_stride(x, W, b, output_shape=None, stride=2):
        # 若没有设置 output_shape
        if type(output_shape) == type(None):
            output_shape = x.get_shape().as_list()
            output_shape[1] *= stride
            output_shape[2] *= stride
            output_shape[3] = W.get_shape().as_list()[2]
        conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
        return tf.nn.bias_add(conv, b)


    ''' max pooling '''
    @staticmethod
    def max_pool(x, k_size):
        k_size = list(hstack([1, k_size, 1]))
        return tf.nn.max_pool(x, ksize=k_size, strides=k_size, padding='SAME')


    ''' mean pooling '''
    @staticmethod
    def avg_pool(x, k_size):
        k_size = list(hstack([1, k_size, 1]))
        return tf.nn.avg_pool(x, ksize=k_size, strides=k_size, padding='SAME')

    # *************************** 与 训练有关 的 常用函数 ***************************

    ''' 获取 train_op '''
    def get_train_op(self, loss, learning_rate, global_step):
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
        return tf.nn.dropout(a, keep_prob)

    # ********************** 其他常用函数 (与 DL 无关) *********************

    ''' 输出展示 '''
    @staticmethod
    def echo(msg, crlf=True):
        if crlf:
            print msg
        else:
            sys.stdout.write(msg)
            sys.stdout.flush()
