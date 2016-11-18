# coding=utf-8
#下载并导入MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/', one_hot=True)

import tensorflow as tf

sess = tf.InteractiveSession()
#x 是输入图 shape=[第一个维度值，维度]，None表示其值大小不一定在这里作为第一个维度值，用以指代batch的大小，意即x的数量不定；784（=28*28）是图片的维度
x = tf.placeholder("float",shape=[None,784])
#y 是输出类每一行为一个10维的one-hot向量（Tensor张量）
#神经网络使用10个出口节点就可以编码表示0-9；
#  1 -> [0,1.0,0,0,0,0,0,0,0]   one_hot表示只有一个出口节点是hot
#  2 -> [0,0.1,0,0,0,0,0,0,0]
#  5 -> [0,0,0,0,0,1.0,0,0,0]
y_ = tf.placeholder("float",shape=[None,10])


#weight_variable  多卷积网络需要多个权值，用此函数生成权值
#tf.truncated_normal(shape,stddev=0.1) 随机生成一个服从截断正态分布的tensor，stddev表示标准偏差值
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#bias_variable  生成偏置项
#tf.constant(0.1,shape=shape) 生成一个大小为0.1的常量tensor
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#卷积使用1步长（strides） 0边距（padding）的模版
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#池化是用2x2大小的模版，ksize：为了输入tensor的四维的窗口的大小，pooling：其实也就是把特征图像区域的一部分求个均值或者最大值，用来代表这部分区域。
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#第一层卷积，他有一个卷积接着一个max pooling完成
#卷积在5x5的patch中算出32个特征[5,5,1,32]其中前两个维度表示patch的大小为5x5，1为输入的通道数，32为输出的通道数
#b_conv1是每一个输出通道的偏置量
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
#为了用这一层，将x变成一个4维向量，第2、3维表示图片的长宽最后一维表示颜色通道 1表示灰度图，3表示rgb图
x_image = tf.reshape(x,[-1,28,28,1])
#把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max——pooling
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积构建更深的网络，把几个类似的层堆叠起来
#每一个5x5的patch会得到64个特征 32个输入通道是因为第一个卷积的输出特征为32个
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#密集连接层
#现在把图像尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#在输出层之前加入dropout层，为了减少过拟合
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层 添加一个softmax层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#训练与评估代码
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(5000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print "step %d, training accuracy %g"%(i, train_accuracy)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print "test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
#准确率达到98.56%