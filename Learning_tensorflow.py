# coding=utf-8
import tensorflow as tf
#————————————task_one————————————————#
# matrix1 = tf.constant([[3.0, 3.0]])
# matrix2 = tf.constant([[2.0], [2.0]])
#
# product = tf.matmul(matrix1, matrix2)
#
# sess = tf.Session()
#
# result = sess.run(product)
# print result
# sess.close()

#————————————task_two————————————————#

state = tf.Variable(0, name="counter")
# 添加一个op（节点operation），其作用是使得state增加1
one = tf.constant(1)  # 定义一个[1]的矩阵
new_value = tf.add(state, one)
update = tf.assign(state, new_value)#tf.assign(变量，新值)分配一个新的值给变量

#启动图前必须先初始化op

init_op = tf.global_variables_initializer()

#启动图，运行op

with tf.Session() as sess:
    #先初始化op
    sess.run(init_op)
    #打印初始的state
    print sess.run(state)
    #运行op， 先更新state，再打印state
    for _ in range(3):
        sess.run(update)
        print sess.run(state)