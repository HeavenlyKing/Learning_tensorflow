# coding=utf-8
import tensorflow as tf
#————————————Point_one————————————————#
 matrix1 = tf.constant([[3.0, 3.0]])
 matrix2 = tf.constant([[2.0], [2.0]])

 product = tf.matmul(matrix1, matrix2)

 sess = tf.Session()

 result = sess.run(product)
 print result
 sess.close()

#————————————Point_two————————————————#

state = tf.Variable(0, name="counter")
# 添加一个op（节点operation），其作用是使得state增加1
one = tf.constant(1)  # 定义一个[1]的矩阵
new_value = tf.add(state, one)
update = tf.assign(state, new_value)#tf.assign(变量，新值)分配一个新的值给变量

#启动图前必须先初始化op

init_op = tf.global_variables_initializer()

#启动图，运行op
#tf.Session()默认图
with tf.Session() as sess:
    #先初始化op
    sess.run(init_op)
    #打印初始的state
    print sess.run(state)
    #运行op， 先更新state，再打印state
    for _ in range(3):
        sess.run(update)
        print sess.run(state)

#————————————Point_three————————————————#
# Fetch机制
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

intermed = tf.add(input1,input2)
#tf.mul(A,B) 为A，B向量相乘
mul = tf.mul(input3,intermed)

with tf.Session() as sess:
    result = sess.run([mul , intermed])#取回多个tensor
    print result

#————————————Point_four————————————————#
#feed机制：可以临时替代图中的任意操作中的tensor，可以对图中任何操作提交补丁，直接插入一个tensor
#tf.palceholder(type) 给定一个占位符，为feed提供标记作用
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.mul(input1,input2)

with tf.Session() as sess:
    print sess.run([output],feed_dict={input1:[7.0],input2:[3.0]})
    # print sess.run([input1 , input2]) #error no give feed!
