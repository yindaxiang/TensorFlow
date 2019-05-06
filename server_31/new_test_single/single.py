type = "tensorflow"
enter = "single"
learn_rate = 0.001
train_step = 100

import tensorflow as tf
import pickle
import numpy as np
import pandas as pd


NUM_LABLES = 10  # 分类结果为10类
FC_SIZE =100   # 全连接隐藏层节点个数
BATCH_SIZE = 128  # 每次训练batch数
checkpoint_path = r"./model"



train_data = {b'data': [], b'labels': []}
for i in range(5):
    with open(r"./data/cifar-10-batches-py/data_batch_" + str(i + 1), mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
        train_data[b'data'] += list(data[b'data'])
        train_data[b'labels'] += data[b'labels']
x_train = np.array(train_data[b'data']) / 255
y_train = np.array(pd.get_dummies(train_data[b'labels']))


def test_data():
    test_data = {b'data': [], b'labels': []}
    with open(r"./data/cifar-10-batches-py/test_batch", mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
        test_data[b'data']+=list(data[b'data'])
        test_data[b'labels']+=data[b'labels']
    x_test = np.array(test_data[b'data']) / 255
    y_test = np.array(pd.get_dummies(test_data[b'labels']))
    return x_test,y_test


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],strides = [1, 2, 2, 1], padding = 'SAME')


def build_model():
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(tf.float32, [None, 3072],name="input")
        y_ = tf.placeholder(tf.float32, [None, NUM_LABLES],name="label")
        x_image = tf.reshape(x, [-1, 3, 32, 32])
        x_image = tf.transpose(x_image, [0, 2, 3, 1])  # [-1, 32, 32, 3]

        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 卷积 [-1, 32, 32, 32]
        h_pool1 = max_pool_2x2(h_conv1)  # 池化 [-1, 16, 16, 32]

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # [-1, 16, 16, 64]
        h_pool2 = max_pool_2x2(h_conv2)  # [-1, 8, 8, 64]

        h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])

        W_fc1 = weight_variable([8 * 8 * 64, FC_SIZE])
        b_fc1 = bias_variable([FC_SIZE])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        keep_prob = tf.placeholder(tf.float32,name="keep_prob")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([FC_SIZE, NUM_LABLES])
        b_fc2 = bias_variable([NUM_LABLES])
        y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]),name="loss")
        global_step = tf.Variable(0, trainable=False, name="global_step")
        train_op = tf.train.AdamOptimizer(0.001).minimize(cross_entropy, global_step=global_step,name="train_op")

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")
    return g


def train_model(graph):
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    x = graph.get_tensor_by_name("input:0")
    y_ = graph.get_tensor_by_name("label:0")
    global_step = graph.get_tensor_by_name("global_step:0")
    train_op = graph.get_tensor_by_name("train_op:0")
    cross_entropy = graph.get_tensor_by_name("loss:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    print(x_train[0: BATCH_SIZE].shape)
    print(y_train[0: BATCH_SIZE].shape)
    with tf.Session(graph=graph) as sess:
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=5)
        for i in range(10):
            start = i * BATCH_SIZE % 50000
            sess.run(train_op, feed_dict={x: x_train[start: start + BATCH_SIZE],
                                          y_: y_train[start: start + BATCH_SIZE], keep_prob: 0.5})
            train_accuracy,train_loss,g_strp= sess.run([accuracy,cross_entropy,global_step],feed_dict={x: x_train[start: start + BATCH_SIZE],
                                        y_: y_train[start: start + BATCH_SIZE], keep_prob: 0.5})
            current_info = "step %d, train_accuracy %.4f train_loss %.4f global_step %d" % (i, train_accuracy, train_loss,g_strp)
            print(current_info)
        saver.save(sess, checkpoint_path + "/model.ckpt", global_step=global_step)


if __name__ == "__main__":
    graph = build_model()
    train_model(graph)
