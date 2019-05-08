import tensorflow as tf


class BuildModel(object):
    def __init__(self, input_shape):
        self.graph = tf.Graph()
        self.input_shape = input_shape
        self.keep_prob_value = 1
        self.lay_num = 0
        with self.graph.as_default():
            tf.placeholder(tf.float32, self.input_shape, name="lay_0")
            tf.Variable(0, trainable=False, name="global_step")

    def dense(self, out_size, activation=None):
        last_lay = self.graph.get_tensor_by_name("lay_%s:0" % self.lay_num)
        print('dense', last_lay)
        self.lay_num += 1
        with self.graph.as_default():
            weights = tf.Variable(tf.truncated_normal([int(last_lay.shape[-1]), out_size], stddev=0.1))
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            wx = tf.matmul(last_lay, weights)
            if activation is None:
                tf.add(wx, biases, name="lay_%s" % self.lay_num)
            else:
                if activation == "relu":
                    tf.nn.relu(tf.add(wx, biases), name="lay_%s" % self.lay_num)
                elif activation == "softmax":
                    tf.nn.softmax(tf.add(wx, biases), name="lay_%s" % self.lay_num)

    def conv(self, filters, kernel_size, padding="SAME"):
        last_lay = self.graph.get_tensor_by_name("lay_%s:0" % self.lay_num)
        print('conv', last_lay)
        self.lay_num += 1
        with self.graph.as_default():
            if len(last_lay.shape) != 4:
                print("The input dimensions do not meet the requirements")
            else:
                weights_shape = [kernel_size[0], kernel_size[1], int(last_lay.shape[3]), filters]
                weights = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.1))
                biases = tf.Variable(tf.constant(0.1, shape=[filters]))
                current_lay = tf.nn.conv2d(last_lay, weights, strides=[1, 1, 1, 1], padding=padding)+biases
                current_lay = tf.add(current_lay, biases, name="lay_%s" % self.lay_num)

    def activation(self, activation):
        last_lay = self.graph.get_tensor_by_name("lay_%s:0" % self.lay_num)
        print('activation', last_lay)
        self.lay_num += 1
        with self.graph.as_default():
            if activation == "relu":
                current_lay = tf.nn.relu(last_lay, name="lay_%s" % self.lay_num)

    def pool(self, pool_size, padding="SAME"):
        last_lay = self.graph.get_tensor_by_name("lay_%s:0" % self.lay_num)
        print('pool', last_lay)
        self.lay_num += 1
        with self.graph.as_default():
            if len(last_lay.shape) != 4:
                print("The input dimensions do not meet the requirements")
            else:
                ksize = [1, pool_size[0], pool_size[1], 1]
                current_lay = tf.nn.max_pool(last_lay, ksize=ksize, strides=ksize,
                                             padding=padding, name="lay_%s" % self.lay_num)

    def dropout(self, keep_prob_value):
        last_lay = self.graph.get_tensor_by_name("lay_%s:0" % self.lay_num)
        print('dropout', last_lay)
        self.lay_num += 1
        with self.graph.as_default():
            keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            current_lay = tf.nn.dropout(last_lay, keep_prob, name="lay_%s" % self.lay_num)
            self.keep_prob_value = keep_prob_value

    def lstm(self):
        pass

    def reshape(self, shape):
        last_lay = self.graph.get_tensor_by_name("lay_%s:0" % self.lay_num)
        print('reshape', last_lay)
        self.lay_num += 1
        with self.graph.as_default():
            current_lay = tf.reshape(last_lay, shape, name="lay_%s" % self.lay_num)

    def losses(self, method="softmax"):
        last_lay = self.graph.get_tensor_by_name("lay_%s:0" % self.lay_num)
        print(last_lay)
        self.lay_num += 1
        with self.graph.as_default():
            label = tf.placeholder(tf.float32, list(last_lay.shape), name="label")
            if method == "softmax":
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=last_lay, labels=label, name="loss")
            elif method == "sparse_softmax":
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(last_lay, label, name="loss")
            elif method == "sigmoid":
                loss = tf.nn.sigmoid_cross_entropy_with_logits(last_lay, label, name="loss")
            elif method == "weighted":
                loss = tf.nn.weighted_cross_entropy_with_logits(last_lay, label, name="loss")
            elif method == "yin":
                loss = tf.reduce_mean(-tf.reduce_sum(
                    label*tf.log(last_lay), reduction_indices=[1]), name="loss")

    def optimizer(self, method="adam", learn_rate=0.001):
        loss = self.graph.get_tensor_by_name("loss:0")
        global_step = self.graph.get_tensor_by_name("global_step:0")
        print(loss)
        self.lay_num += 1
        with self.graph.as_default():
            if method == "adam":
                train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss, global_step=global_step, name="train_op")

    def create_graph(self):
        return self.graph

    def train_model(self, train_step):
        import time
        from produce_data import get_train_batch
        start_time = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(train_step):
                batch_x, batch_y = get_train_batch(step)
                feed_dict = {self.input_tensor: batch_x, self.label: batch_y, self.keep_prob: self.keep_prob_value}
                # print('self.loss:', self.loss)
                _, global_step = sess.run([self.train_op, self.global_step], feed_dict=feed_dict)
                # print('loss_value:', loss_value)
                # print('global_step:', global_step)
                # print('step:', step)
                info = "Used_time_seconds is %.4f, train_step_num is %d, global_step is %d, loss is None," \
                       "acc is None" % (time.time() - start_time, step, global_step)
                print(info)


def train_model(graph, train_step):
    input_tensor = graph.get_tensor_by_name("lay_0:0")
    label = graph.get_tensor_by_name("label:0")
    # keep_prob = graph.get_tensor_by_name("keep_prob:0")
    # keep_prob_value = b.keep_prob_value
    loss = graph.get_tensor_by_name("loss:0")
    train_op = graph.get_operation_by_name("train_op")
    global_step = graph.get_tensor_by_name("global_step:0")
    import time
    from produce_data import get_train_batch
    start_time = time.time()
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(train_step):
            batch_x, batch_y = get_train_batch(step)
            feed_dict = {input_tensor: batch_x, label: batch_y}
            # print('loss:', loss)
            _, loss_value, global_step_value = sess.run([train_op, loss, global_step], feed_dict=feed_dict)
            print('loss_value:', loss_value)
            info = "Used_time_seconds is %.4f, train_step_num is %d, global_step is %d, loss is %.3f," \
                   "acc is None" % (time.time() - start_time, step, global_step_value, loss_value)
            print(info)


b = BuildModel([None, 784])
b.reshape([-1, 28, 28, 1])
# lay_num = b.lay_num
# last_lay = b.graph.get_tensor_by_name("lay_%s:0" % lay_num)
# print(last_lay)
b.conv(32, [5, 5], padding="SAME")
b.pool([2, 2], padding="SAME")
b.conv(64, [5, 5], padding="SAME")
b.pool([2, 2], padding="SAME")
b.reshape([-1, 7*7*64])
b.dense(1024, activation="relu")
# b.dropout(0.5)
b.dense(10, activation="softmax")
b.losses(method="yin")
b.optimizer(method="adam", learn_rate=1e-4)
g = b.create_graph()
train_model(g, 100)
tf.nn.rnn_cell.LSTMCell
