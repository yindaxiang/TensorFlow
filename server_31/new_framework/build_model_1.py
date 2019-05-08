import tensorflow as tf


class BuildModel(object):
    def __init__(self, input_shape):
        self.graph = tf.Graph()
        self.graph.as_default()
        self.input_shape = input_shape
        self.input_tensor = tf.placeholder(tf.float32, self.input_shape, name="input")
        self.last_output = self.input_tensor
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.keep_prob = None
        self.keep_prob_value = 1
        self.label = None
        self.loss = None
        self.acc = None
        self.train_op = None

    def dense(self, out_size, activation=None):
        weights = tf.Variable(tf.random_normal([int(self.last_output.shape[-1]), out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        wx_plus_b = tf.matmul(self.last_output, weights) + biases
        if activation is None:
            self.last_output = wx_plus_b
        else:
            if activation == "relu":
                self.last_output = tf.nn.relu(wx_plus_b)
            elif activation == "softmax":
                self.last_output = tf.nn.softmax(wx_plus_b)

    def conv(self, filters, kernel_size, padding="SAME"):
        if len(self.last_output.shape) != 4:
            print("The input dimensions do not meet the requirements")
        else:
            weights_shape = [kernel_size[0], kernel_size[1], int(self.last_output.shape[3]), filters]
            weights = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.1))
            biases = tf.Variable(tf.constant(0.1, shape=[filters]))
            self.last_output = tf.nn.conv2d(self.last_output, weights, strides=[1, 1, 1, 1], padding=padding) + biases

    def activation(self, activation):
        if activation == "relu":
            self.last_output = tf.nn.relu(self.last_output)

    def pool(self, pool_size=[2, 2], padding="SAME"):
        if len(self.last_output.shape) != 4:
            print("The input dimensions do not meet the requirements")
        else:
            ksize = [1, pool_size[0], pool_size[1], 1]
            self.last_output = tf.nn.max_pool(self.last_output, ksize=ksize, strides=ksize, padding=padding)

    def dropout(self, keep_prob_value):
        self.keep_prob = tf.placeholder(tf.float32)
        self.last_output = tf.nn.dropout(self.last_output, self.keep_prob)
        self.keep_prob_value = keep_prob_value

    def lstm(self):
        pass

    def reshape(self, shape):
        self.last_output = tf.reshape(self.last_output, shape)

    def losses(self, method="softmax"):
        self.label = tf.placeholder(tf.float32, list(self.last_output.shape), name="label")
        if method == "softmax":
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.last_output, labels=self.label, name="loss")
        elif method == "sparse_softmax":
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.last_output, self.label, name="loss")
        elif method == "sigmoid":
            self.loss = tf.nn.sigmoid_cross_entropy_with_logits(self.last_output, self.label, name="loss")
        elif method == "weighted":
            self.loss = tf.nn.weighted_cross_entropy_with_logits(self.last_output, self.label, name="loss")
        elif method == "yin":
            self.loss = tf.reduce_mean(-tf.reduce_sum(
                self.label*tf.log(self.last_output), reduction_indices=[1]), name="loss")

    def optimizer(self, method="adam", learn_rate=0.001):
        if method == "adam":
            self.train_op = tf.train.AdamOptimizer(learn_rate).minimize(
                self.loss, global_step=self.global_step, name="train_op")

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


def train_model(input_tensor, label, keep_prob, keep_prob_value, loss, train_op, global_step, train_step):
    import time
    from produce_data import get_train_batch
    start_time = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(train_step):
            batch_x, batch_y = get_train_batch(step)
            feed_dict = {input_tensor: batch_x, label: batch_y, keep_prob: keep_prob_value}
            print('loss:', loss)
            _, loss_value, global_step_value = sess.run([train_op, loss, global_step], feed_dict=feed_dict)
            print('loss_value:', loss_value)
            # print('global_step:', global_step)
            # print('step:', step)
            info = "Used_time_seconds is %.4f, train_step_num is %d, global_step is %d, loss is %.3f," \
                   "acc is None" % (time.time() - start_time, step, global_step_value, loss_value)
            print(info)


b = BuildModel([None, 784])
print(b.input_tensor)
input_tensor = b.input_tensor
b.reshape([-1, 28, 28, 1])
print(b.last_output)
b.conv(32, [5, 5], padding="SAME")
print(b.last_output)
b.pool([2, 2], padding="SAME")
print(b.last_output)
b.conv(64, [5, 5], padding="SAME")
print(b.last_output)
b.pool([2, 2], padding="SAME")
print(b.last_output)
b.reshape([-1, 7*7*64])
print(b.last_output)
b.dense(1024, activation="relu")
print(b.last_output)
b.dropout(0.5)
keep_prob = b.keep_prob
keep_prob_value = b.keep_prob_value
print(b.last_output)
b.dense(10, activation="softmax")
print(b.last_output)
b.losses(method="yin")
loss = b.loss
label = b.label
print(b.loss)
b.optimizer(method="adam", learn_rate=1e-4)
print(b.train_op)
train_op = b.train_op
global_step = b.global_step
# b.train_model(100)
# graph = b.graph
# input_tensor = graph.get_tensor_by_name()
train_model(input_tensor, label, keep_prob, keep_prob_value, loss, train_op, global_step, 100)
