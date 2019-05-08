from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

data_dir = r"./data/mnist"

mnist = input_data.read_data_sets(data_dir, one_hot=True)


def get_train_batch(step=0):
    batch_x, batch_y = mnist.train.next_batch(128)
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    return batch_x, batch_y
