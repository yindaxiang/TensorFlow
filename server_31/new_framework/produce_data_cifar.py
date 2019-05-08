import pickle
import numpy as np
import pandas as pd

BATCH_SIZE = 128
train_data = {b'data': [], b'labels': []}
for i in range(5):
    with open(r"./data/cifar-10-batches-py/data_batch_" + str(i + 1), mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
        train_data[b'data'] += list(data[b'data'])
        train_data[b'labels'] += data[b'labels']
x_train = np.array(train_data[b'data']) / 255
y_train = np.array(pd.get_dummies(train_data[b'labels']))


def get_train_batch(step=0):
    start = step * BATCH_SIZE % 50000
    train_x = x_train[start: start + BATCH_SIZE]
    train_y = y_train[start: start + BATCH_SIZE]
    return train_x, train_y
