import re
import json
import jieba
import pickle
import numpy as np
import pandas as pd
import random

# 产生随机的embedding矩阵
def random_embedding(id2word, embedding_dim):
    """

    :param id2word:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(id2word), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def get_batch(data,batch_size, shuffle=False):
    n_sample = int(batch_size / 2)
    class_0_data = []
    class_1_data = []
    for i in range(len(data)):
        if data[i][2] == [1,0]:
            class_0_data.append(data[i])
        else:
            class_1_data.append(data[i])

    for i in range(len(data)//batch_size):
        class_0_Item = len(class_0_data)
        class_samp_0_ids = [i for i in sorted(random.sample([i for i in range(class_0_Item)], n_sample))]
        class_data_0 = np.array(class_0_data)[class_samp_0_ids]

        class_1_Item = len(class_1_data)
        class_samp_1_ids = [i for i in sorted(random.sample([i for i in range(class_1_Item)], n_sample))]
        class_data_1 = np.array(class_1_data)[class_samp_1_ids]
        data = np.concatenate([class_data_0,class_data_1],axis=0)
        if shuffle:
            np.random.shuffle(data)
        s1_data, s2_data, label_data = [], [], []
        for (s1_set, s2_set, y_set) in data:
            s1_data.append(s1_set)
            s2_data.append(s2_set)
            label_data.append(y_set)

        yield np.array(s1_data), np.array(s2_data), np.array(label_data)

# 将数据pad，生成batch数据返回，这里没有取余数。
# a = np.random.random([100,2])
def get_batch_(data, batch_size, shuffle=False):
    """
    :param data:
    :param batch_size:
    :param shuffle:
    :return:
    """
    # 乱序没有加
    if shuffle:
        np.random.shuffle(data)

    for i in range(len(data) // batch_size):
        data_size = data[i * batch_size: (i + 1) * batch_size]
        s1_data, s2_data, label_data = [], [], []
        for (s1_set, s2_set, y_set) in data_size:
            s1_data.append(s1_set)
            s2_data.append(s2_set)
            label_data.append(y_set)
        yield np.array(s1_data), np.array(s2_data), np.array(label_data)
    if len(data)%batch_size != 0:
        s1_data, s2_data, label_data = [], [], []
        data_size = data[-batch_size:]
        for (s1_set, s2_set, y_set) in data_size:
            s1_data.append(s1_set)
            s2_data.append(s2_set)
            label_data.append(y_set)
        yield np.array(s1_data), np.array(s2_data), np.array(label_data)

