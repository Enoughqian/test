import numpy as np
import tensorflow as tf
from data_utils import data_utils
from param import params
import pickle

data_pk = '../processed_data/word_data.pk'
char_data_pk = '../processed_data/char_data.pk'
# with open(data_pk, 'rb') as f:
#     all_data, word2id, id2word = pickle.load(f)
with open(char_data_pk, 'rb') as f1:
    train_data, test_data, char2id, id2char = pickle.load(f1)

# np.random.shuffle(all_data)
# ratio = int(len(all_data)*0.8)
# train_set = all_data[:ratio]
# test_set = all_data[ratio:]

embeddings = data_utils.random_embedding(id2char, params.embedding_dim)

n = 15      #文本序列长度
d = 100      #词向量维度
u = 300      #隐藏神经元个数
d_a = 40    #设置维度
r = 20       #设置维度

graph_mhd = tf.Graph()
with graph_mhd.as_default():
    """
    构建神经网络的结构、损失、优化方法和评估方法
    """
    # shape[batch_size, sentences]
    left_input = tf.placeholder(tf.int32, shape=[None, params.max_sentence_length], name="left_input")
    # shape[batch_size, sentences]
    right_input = tf.placeholder(tf.int32, shape=[None, params.max_sentence_length], name="right_input")

    # shape[batch_size, sentences, labels]
    labels = tf.placeholder(tf.float32, shape=[None, 2], name="labels")

    # dropout keep_prob
    dropout_pl = tf.placeholder(dtype=tf.float32, shape=(), name="dropout")

    with tf.variable_scope("embeddings"):  # 命名空间
        _word_embeddings = tf.Variable(embeddings,  # shape[len_words,300]
                                       dtype=tf.float32,
                                       trainable=True,  # 嵌入层是否可以训练
                                       name="embedding_matrix")
        left_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=left_input, name="left_embeddings")
        right_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=right_input, name="right_embeddings")

        left_embeddings = tf.nn.dropout(left_embeddings, dropout_pl)
        right_embeddings = tf.nn.dropout(right_embeddings, dropout_pl)

    with tf.variable_scope("cell_by_one_layer_bi-lstm"):
        # 词1层bi-lstm
        cell_fw = tf.nn.rnn_cell.LSTMCell(u)
        cell_bw = tf.nn.rnn_cell.LSTMCell(u)
        (left_output_fw_seq, left_output_bw_seq), left_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                                left_embeddings,
                                                                                                dtype=tf.float32)
        left_bi_result = tf.concat([left_output_fw_seq, left_output_bw_seq], axis=-1)
        left_bi_feature = tf.concat([left_states[0].h,left_states[1].h],axis=1)
    with tf.variable_scope("self-attention"):
        W_s1 = tf.get_variable("w_s1",shape=[d_a, 2*u],initializer=tf.truncated_normal_initializer())
        W_s2 = tf.get_variable("w_s2",shape=[r, d_a],initializer=tf.truncated_normal_initializer())

    with tf.variable_scope("self-attention",reuse=True):
        left_W_s1 = tf.get_variable('w_s1', shape=[d_a, 2 * u])
        left_W_s2 = tf.get_variable('w_s2', shape=[r, d_a])

        left_A = tf.nn.softmax(tf.map_fn(lambda x: tf.matmul(left_W_s2, x),
                                        tf.tanh(tf.map_fn(lambda x: tf.matmul(left_W_s1, tf.transpose(x)), left_bi_result))))
        left_M = tf.matmul(left_A, left_bi_result)
        left_feature = tf.squeeze(tf.reduce_mean(left_M,axis=1))

    with tf.variable_scope("right_one_layer_bi-lstm"):
        (right_output_fw_seq, right_output_bw_seq), right_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                                   right_embeddings,
                                                                                                   dtype=tf.float32)

        right_bi_result = tf.concat([right_output_fw_seq, right_output_bw_seq], axis=-1)
        right_bi_feature = tf.concat([right_states[0].h, right_states[1].h], axis=1)
    with tf.variable_scope("self-attention",reuse=True):
        right_W_s1 = tf.get_variable('w_s1', shape=[d_a, 2*u])
        right_W_s2 = tf.get_variable('w_s2', shape=[r, d_a])

        right_A = tf.nn.softmax(tf.map_fn(lambda x: tf.matmul(right_W_s2, x),
                                          tf.tanh(tf.map_fn(lambda x: tf.matmul(left_W_s1, tf.transpose(x)), right_bi_result))))

        right_M = tf.matmul(right_A, right_bi_result)
        right_feature = tf.squeeze(tf.reduce_mean(right_M,axis=1))

    with tf.variable_scope("Similarity_calculation_layer"):

        def manhattan_dist(input1, input2):

            score = tf.exp(-tf.reduce_sum(tf.abs(input1 - input2), 1))

            # score = tf.exp(-tf.reduce_sum(tf.square(input1 - input2), 1)/2.)
            return score


        left_vector = tf.reshape(left_feature, [tf.cast(tf.shape(labels)[0], tf.int32), 2*u])
        right_vector = tf.reshape(right_feature, [tf.cast(tf.shape(labels)[0], tf.int32), 2*u])

        left_vector = tf.concat([left_vector,left_bi_feature],axis=1)
        right_vector = tf.concat([right_vector,right_bi_feature],axis=1)

        # 曼哈顿距离
        output = tf.expand_dims(manhattan_dist(left_vector, right_vector), -1)

    with tf.variable_scope("classification"):
        # logits:shape[batch_size,num_tags]
        logits = tf.layers.dense(inputs=output, units=2)

    # 计算损失
    with tf.variable_scope("loss"):
        logits__ = tf.nn.softmax(logits)
        loss = (-0.25 * tf.reduce_sum(labels[:, 0] * tf.log(logits__[:, 0]))
                - 0.75 * tf.reduce_sum(labels[:, 1] * tf.log(logits__[:, 1]))
                ) / tf.cast(tf.shape(labels)[0],tf.float32)
        # losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
        # loss = tf.reduce_mean(losses)

    #选择优化器
    with tf.variable_scope("train_step"):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        global_add = global_step.assign_add(1)#用于计数
        train_op = tf.train.AdamOptimizer(params.lr).minimize(loss)

    # 准确率/f1/p/r计算
    with tf.variable_scope("evaluation"):
        true = tf.cast(tf.argmax(labels, axis=-1), tf.float32)  # 真实序列的值
        labels_softmax = tf.nn.softmax(logits)
        labels_softmax_ = tf.argmax(labels_softmax, axis=-1)
        pred = tf.cast(labels_softmax_, tf.float32)  # 预测序列的值
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32))

        epsilon = 1e-7
        cm = tf.contrib.metrics.confusion_matrix(true, pred, num_classes=2)
        precision = cm[1][1] / tf.reduce_sum(tf.transpose(cm)[1])
        recall = cm[1][1] / tf.reduce_sum(cm[1])
        fbeta_score = (2 * precision * recall / (precision + recall + epsilon))



with tf.Session(graph=graph_mhd) as sess:
    if params.isTrain:
        saver = tf.train.Saver(tf.global_variables())
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(params.epoch_num):
            for s1, s2, y in data_utils.get_batch(train_data, params.batch_size, shuffle=True):
                _, l, acc, p, r, f,global_nums = sess.run(
                    [train_op, loss,accuracy,precision,recall,fbeta_score,global_add], {
                    left_input: s1,
                    right_input: s2,
                    labels : y,
                    dropout_pl: 1.
                })
                if global_nums % 50 == 0:
                    saver.save(sess, '../model_save/model.ckpt', global_step=global_nums)
                    print(
                        'train: epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4} , precision {}, recall: {:.4}, fbeta_score: {:.4} '.format(epoch , global_nums,
                                                                                          l, acc,p, r, f))


                if global_nums % 200 == 0:
                    print('-----------------valudation---------------')
                    s1, s2, y = next(data_utils.get_batch(test_data, np.shape(test_data)[0],  shuffle=True))
                    l, acc, p, r, f, global_nums = sess.run(
                        [ loss, accuracy, precision, recall, fbeta_score, global_add], {
                            left_input: s1,
                            right_input: s2,
                            labels: y,
                            dropout_pl: 1.
                        })
                    print(
                        'valudation: epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4} , precision {}, recall: {:.4}, fbeta_score: {:.4} '.format(
                            epoch, global_nums,
                            l, acc, p, r, f))
                    print('-----------------train---------------')
        s1, s2, y = next(data_utils.get_batch(test_data, np.shape(test_data)[0], shuffle=True))
        l, acc, p, r, f, global_nums = sess.run(
                    [loss, accuracy, precision, recall, fbeta_score, global_add], {
                        left_input: s1,
                        right_input: s2,
                        labels: y,
                        dropout_pl: 1.
                    })
        print('test: loss: {:.4}, accuracy: {:.4} , precision {}, recall: {:.4}, fbeta_score: {:.4} '.format(
                        l, acc, p, r, f))

