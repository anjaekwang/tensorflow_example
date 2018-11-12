import tensorflow as tf
import numpy as np
import os
import time
from utils import ckpt_load,load_data,to_idx,cal_accuracy

learning_rate = 0.001
n_class = 20
m1_hidden_size = 72
m2_hidden_size = 9
batch_size = 100
training_epoch = 200
keep_prob = 1  # train 0.7 test 1

m2_x, m1_x,label_data = load_data(1)
n_data = m1_x.shape[0]

m1_X = tf.placeholder(tf.float32, shape=[None,30,72])
m2_X = tf.placeholder(tf.float32, shape=[None,30,9])
Y = tf.placeholder(tf.float32, shape=[None,n_class])

# model1
with tf.variable_scope("model1_LSTM") :
    m1_cell = tf.contrib.rnn.BasicLSTMCell(num_units=m1_hidden_size, state_is_tuple=True)
    m1_initial_state = m1_cell.zero_state(batch_size, tf.float32)
    m1_outputs, _m1_states = tf.nn.dynamic_rnn(m1_cell, m1_X, initial_state=m1_initial_state, dtype=tf.float32)
    m1_outputs = tf.nn.relu(m1_outputs[:,-1]) # 마지막 결과만 사용 shape = (batch_size,hidden_size =72)
    m1_outputs = tf.nn.dropout(m1_outputs, keep_prob=keep_prob)

    model1_LSTM_hist = tf.summary.histogram("m1_outputs", m1_outputs)


with tf.name_scope("layer1") as scope:
    w1 = tf.get_variable("w1", shape=[72, n_class], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', shape=[n_class], initializer=tf.contrib.layers.xavier_initializer())
    m1_hypothesis = tf.nn.relu(tf.matmul(m1_outputs, w1) + b1)
    m1_hypothesis = tf.nn.dropout(m1_hypothesis, keep_prob=keep_prob)

    w1_hist = tf.summary.histogram("weights1", w1)
    b1_hist = tf.summary.histogram("biases1", b1)
    m1_hypothesis_hist = tf.summary.histogram("m1_hypothesis", m1_hypothesis)

# model2
with tf.variable_scope("model2_LSTM"):
    m2_cell = tf.contrib.rnn.BasicLSTMCell(num_units=m2_hidden_size, state_is_tuple=True)
    m2_initial_state = m2_cell.zero_state(batch_size, tf.float32)
    m2_outputs, _m2_states = tf.nn.dynamic_rnn(m2_cell, m2_X, initial_state=m2_initial_state, dtype=tf.float32)
    m2_outputs = tf.nn.relu(m2_outputs[:,-1]) # 마지막 결과만 사용 shape = (batch_size,hidden_size =9)
    m2_outputs = tf.nn.dropout(m2_outputs, keep_prob=keep_prob)

    model2_LSTM_hist = tf.summary.histogram("m2_outputs", m2_outputs)


with tf.name_scope("layer1_2") as scope:
    w2 = tf.get_variable("w2", shape=[9, n_class], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2', shape=[n_class], initializer=tf.contrib.layers.xavier_initializer())
    m2_hypothesis = tf.nn.relu(tf.matmul(m2_outputs, w2) + b2)
    m2_hypothesis = tf.nn.dropout(m2_hypothesis, keep_prob=keep_prob)

    w2_hist = tf.summary.histogram("weights2", w2)
    b2_hist = tf.summary.histogram("biases2", b2)
    m2_hypothesis_hist = tf.summary.histogram("m2_hypothesis", m2_hypothesis)


# Fully connected
with tf.name_scope("fc") as scope:
    w3 = tf.get_variable('w3', shape=[n_class, n_class],initializer=tf.contrib.layers.xavier_initializer())
    w4 = tf.get_variable('w4', shape=[n_class, n_class],initializer=tf.contrib.layers.xavier_initializer())
    hypothesis = (tf.matmul(m1_hypothesis, w3) + tf.matmul(m2_hypothesis, w4)) / 2

    w3_hist = tf.summary.histogram("weights3", w3)
    w4_hist = tf.summary.histogram("weights4", w4)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

# train
with tf.name_scope("loss") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
    cost_summ = tf.summary.scalar("cost", cost)

with tf.name_scope("train") as scope:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope("accuracy") as scope:
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc_summ = tf.summary.scalar("accuracy", accuracy)


with tf.Session() as sess:
    checkpoint_dir = './checkpoint'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    if ckpt_load(sess, checkpoint_dir, saver):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    config = input("[?] Test or Train? ")
    if(config.__eq__('Train')):
        print("Training...")
        checkpoint_dir = os.path.join(checkpoint_dir, "simple_cnn")
        start_time = time.time()
        merged_summary = tf.summary.merge_all()
        summ_dir = './board/sample'
        s_count = 0
        while os.path.exists(summ_dir):
            s_count += 1
            summ_dir = summ_dir + '_%d'%s_count

        writer = tf.summary.FileWriter(summ_dir)
        writer.add_graph(sess.graph)  # Show the graph

        stride = 1

        _batch_size = batch_size * 30

        for epoch in range(training_epoch):
            n_batch = n_data//_batch_size
            for j in range(n_batch): # 배치만큼 빼는 구간.
                m1_data = np.array([])
                m2_data = np.array([])
                target = np.array([])

                if(j*_batch_size+_batch_size > n_data):
                    break
                m1_batch_x = m1_x[j*_batch_size:j*_batch_size+_batch_size]
                m2_batch_x = m2_x[j*_batch_size:j*_batch_size+_batch_size]
                batch_y = label_data[j*_batch_size:j*_batch_size+_batch_size]
                count = 0  # 900개씩 빼고 그 범주내에서 count를..

                for i in range(_batch_size):
                    left_idx = i - 15
                    right_idx = i + 15
                    if left_idx >= 0 and right_idx < _batch_size:
                        count += 1
                        m1_data = np.append(m1_data, m1_batch_x[left_idx:right_idx])
                        m2_data = np.append(m2_data, m2_batch_x[left_idx:right_idx])
                        target = np.append(target, np.eye(n_class)[to_idx(batch_y[right_idx - 1])])

                        if(count == batch_size):  # feed_dict 구간. batch_size 만큼 채워졌을때.
                            m1_data = m1_data.reshape(batch_size, 30, 72)
                            m2_data = m2_data.reshape(batch_size, 30, 9)
                            target = target.reshape(batch_size, 20)
                            summary, c, _ = sess.run([merged_summary, cost, optimizer],feed_dict={m1_X: m1_data, m2_X: m2_data, Y: target})

                            m1_data = np.array([])
                            m2_data = np.array([])
                            target = np.array([])
                            count = 0

            writer.add_summary(summary, global_step=epoch)
            print("Epoch: [%2d], time: [%4.4f], loss: [%.8f]" % (epoch + 1, time.time() - start_time, c))
        saver.save(sess, checkpoint_dir)
        print('Learning Finished..!')

    else:
        m2_test_x, m1_test_x, t_target = load_data(2)

        print('Test Acc : [%.8f]' % cal_accuracy(m1_test_x, m2_test_x, t_target, sess, accuracy,correct_prediction, hypothesis, m1_X, m2_X, Y))
        print('Train Acc : [%.8f]' % cal_accuracy(m1_x, m2_x, label_data, sess, accuracy, correct_prediction, hypothesis, m1_X, m2_X, Y)) #에러분석시 주석처리
       # os.system('shutdown -s -f -t 0')