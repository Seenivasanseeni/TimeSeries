import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
import sys

start_time = time.time()
restoreFlag=int(sys.argv[1])
predictFlag=int(sys.argv[2])

def elapsed(sec):
    if (sec < 60):
        return str(sec) + " sec"
    elif sec < (60 * 60):
        return str(sec // 60) + " min"
    else:
        return str(sec // (60 * 60)) + " hr"


logs_path = 'TEMP/tensorflow/rnn_words'
writer = tf.summary.FileWriter(logs_path)

training_file = 'Story.txt'


def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [line.strip() for line in content]
    c = []
    for line in content:
        c.extend(line.split())
    content = c
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content


training_data = read_data(training_file)
print(training_data.shape)
print("Loaded training data")


def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary


dictionary, reverse_dictionary = build_dataset(training_data)

vocab_size = len(dictionary)

learning_rate = 0.001
training_iters = 50000
display_step = 1000
n_input = 3

n_hidden = 512

x = tf.placeholder("float", [None, n_input, 1],name="x")
y = tf.placeholder("float", [None, vocab_size],name="y")

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}

biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}


def RNN(x, weights, biases):
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_input, 1)

    rnn_cell = rnn.MultiRNNCell(
        [rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)]
    )

    output, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(output[-1], weights["out"]) + biases['out']


pred = RNN(x, weights, biases)

cost = tf.reduce_sum(
    tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
)

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
correct_pred = tf.equal(
    tf.arg_max(pred, 1), tf.argmax(y, 1)
)

accuracy = tf.reduce_mean(
    tf.cast(correct_pred, tf.float32)
)

init = tf.global_variables_initializer()

saver=tf.train.Saver()

with tf.Session() as sess:
    if(restoreFlag):
        saver.restore(sess,'tmp/model.ckpt')
        print("Model Restored")
    sess.run(init)
    step = 0
    offset = random.randint(0, n_input + 1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0
    writer.add_graph(sess.graph)

    while (step < training_iters and predictFlag==0):
        offset+=n_input+1
        if (offset > len(training_data) - end_offset):  # for what
            offset = random.randint(0, n_input + 1)
        step+=1
        symbols_in_keys = [[dictionary[str(training_data[i])]] for i in range(offset, offset + n_input)]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[str(training_data[offset + n_input])]] = 1.0

        _, acc, loss, onehot_pred = sess.run(
            [optimizer, accuracy, cost, pred], feed_dict={
                x: symbols_in_keys, y: [symbols_out_onehot]
            }
        )

        loss_total += loss
        acc_total += acc
        if (step % display_step == 0):
            saver.save(sess, "tmp/model.ckpt")
            print("Iteration {} ACC {} LOSS {}".format(step, acc_total / display_step, loss_total / display_step))
            acc_total = 0
            loss_total = 0

            symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
            symbols_out = training_data[offset + n_input]
            symobls_out_pred = reverse_dictionary[int(
                tf.argmax(onehot_pred, 1).eval()
            )]
            print("{} = {} vs {} ".format(
                symbols_in, symbols_out, symobls_out_pred
            ))
    print("Optimization finished")
    saver.save(sess,"tmp/model.ckpt")
    print("Model Saved")
    print("Time : ", elapsed(time.time() - start_time))
    print("Prediction")
    while predictFlag==1:
        sentence = input("%s words:".format(n_input))
        sentence = sentence.strip()
        words = sentence.split()
        if (len(words) != n_input):
            continue
        try:
            symbols_in_keys = [[dictionary[word]] for word in words]
            print(symbols_in_keys)
            import pdb
          #  pdb.set_trace()

            for i in range(32):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_pred = sess.run(pred, feed_dict={
                    x: [symbols_in_keys]
                })
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence, reverse_dictionary[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append([onehot_pred_index])
            print("FINAL sentence", sentence)
        except:
            print("Word not in dictionary")
