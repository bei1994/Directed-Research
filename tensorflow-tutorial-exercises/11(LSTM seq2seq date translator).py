import random
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import babel
import tensorflow as tf
import tensorflow.contrib.legacy_seq2seq as seq2seq
from babel.dates import format_date
from faker import Faker
from sklearn.model_selection import train_test_split


def create_date():
    """
        Creates some fake dates
        :returns: tuple containing
                  1. human formatted string
                  2. machine formatted string
                  3. date object.
    """
    dt = fake.date_object()

    # wrapping this in a try catch because
    # the locale 'vo' and format 'full' will fail
    try:
        human = format_date(dt,
                            format=random.choice(FORMATS),
                            locale=random.choice(LOCALES))

        case_change = random.randint(0, 3)  # 1/2 chance of case change
        if case_change == 1:
            human = human.upper()
        elif case_change == 2:
            human = human.lower()

        machine = dt.isoformat()
    except AttributeError as e:
        return None, None, None

    return human, machine


def batch_data(x, y, batch_size):
    shuffle = np.random.permutation(len(x))
    start = 0
#     from IPython.core.debugger import Tracer; Tracer()()
    x = x[shuffle]
    y = y[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start + batch_size], y[start:start + batch_size]
        start += batch_size


fake = Faker()
fake.seed(42)
random.seed(42)

FORMATS = ['short',
           'medium',
           'long',
           'full',
           'd MMM YYY',
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY',
           ]

# change this if you want it to work with only a single language
LOCALES = babel.localedata.locale_identifiers()
LOCALES = [lang for lang in LOCALES if 'en' in str(lang)]

# create date dataset
data = [create_date() for _ in range(50000)]

# preprocess data
x = [x for x, y in data]
y = [y for x, y in data]

u_characters = set(' '.join(x))
char2numX = dict(zip(u_characters, range(len(u_characters))))
u_characters = set(' '.join(y))
char2numY = dict(zip(u_characters, range(len(u_characters))))

char2numX['<PAD>'] = len(char2numX)
num2charX = dict(zip(char2numX.values(), char2numX.keys()))
max_len = max([len(date) for date in x])
x = [[char2numX['<PAD>']] * (max_len - len(date)) + [char2numX[x_] for x_ in date]
     for date in x]
x = np.array(x)

char2numY['<GO>'] = len(char2numY)
num2charY = dict(zip(char2numY.values(), char2numY.keys()))
y = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in y]
y = np.array(y)

x_seq_length = len(x[0])
y_seq_length = len(y[0]) - 1

epochs = 21
batch_size = 128
num_units = 32
embed_size = 10
bidirectional = True

tf.reset_default_graph()
sess = tf.InteractiveSession()

# Tensor where we will feed the data into graph
inputs = tf.placeholder(tf.int32, (None, x_seq_length), 'inputs')  # [batch_size, x_seq_length]
outputs = tf.placeholder(tf.int32, (None, None), 'output')  # [batch_size, y_seq_length]
targets = tf.placeholder(tf.int32, (None, None), 'targets')  # [batch_size, y_seq_length]

# Embedding layers
input_embedding = tf.Variable(tf.random_uniform(
    (len(char2numX), embed_size), -1.0, 1.0),
    name='enc_embedding')
output_embedding = tf.Variable(tf.random_uniform(
    (len(char2numY), embed_size), -1.0, 1.0),
    name='dec_embedding')
# [batch_size, x_seq_length, embed_size]
date_input_embed = tf.nn.embedding_lookup(input_embedding, inputs)
# [batch_size, y_seq_length, embed_size]
date_output_embed = tf.nn.embedding_lookup(output_embedding, outputs)


with tf.variable_scope("encoding") as encoding_scope:
    if not bidirectional:
        # Regular approach with LSTM units
        lstm_enc = tf.contrib.rnn.LSTMCell(num_units)
        _, last_state = tf.nn.dynamic_rnn(lstm_enc,
                                          inputs=date_input_embed,
                                          dtype=tf.float32)
    else:
        # Using a bidirectional LSTM architecture instead
        enc_fw_cell = tf.contrib.rnn.LSTMCell(num_units)
        enc_bw_cell = tf.contrib.rnn.LSTMCell(num_units)
        ((enc_fw_out, enc_bw_out), (enc_fw_final, enc_bw_final)) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=enc_fw_cell,
            cell_bw=enc_bw_cell,
            inputs=date_input_embed,
            dtype=tf.float32)

        enc_fin_c = tf.concat((enc_fw_final.c, enc_bw_final.c), 1)
        enc_fin_h = tf.concat((enc_fw_final.h, enc_bw_final.h), 1)
        last_state = tf.contrib.rnn.LSTMStateTuple(c=enc_fin_c, h=enc_fin_h)

with tf.variable_scope("decoding") as decoding_scope:
    if not bidirectional:
        lstm_dec = tf.contrib.rnn.LSTMCell(num_units)
    else:
        lstm_dec = tf.contrib.rnn.LSTMCell(2 * num_units)

    # dec_outputs: [batch_size, y_seq_length, num_units]
    dec_outputs, _ = tf.nn.dynamic_rnn(lstm_dec,
                                       inputs=date_output_embed,
                                       initial_state=last_state)

# logits: [batch_size, y_seq_length, units]
logits = tf.layers.dense(dec_outputs,
                         units=len(char2numY),
                         use_bias=True)

# connect outputs to
with tf.name_scope("optimization"):
    # Loss function
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, y_seq_length]))
    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

# generate daata
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

sess.run(tf.global_variables_initializer())
for epoch_i in range(epochs):
    start_time = time.time()
    for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
        _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
                                               feed_dict={inputs: source_batch,
                                                          outputs: target_batch[:, :-1],
                                                          targets: target_batch[:, 1:]})

    accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:, 1:])
    print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(
        epoch_i, batch_loss, accuracy, time.time() - start_time))


# translate on test set
source_batch, target_batch = next(batch_data(X_test, y_test, batch_size))
dec_input = np.zeros((len(source_batch), 1)) + char2numY['<GO>']

for i in range(y_seq_length):
    batch_logits = sess.run(logits,
                            feed_dict={inputs: source_batch,
                                       outputs: dec_input})
    prediction = batch_logits[:, -1].argmax(axis=-1)
    dec_input = np.hstack([dec_input, prediction[:, None]])

print('Accuracy on test set is: {:>6.3f}'.format(np.mean(dec_input == target_batch)))

# randomly choose 2 items to see what it will spit out
num_preds = 2
source_chars = [[num2charX[l] for l in sent if num2charX[l] != "<PAD>"] for sent in source_batch[:num_preds]]
dest_chars = [[num2charY[l] for l in sent] for sent in dec_input[:num_preds, 1:]]

for date_in, date_out in zip(source_chars, dest_chars):
    print(''.join(date_in) + ' => ' + ''.join(date_out))
