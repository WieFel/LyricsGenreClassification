import numpy as np
import tensorflow as tf
import pickle
from Preprocessing import FINAL_OUTPUT


# returns data as pandas dataframe
def read_dataset(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


print("Reading dataset...")
data = np.load(FINAL_OUTPUT)
genres = data[:, 0]  # take first column: genres
lyrics = data[:, 1]  # take second column: lyrics

len_data = len(data)

max_len = 0
for l in lyrics:
    if len(l) > max_len:
        max_len = len(l)

NUM_STEPS = 1000
DISPLAY_STEP = 100
BATCH_SIZE = 128
EMBEDDING_DIMENSION = 64
NUM_CLASSES = len(set(genres))
TRAINING_PARTITION_SIZE = 0.7  # percent

# network parameters
HIDDEN_LAYER_SIZE = 32
NUM_LSTM_LAYERS = 2

LEARNING_RATE = 0.001
DECAY = 0.9

seqlens = []

for lyric_id in range(len_data):
    seqlens.append(len(lyrics[lyric_id]))

    # if lyrics text is not as long as max_len -> pad it to be equally long
    if len(lyrics[lyric_id]) < max_len:
        pads = ["PAD"] * (max_len - len(lyrics[lyric_id]))
        lyrics[lyric_id] = lyrics[lyric_id] + pads

# seqlens *= 2
unique_genres = list(set(genres))
genre_dict = {}
for g in unique_genres:
    genre_dict[g] = unique_genres.index(g)

for i in range(len(genres)):
    label = genre_dict[genres[i]]  # get genre index
    one_hot_encoding = [0] * NUM_CLASSES
    one_hot_encoding[label] = 1
    genres[i] = one_hot_encoding

word2index_map = {}
index = 0
for sent in lyrics:
    for word in sent:
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1

index2word_map = {index: word for word, index in word2index_map.items()}

vocabulary_size = len(index2word_map)

data_indices = list(range(len(data)))
np.random.shuffle(data_indices)
data = np.array(lyrics)[data_indices]
labels = np.array(genres)[data_indices]
seqlens = np.array(seqlens)[data_indices]
train_x = lyrics[:(int(len(data_indices) * TRAINING_PARTITION_SIZE))]
train_y = labels[:(int(len(data_indices) * TRAINING_PARTITION_SIZE))]
train_seqlens = seqlens[:(int(len(data_indices) * TRAINING_PARTITION_SIZE))]

test_x = lyrics[(int(len(data_indices) * TRAINING_PARTITION_SIZE)):]
test_y = labels[(int(len(data_indices) * TRAINING_PARTITION_SIZE)):]
test_seqlens = seqlens[(int(len(data_indices) * TRAINING_PARTITION_SIZE)):]


def get_sentence_batch(batch_size, data_x, data_y, data_seqlens):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [[word2index_map[word] for word in data_x[i]]
         for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    return x, y, seqlens


_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE, max_len])
_labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_CLASSES])
# seqlens for dynamic calculation
_seqlens = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

with tf.name_scope("embeddings"):
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIMENSION], -1.0, 1.0), name='embedding')
    embed = tf.nn.embedding_lookup(embeddings, _inputs)

with tf.variable_scope("lstm"):
    # Define a function that gives the output in the right shape
    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(HIDDEN_LAYER_SIZE, forget_bias=1.0)


    cell = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell() for _ in range(NUM_LSTM_LAYERS)],
                                       state_is_tuple=True)
    outputs, states = tf.nn.dynamic_rnn(cell, embed,
                                        sequence_length=_seqlens,
                                        dtype=tf.float32)

# randomly initialize weights
weights = {
    'linear_layer': tf.Variable(tf.truncated_normal([HIDDEN_LAYER_SIZE, NUM_CLASSES], mean=0, stddev=.01))
}

# randomly initialize biases
biases = {
    'linear_layer': tf.Variable(tf.truncated_normal([NUM_CLASSES], mean=0, stddev=.01))
}
# extract the last relevant output and use in a linear layer
final_output = tf.matmul(states[NUM_LSTM_LAYERS - 1][1],
                         weights["linear_layer"]) + biases["linear_layer"]

softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output,
                                                  labels=_labels)
cross_entropy = tf.reduce_mean(softmax)

train_step = tf.train.RMSPropOptimizer(LEARNING_RATE, DECAY).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(_labels, 1),
                              tf.argmax(final_output, 1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction,
                                   tf.float32))) * 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(NUM_STEPS):
        x_batch, y_batch, seqlen_batch = get_sentence_batch(BATCH_SIZE, train_x, train_y, train_seqlens)
        sess.run(train_step, feed_dict={_inputs: x_batch, _labels: y_batch,
                                        _seqlens: seqlen_batch})

        if step % DISPLAY_STEP == 0:
            train_acc = sess.run(accuracy, feed_dict={_inputs: x_batch,
                                                      _labels: y_batch,
                                                      _seqlens: seqlen_batch})
            print("Step: " + str(step) + "\t\tTraining_accuracy=" + "{:.2f}".format(
                train_acc))

    for test_batch in range(5):
        x_test, y_test, seqlen_test = get_sentence_batch(BATCH_SIZE, test_x, test_y, test_seqlens)
        batch_pred, batch_acc = sess.run([tf.argmax(final_output, 1), accuracy],
                                         feed_dict={_inputs: x_test,
                                                    _labels: y_test,
                                                    _seqlens: seqlen_test})
        print("Test batch accuracy %d: %.5f" % (test_batch, batch_acc))
