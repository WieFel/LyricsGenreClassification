import numpy as np
from FeatureExtraction import VECTORIZED_DATA, DIMENSIONS
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from collections import Counter


def get_batch(batch_size, data_x, data_y):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [data_x[i] for i in batch]
    y = [data_y[i] for i in batch]
    return np.array(x), np.array(y)


print("Reading dataset...")
data = np.load(VECTORIZED_DATA)
features = data[:, :-1]  # take first column: lyrics
genres = data[:, -1]  # take last column: genres

# Parameters
learning_rate = 0.000001
num_steps = 10000
batch_size = 128
display_step = 1000

# Network Parameters
n_hidden_1 = 128  # 1st layer number of neurons
n_hidden_2 = 128  # 2nd layer number of neurons
n_hidden_3 = 128  # 3nd layer number of neurons
n_hidden_4 = 128  # 3nd layer number of neurons
n_hidden_5 = 128  # 3nd layer number of neurons
n_hidden_6 = 128  # 3nd layer number of neurons
n_hidden_7 = 128  # 3nd layer number of neurons
num_input = DIMENSIONS
num_classes = len(set(genres))

unique_genres = list(set(genres))
genre_to_index = {}
index_to_genre = [""] * len(unique_genres)  # init list with number of genres
for g in unique_genres:
    i = unique_genres.index(g)
    genre_to_index[g] = i
    index_to_genre[i] = g

labels = np.empty((len(genres), num_classes))
for i in range(len(genres)):
    label = genre_to_index[genres[i]]  # get genre index
    one_hot_encoding = [0] * num_classes
    one_hot_encoding[label] = 1
    labels[i, :] = np.array(one_hot_encoding)


accuracies = []
# perform k-fold cross validation with k=5
k_fold = KFold(n_splits=5)
for train_indices, test_indices in k_fold.split(features):
    X_train, X_test = features[train_indices], features[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]

    # print counts of each genre
    ytrain = [index_to_genre[np.nonzero(onehot)[0][0]] for onehot in y_train]
    ytest = [index_to_genre[np.nonzero(onehot)[0][0]] for onehot in y_test]
    print("y_train: " + str(Counter(ytrain)))
    print("y_test: " + str(Counter(ytest)))

    # tf Graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
        'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
        'h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6])),
        'h7': tf.Variable(tf.random_normal([n_hidden_6, n_hidden_7])),
        'out': tf.Variable(tf.random_normal([n_hidden_7, num_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'b4': tf.Variable(tf.random_normal([n_hidden_4])),
        'b5': tf.Variable(tf.random_normal([n_hidden_5])),
        'b6': tf.Variable(tf.random_normal([n_hidden_6])),
        'b7': tf.Variable(tf.random_normal([n_hidden_7])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }


    # Create model
    def neural_net(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
        layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4']))
        layer_5 = tf.nn.relu(tf.add(tf.matmul(layer_4, weights['h5']), biases['b5']))
        layer_6 = tf.nn.relu(tf.add(tf.matmul(layer_5, weights['h6']), biases['b6']))
        layer_7 = tf.nn.relu(tf.add(tf.matmul(layer_6, weights['h7']), biases['b7']))

        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_7, weights['out']) + biases['out']
        return out_layer


    # Construct model
    logits = neural_net(X)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        for step in range(1, num_steps + 1):
            batch_x, batch_y = get_batch(batch_size, X_train, y_train)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(
                    loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

        print("Optimization Finished!")

        accuracy = sess.run(accuracy, feed_dict={X: X_test, Y: y_test})
        print("Testing Accuracy:", accuracy)
        accuracies.append(accuracy)

print("Average accuracy: " + str(float(sum(accuracies))/len(accuracies)))
