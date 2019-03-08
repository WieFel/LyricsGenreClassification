import numpy as np
from FeatureExtraction import VECTORIZED_DATA, DIMENSIONS
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
import os
from tensorflow.python.tools import inspect_checkpoint as chkp


def get_batch(batch_size, data_x, data_y):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [data_x[i] for i in batch]
    y = [data_y[i] for i in batch]
    return np.array(x), np.array(y)

#Path were weights will be saved into
SAVE_PATH = './save'
MODEL_NAME = 'classifier'
FOLD_NUMBER = 1
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)


print("Reading dataset...")
data = np.load(VECTORIZED_DATA)
features = data[:, :-1]  # take first column: lyrics
genres = data[:, -1]  # take last column: genres

# Parameters
learning_rate = 0.001
num_steps = 1000
batch_size = 256
display_step = 20000

# Network Parameters
n_hidden_1 = 32  # 1st layer number of neurons
n_hidden_2 = 64  # 2nd layer number of neurons
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

X_learn, X_test, y_learn, y_test = train_test_split(features, labels, test_size=0.2, random_state=27*5)

accuracies = []
# perform k-fold cross validation with k=5
k_fold = KFold(n_splits=5)
for train_indices, test_indices in k_fold.split(X_learn):
    X_train, X_validation = features[train_indices], features[test_indices]
    y_train, y_validation = labels[train_indices], labels[test_indices]

    # tf Graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }


    # Create model
    def neural_net(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))

        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
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
    #Save weights
    saver = tf.train.Saver(weights)

    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        for step in range(1, num_steps + 1):
            batch_x, batch_y = get_batch(batch_size, X_train, y_train)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

            if step % display_step == 0:
                # Calculate batch loss and accuracy
                train_loss, train_acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                validation_loss, validation_acc = sess.run([loss_op, accuracy],
                                                           feed_dict={X: X_validation, Y: y_validation})
                print("Step: " + str(step) + "\t\tTrain_loss=" + "{:.2f}".format(
                    train_loss) + "\t\tTraining_accuracy=" + "{:.2f}".format(
                    train_acc) + "\t\tValidation_loss=" + "{:.2f}".format(
                    validation_loss) + "\t\tValidation_accuracy=" + "{:.2f}".format(validation_acc))

        if FOLD_NUMBER == 5:
            path = saver.save(sess, SAVE_PATH + '/' + MODEL_NAME + str(FOLD_NUMBER) + '.ckpt')
            print("saved at {}".format(path))
        FOLD_NUMBER += 1

        print("Optimization Finished!")

        accuracy = sess.run(accuracy, feed_dict={X: X_test, Y: y_test})
        print("Testing Accuracy:", accuracy)
        accuracies.append(accuracy)

print("Average accuracy: " + str(float(sum(accuracies)) / len(accuracies)))

#Print saved weights
#chkp.print_tensors_in_checkpoint_file('./save/classifier5.ckpt', tensor_name='', all_tensors=True)

