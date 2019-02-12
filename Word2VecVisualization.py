# for the visualization of the word2vec model
import os
import numpy as np
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

DATA_PATH = os.path.expanduser("~/NLP_Data/")
DIMENSIONS = 300

# loading your gensim
model = Word2Vec.load(DATA_PATH + "word2vec_model")

# project part of vocab
with open( DATA_PATH + "./projector/prefix_metadata.tsv", 'w+') as file_metadata:
    w2v_10K = np.zeros((len(model.wv.index2word), DIMENSIONS))
    for i, word in enumerate(model.wv.index2word):
        w2v_10K[i] = model[word]
        file_metadata.write(word.encode('utf-8') + '\n')

# define the model without training
sess = tf.InteractiveSession()

with tf.device("/cpu:0"):
    embedding = tf.Variable(w2v_10K, trainable=False, name='word_embeddings')

tf.global_variables_initializer().run()

saver = tf.train.Saver()
writer = tf.summary.FileWriter('./projector', sess.graph)

# adding into projector
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = 'fs_embedding:0'
embed.metadata_path = DATA_PATH + './projector/prefix_metadata.tsv'

# Specify the width and height of a single thumbnail.
projector.visualize_embeddings(writer, config)

saver.save(sess, DATA_PATH + './projector/prefix_model.ckpt', global_step=10000)

# open tensorboard with logdir, check localhost:6006 for viewing your embedding.
# tensorboard --logdir="./projector/"