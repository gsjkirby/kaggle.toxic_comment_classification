# Original Model: Chip Hugyen
# Changes by Georgina and Carole:
#
#==============================================================================

# Future Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports
import os
import numpy as np

# Tensorflow Imports
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

# Internal Import
from process_data import process_data

# Variables
# Number of words to train on word embedding
VOCAB_SIZE = 50000
# Number of words to train in a single epoch
BATCH_SIZE = 128
# Dimension of the word embedding
EMBED_SIZE = 128
# The context window
SKIP_WINDOW = 1
# Number of negative samples
NUM_SAMPLED = 64
# Learning rate
LEARNING_RATE = 1.0
# Number of training steps
NUM_TRAIN_STEPS = 10000
# How many steps to do before printing out a loss
SKIP_STEP = 2000

def word2vec(batch_gen):
    """
    Build the graph for word2vec model and train it.

    Key Arguments:
    batch_gen: Batch generator
    """
    # Input and Output Placeholders
    with tf.name_scope('data'):
        center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE],
                                      name='center_words')
        target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1],
                                      name='target_words')

    # Weights of the models (word embeddings)
    with tf.name_scope('embedding_matrix'):
        embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE],
                                                     -1.0, 1.0),
                                                     name='embed_matrix')

    # Loss function
    with tf.name_scope('loss'):
        # Look up word embedding for each word
        embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')

        # Initialise weights and bias
        nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE],
                                                    stddev=1.0 / (EMBED_SIZE ** 0.5)),
                                                    name='nce_weight')
        nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')

        # Loss Function
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                             biases=nce_bias,
                                             labels=target_words,
                                             inputs=embed,
                                             num_sampled=NUM_SAMPLED,
                                             num_classes=VOCAB_SIZE),
                              name='loss')

    # Optimiser (GradientDescentOptimizer)
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    with tf.Session() as sess:
        # Initliaise all variables in graph
        sess.run(tf.global_variables_initializer())
        # Initialise loss to zero
        total_loss = 0.0
        # Write tensor graph to directory
        writer = tf.summary.FileWriter('./graphs/', sess.graph)
        # Go through and train model
        for index in range(NUM_TRAIN_STEPS):
            # Generate a new batch
            centers, targets = next(batch_gen)
            # Run the optimiser and return the loss
            loss_batch, _ = sess.run([loss, optimizer],
                                    feed_dict={center_words: centers,
                                               target_words: targets})
            # Add up the loss
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index,
                                                                total_loss / SKIP_STEP))
                total_loss = 0.0
        writer.close()

def main():
    # Collect your batch
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    # Train word2vec
    word2vec(batch_gen)

if __name__ == '__main__':
    main()
