# Original Model: Chip Hugyen
# Changes by Georgina and Carole:
# - Added method text_to_word_list which cleans and formats the words
# Methods used by word2vec.py
#==============================================================================

# Future Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports
from collections import Counter
import random
import os
import sys
sys.path.append('..')
import zipfile
import re
from gensim.models import Phrases
import numpy as np
import utils
import pandas as pd
from six.moves import urllib
from itertools import chain, combinations
from nltk.stem import WordNetLemmatizer
import nltk

# Tensorflow Imports
import tensorflow as tf

# Directory Paths
path_to_text = os.getcwd() + '/csv_files/train_data.csv'

# Stopwords
stopwords = set(word.rstrip() for word in open('stopwords.txt'))

def text_to_word_list(text):
    """
    Text to words including processing steps.
    Reads in sentences and returns words.

    Keyword Arguments:
    text: Text to preprocess.
    """
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()
    # Remove short words, they're probably not useful
    text = [word for word in text if len(word) > 2]
    # Remove stopwords
    text = [word for word in text if word not in stopwords]
    # Remove any digits, i.e. "3rd edition"
    #text = [t for t in text if not any(c.isdigit() for c in t)]
    # Returns a list of words in one sentence
    return text

def read_data(file_path):
    """
    Returns a list of words in sentence order.

    Key Arguments:
    file_path: Path to training data csv.
    """
    # Read in data source using pandas
    # Function that cleans it and returns it in the correct format
    return words

def build_vocab(words, vocab_size):
    """
    Build vocabulary of VOCAB_SIZE most frequent words

    Key Arguments:
    words: Takes words from read_data() as a list of strings.
    vocab_size: Number of words to train word embeddings.
    """
    # Dictionary
    dictionary = dict()
    # Add Unknown for words outside VOCAB_SIZE
    count = [('UNK', -1)]
    # Count top vocab_size most common words
    count.extend(Counter(words).most_common(vocab_size - 1))
    # Initialise index
    index = 0
    # Make directoary
    utils.make_dir('processed')
    # Open file in write mode
    with open('processed/vocab_1000.tsv', "w") as file:
        # For word in top most common words
        for word, _ in count:
            # Make a dictionary entry with the index being the order the word
            # appears
            dictionary[word] = index
            # When the index is smaller than the vocab size
            if index < 1000:
                # Write to vocab file
                file.write(word + "\n")
            index += 1
    # Index to Value (inverse dictionary)
    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, index_dictionary

def convert_words_to_index(words, dictionary):
    """
    Replace each word in the dataset with its index in the dictionary.

    Key Arguments:
    words: List of words
    dictionary: Dictionary used to convert words to indices
    """
    return [dictionary[word] if word in dictionary else 0 for word in words]

def generate_sample(index_words, context_window_size):
    """
    Form training pairs according to the skip-gram model.

    Key Arguments:
    index_words: List of converted words into indices.
    context_window_size: How many words to consider either side of center.
    """
    for index, center in enumerate(index_words):
        # Generate a random number between 1 and the context window size
        context = random.randint(1, context_window_size)
        # Get random targets before the center word
        for target in index_words[max(0, index - context): index]:
            yield center, target
        # Get random targets after the center word
        for target in index_words[index + 1: index + context + 1]:
            yield center, target

def get_batch(iterator, batch_size):
    """
    Group a numerical stream into batches and yield them as Numpy arrays.

    Key Arguments:
    iterator: Generator function (center, target).
    batch_size: How many sample pairs to have in one batch.
    """
    while True:
        # Initalise zero matrices
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        # For each batch
        for index in range(batch_size):
            # Call next pair
            center_batch[index], target_batch[index] = next(iterator)
        # Yields a batch size
        yield center_batch, target_batch

def process_data(vocab_size, batch_size, skip_window):
    words = read_data(file_path)
    dictionary, _ = build_vocab(words, vocab_size)
    index_words = convert_words_to_index(words, dictionary)
    # Delete words once no longer needed to save memory
    del words
    single_gen = generate_sample(index_words, skip_window)
    # Returns batch generator
    return get_batch(single_gen, batch_size)

def get_index_vocab(vocab_size):
    file_path = download(FILE_NAME, EXPECTED_BYTES)
    words = read_data(file_path)
    return build_vocab(words, vocab_size)
