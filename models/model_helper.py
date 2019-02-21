# -*- encoding:utf-8 -*-
from __future__ import print_function

import os
import sys
import numpy as np
import array
import numpy as np
import tensorflow as tf


def load_pretrained_embedding(embedding_file):
    '''prepare required embedding from pretained word embedding file
    '''
    print('Now indexing word vectors...')

    embeddings_index = {}
    f = open(embedding_file, encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except ValueError:
            continue
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def build_embedding_matrix(embeddings_index,max_nb_words, word_index, emb_dim):
    print ("Now loading embedding matrix...")
    num_words = min(max_nb_words, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words , emb_dim))
    for word, i in word_index.items():
        if i >= max_nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
