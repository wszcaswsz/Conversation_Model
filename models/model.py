# -*- encoding:utf-8 -*-
from __future__ import print_function

import os
import sys
import numpy as np
from models import model_helper
from models.attention import Position_Embedding, Attention
import numpy as np
import pickle
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Input, Flatten, Dropout, LSTM, Activation, Multiply,concatenate#,Merge
from keras.layers import Conv1D, MaxPooling1D, Embedding, merge,GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Model
#from keras.utils.vis_utils import plot_model
import time
from utilities import my_callbacks
import argparse
from utilities.data_helper import compute_recall_ks, str2bool,subsample_Train_idx,subsample_Dev_idx
import keras

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import keras
import numpy as np



import random, os, sys
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf
from keras.engine.topology import Layer


class TrainingPlot(keras.callbacks.Callback):

    def __init__(self, filename):
        self.filename = filename

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        #self.acc = []
        self.val_losses = []
        #self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        #self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        #self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            #plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, label = "val_loss")
            #plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig(self.filename)
            plt.close()


def transformer_encoder(emb_dim,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH,embedding_matrix,optimizer):
    context_input = Input(shape=(None,), dtype='int32')
    response_input = Input(shape=(None,), dtype='int32')
    #context_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    #response_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(output_dim=emb_dim,
                            input_dim=MAX_NB_WORDS,
                            input_length=MAX_SEQUENCE_LENGTH,
                            weights=[embedding_matrix],
                            #mask_zero=True,
                            trainable=True)

    embedded_sequences_c = embedding_layer(context_input)
    embedded_dropout_c=Dropout(0.2)(embedded_sequences_c)
    embeddings_final_c = Position_Embedding()(embedded_dropout_c)   ## add positional embedding from self-attention
    embedded_sequences_r = embedding_layer(response_input)
    embedded_dropout_r=Dropout(0.2)(embedded_sequences_r)
    embeddings_final_r = Position_Embedding()(embedded_dropout_r)
    print("Now building encoder model with self attention...")

    c_seq = Attention(8, 16)([embeddings_final_c, embeddings_final_c, embeddings_final_c])   ## the three embedding input is for K,V,Q needed for self-attention
    c_seq = GlobalAveragePooling1D()(c_seq)
    c_seq = Dropout(0.2)(c_seq)

    r_seq = Attention(8, 16)([embeddings_final_r, embeddings_final_r,embeddings_final_r])  ## the three embedding input is for K,V,Q needed for self-attention
    r_seq = GlobalAveragePooling1D()(r_seq)
    r_seq = Dropout(0.2)(r_seq)


    concatenated = Multiply()([c_seq, r_seq])

    out = Dense((1), activation = "sigmoid") (concatenated)

    model = Model([context_input, response_input], out)
    model.compile(loss='binary_crossentropy',
                optimizer=optimizer)
# print(encoder.summary())
    print(model.summary())
    return model




def transformer_encoder_v2(emb_dim,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH,embedding_matrix,optimizer):
    context_input = Input(shape=(None,), dtype='int32')
    response_input = Input(shape=(None,), dtype='int32')
    #context_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    #response_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    # embedding_layer = Embedding(output_dim=emb_dim,
    #                         input_dim=MAX_NB_WORDS,
    #                         input_length=MAX_SEQUENCE_LENGTH,
    #                         weights=[embedding_matrix],
    #                         #mask_zero=True,
    #                         trainable=True)

    embedded_sequences_c = Embedding(MAX_NB_WORDS,128)(context_input)
    #embedded_dropout_c=Dropout(0.2)(embedded_sequences_c)
    embeddings_final_c = Position_Embedding()(embedded_sequences_c)   ## add positional embedding from self-attention
    embedded_sequences_r = Embedding(MAX_NB_WORDS,128)(response_input)
    #embedded_dropout_r=Dropout(0.2)(embedded_sequences_r)
    embeddings_final_r = Position_Embedding()(embedded_sequences_r)
    print("Now building encoder model with self attention...")

    c_seq = Attention(8, 16)([embeddings_final_c, embeddings_final_c, embeddings_final_c])   ## the three embedding input is for K,V,Q needed for self-attention
    c_seq = GlobalAveragePooling1D()(c_seq)
    c_seq = Dropout(0.2)(c_seq)

    r_seq = Attention(8, 16)([embeddings_final_r, embeddings_final_r,embeddings_final_r])  ## the three embedding input is for K,V,Q needed for self-attention
    r_seq = GlobalAveragePooling1D()(r_seq)
    r_seq = Dropout(0.2)(r_seq)


    concatenated = Multiply()([c_seq, r_seq])
#concatenated = merge([context_branch, response_branch], mode='mul')
    out = Dense((1), activation = "sigmoid") (concatenated)

    model = Model([context_input, response_input], out)
    model.compile(loss='binary_crossentropy',
                optimizer=optimizer)
# print(encoder.summary())
    print(model.summary())
    return model




def CNN_encoder_multiFilter(emb_dim,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH,embedding_matrix,optimizer):
    context_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    response_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(output_dim=emb_dim,
                            input_dim=MAX_NB_WORDS,
                            input_length=MAX_SEQUENCE_LENGTH,
                            weights=[embedding_matrix],
                            #mask_zero=True,
                            trainable=True)

    embedded_sequences_c = embedding_layer(context_input)
    embedded_dropout_c=Dropout(0.2)(embedded_sequences_c)
    embedded_sequences_r = embedding_layer(response_input)
    embedded_dropout_r=Dropout(0.2)(embedded_sequences_r)

    print("Now building encoder model with multiple filters of CNN...")
#if args.encoder_type == 'CNN':
#    print ('use CNN as encoder...')
    convs_c, convs_r= [],[]
    filter_sizes = [3,4,5]

    for fsz in filter_sizes:
        l_conv_c, l_conv_r = Conv1D(nb_filter=128,filter_length=fsz,activation='relu')(embedded_dropout_c), Conv1D(nb_filter=128,filter_length=fsz,activation='relu')(embedded_dropout_r)
        l_pool_c, l_pool_r = GlobalMaxPooling1D()(l_conv_c), GlobalMaxPooling1D()(l_conv_r)
        convs_c.append(l_pool_c)
        convs_r.append(l_pool_r)

    context_branch = concatenate(convs_c)
    response_branch = concatenate(convs_r)

# concatenated = concatenate([context_branch, response_branch])
    concatenated = Multiply()([context_branch, response_branch])
#concatenated = merge([context_branch, response_branch], mode='mul')
    out = Dense((1), activation = "sigmoid") (concatenated)

    model = Model([context_input, response_input], out)
    model.compile(loss='binary_crossentropy',
                optimizer=optimizer)
# print(encoder.summary())
    print(model.summary())
    return model




def CNN_encoder(emb_dim,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH,embedding_matrix,optimizer):
    encoder = Sequential()
    encoder.add(Embedding(output_dim=emb_dim,
                        input_dim=MAX_NB_WORDS,
                        input_length=MAX_SEQUENCE_LENGTH,
                        weights=[embedding_matrix],
                        #mask_zero=True,
                        trainable=True))

    
    print ('use CNN as encoder...')
    encoder.add(Dropout(0.2))
    encoder.add(Conv1D(512,3,
                 padding='valid',
                 activation='relu',
                 strides=1))
    encoder.add(GlobalMaxPooling1D())

# encoder.add(LSTM(units=args.hidden_size))
    context_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    response_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# encode the context and the response
    context_branch = encoder(context_input)
    response_branch = encoder(response_input)
    concatenated = Multiply()([context_branch, response_branch])
#concatenated = merge([context_branch, response_branch], mode='mul')
    out = Dense((1), activation = "sigmoid") (concatenated)
    model = Model([context_input, response_input], out)
    model.compile(loss='binary_crossentropy',
                optimizer=optimizer)
    return model


def LSTM_encoder(emb_dim,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH,embedding_matrix,optimizer):
    encoder = Sequential()
    encoder.add(Embedding(output_dim=emb_dim,
                        input_dim=MAX_NB_WORDS,
                        input_length=MAX_SEQUENCE_LENGTH,
                        weights=[embedding_matrix],
                        #mask_zero=True,
                        trainable=True))
    print ('use LSTM as encoder...')
    encoder.add(LSTM(units=args.hidden_size))


# encoder.add(LSTM(units=args.hidden_size))
    context_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    response_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# encode the context and the response
    context_branch = encoder(context_input)
    response_branch = encoder(response_input)
    concatenated = Multiply()([context_branch, response_branch])
#concatenated = merge([context_branch, response_branch], mode='mul')
    out = Dense((1), activation = "sigmoid") (concatenated)
    model = Model([context_input, response_input], out)
    model.compile(loss='binary_crossentropy',
                optimizer=optimizer)
    return model


def Train(model,batch_size, n_epochs,train_c_new, train_r_new, train_l_new,dev_c_new, dev_r_new, dev_l_new):
    print("Now training the model...")
    # histories = Histories()
    histories = my_callbacks.Histories()
    start_time = time.time()

    # start_time = time.time()
    # compute_recall_ks(y_pred[:,0])
    # print("---model evaluation time takes %s seconds ---" % (time.time() - start_time))
    bestAcc = 0.0
    patience = 0

    print("\tbatch_size={}, nb_epoch={}".format(batch_size, n_epochs))

    # for ep in range(1, args.n_epochs):
    for ep in range(1, n_epochs):

        model.fit([train_c_new, train_r_new], train_l_new,
                         batch_size=batch_size, epochs=n_epochs, callbacks=[histories],
                         validation_data=([dev_c_new, dev_r_new], dev_l_new), verbose=1)

        curAcc = histories.accs[0]
        if curAcc >= bestAcc:
            bestAcc = curAcc
            patience = 0
        else:
            patience = patience + 1

        # classify the test set
        y_pred = model.predict([test_c_new, test_r_new])

        print("Perform on test set after Epoch: " + str(ep) + "...!")
        recall_k = compute_recall_ks(y_pred[:, 0])

        # stop training the model when patience = 10
        if patience > 10:
            print("Early stopping at epoch: " + str(ep))
            break
    print("---Training finished, model training time takes %s seconds ---" % (time.time() - start_time))
    return model


def Train_withEarlyStopandModelSave(model,model_save_path,loss_plot_path,batch_size, n_epochs,train_c_new, train_r_new, train_l_new,dev_c_new, dev_r_new, dev_l_new):
    print("Now training the model...")
    # histories = Histories()
    #histories = my_callbacks.Histories()
    start_time = time.time()
    model_save_path = model_save_path + '_'+ str(start_time) + '.h5' ## add timestamp to the saved model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(model_save_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    print (loss_plot_path)
    plot_losses = TrainingPlot(loss_plot_path)
    model_history = model.fit([train_c_new, train_r_new], train_l_new,
                         batch_size=batch_size, epochs=n_epochs, callbacks=[es, mc,plot_losses],
                         validation_data=([dev_c_new, dev_r_new], dev_l_new), verbose=1)

    
    print("---Training finished, model training time takes %s seconds ---" % (time.time() - start_time))
    return model, model_history





