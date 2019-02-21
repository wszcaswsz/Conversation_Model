# -*- encoding:utf-8 -*-
from __future__ import print_function

import numpy as np
import pickle
# from keras.models import Sequential
# from keras.utils import np_utils
# from keras.layers import Dense, Input, Flatten, Dropout, LSTM, Activation, Multiply,concatenate#,Merge
# from keras.layers import Conv1D, MaxPooling1D, Embedding, merge,GlobalMaxPooling1D
# from keras.layers.merge import concatenate
# from keras.models import Model
#from keras.utils.vis_utils import plot_model
import time
from utilities import my_callbacks
import argparse
import pandas as pd

import keras
from utilities.data_helper import compute_recall_ks, str2bool,subsample_Train_idx,subsample_Dev_idx
from utilities.evalute_model import evalute_model, write_evaluation_result, evaluate_recall_randomsampling, predict_random, evaluate_random_predictor
from models.model_helper import load_pretrained_embedding, build_embedding_matrix
from models.model import CNN_encoder_multiFilter, transformer_encoder, transformer_encoder_v2, CNN_encoder, LSTM_encoder, Train,Train_withEarlyStopandModelSave


## check whether GPU is available
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import keras
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import sys
print ('python version is')
print(sys.version)


#  MultiFilterCNN_dualencoder_1MGPU_500epoch_EStest.h5_1550600687.5703459.h5

def main():
    
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    parser.add_argument('--emb_dim', type=str, default=100, help='Embeddings dimension')
    parser.add_argument('--hidden_size', type=int, default=300, help='Hidden size')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=500, help='Num epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--n_recurrent_layers', type=int, default=1, help='Num recurrent layers')
    parser.add_argument('--input_dir', type=str, default='../../email_template_recommendation_code/dataset/', help='Input dir')
    parser.add_argument('--save_model', type='bool', default=True, help='Whether to save the model')
    parser.add_argument('--pretrained_model_fname', type=str, default='model_dir/dual_encoder_lstm_classifier_10KTraining.h5', help='pretrained Model filename')
    # parser.add_argument('--model_weight', type=str, default='model/dual_encoder_lstm_classifier_5KTraining_weight.h5', help='Model weight')
    # parser.add_argument('--model_architecture', type=str, default='model/dual_encoder_lstm_classifier_5KTraining_architecture.json', help='Model architecture')
    parser.add_argument('--model_fname', type=str, default='data/model/dual_encoder_CNN_multiFilter.h5', help='Model filename')
    parser.add_argument('--embedding_file', type=str, default='../../dual_encoder_keras_for_MLfoundation/embeddings/glove.6B.100d.txt', help='Embedding filename')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--subsample_percent', type=float, default=1, help='sub sampling rate')
    parser.add_argument('--encoder_type', type=str, default='CNN_multi_filter', help='which encoder to use, options are LSTM, CNN, CNN_multi_filter, feedforward, transformer')
    parser.add_argument('--model_plot', type=str, default='/data/email_template_recommendation/model_architect_plot.jpg', help='file path to store the model architech plot')
    parser.add_argument('--loss_plot_path', type=str, default='/data/email_template_recommendation/loss_plot.png',help='file path to store the training and validation loss plot')
    parser.add_argument('--loss_file_path', type=str, default='/data/email_template_recommendation/train_val_loss.csv',help='file path to store the training and validation loss ')
    parser.add_argument('--evaluation_file_path', type=str, default='/data/email_template_recommendation/test_data_evaluation.txt',help='file path to store the training and validation loss ')
    parser.add_argument('--mode', type=str,default='train', help='choose from train/test,  are we training the model or testing the model  ')

    args = parser.parse_args()
    print ('Model args: ', args)
    np.random.seed(args.seed)
 
    print("Starting...")


    print("Now loading UDC test data...")

    dev_c, dev_r, dev_l = pickle.load(open(args.input_dir + 'dev.pkl', 'rb'))
    test_c, test_r, test_l = pickle.load(open(args.input_dir + 'test.pkl', 'rb'))
    train_c, train_r, train_l = pickle.load(open(args.input_dir + 'train.pkl', 'rb'))
    print('Found %s test samples.' % len(test_c))

    
    ## do subsampling of test data
    subsample_Percent=args.subsample_percent
    print ('subsampling test data ratio is %s'%subsample_Percent)
    if subsample_Percent< 1: 

        test_sub_idx=subsample_Dev_idx(test_c,subsample_Percent)
        test_c_new, test_r_new, test_l_new=test_c[test_sub_idx],test_r[test_sub_idx], np.array([test_l[i] for i in test_sub_idx])


    else:
        train_c_new, train_r_new, train_l_new=train_c, train_r ,train_l
        test_c_new, test_r_new, test_l_new=test_c,test_r, test_l
        dev_c_new, dev_r_new, dev_l_new=dev_c,dev_r, dev_l
    
    print('Found %s test samples after %s percent subsampling' % (len(test_c_new),subsample_Percent))


    ## prepare required embedding and word vector


    ### load parameters
    MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, word_index = pickle.load(open(args.input_dir + 'params.pkl', 'rb'))
    emb_dim= args.emb_dim
    print("MAX_SEQUENCE_LENGTH: {}".format(MAX_SEQUENCE_LENGTH))
    print("MAX_NB_WORDS: {}".format(MAX_NB_WORDS))
    print("emd_dim: {}".format(emb_dim))

    print('Now indexing word vectors...')

    embeddings_index = load_pretrained_embedding(args.embedding_file)




    # embeddings_index = {}
    # f = open(args.embedding_file, encoding="utf-8")
    # for line in f:
    #     values = line.split()
    #     word = values[0]
    #     try:
    #         coefs = np.asarray(values[1:], dtype='float32')
    #     except ValueError:
    #         continue
    #     embeddings_index[word] = coefs
    # f.close()
    
    

    print("Now loading embedding matrix...")
    embedding_matrix= build_embedding_matrix(embeddings_index,MAX_NB_WORDS, word_index, emb_dim)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    # num_words = min(MAX_NB_WORDS, len(word_index)) + 1
    # embedding_matrix = np.zeros((num_words , emb_dim))
    # for word, i in word_index.items():
    #     if i >= MAX_NB_WORDS:
    #         continue
    #     embedding_vector = embeddings_index.get(word)
    #     if embedding_vector is not None:
    #         # words not found in embedding index will be all-zeros.
    #         embedding_matrix[i] = embedding_vector


    print("Now building dual encoder model...")

    if args.encoder_type == 'CNN_multi_filter':              
        model = CNN_encoder_multiFilter(emb_dim,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH,embedding_matrix,args.optimizer)

        # define transformer encoder
    elif args.encoder_type == 'transformer':
        model = transformer_encoder_v2(emb_dim, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, embedding_matrix, args.optimizer)

    # define lstm encoder
    elif args.encoder_type == 'CNN':              
        model = CNN_encoder(emb_dim,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH,embedding_matrix,args.optimizer)

    elif args.encoder_type == 'LSTM':              
        model = LSTM_encoder(emb_dim,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH,embedding_matrix,args.optimizer)






    

    ## reconstruct model architecture and load weight from stored weights
    #model =ReconstructModel_and_loadWeight(args.pretrained_model_fname)

    ## start the training process
    print(model.summary())
    #Train(model, args.batch_size, args.n_epochs,)
   # model = Train(model, args.batch_size, args.n_epochs, train_c_new, train_r_new, train_l_new, dev_c_new, dev_r_new, dev_l_new)
    if args.mode == 'train':
        print ('start training process')
        model, history = Train_withEarlyStopandModelSave(model,args.model_fname, args.loss_plot_path, args.batch_size, args.n_epochs, train_c_new, train_r_new, train_l_new, dev_c_new, dev_r_new, dev_l_new)
        print ('training finish')
        print (history.history['loss'])
        print (history.history['val_loss'])
        loss_data= pd.DataFrame.from_dict({'train_loss':history.history['loss'],'val_loss':history.history['val_loss']})
        print (loss_data)
        loss_data.to_csv(args.loss_file_path)
    #f_out = open(args.loss_file_path,'w')


#     val_loss = model.history['val_loss'])
#     val_loss = model.history['val_loss']
    #if args.save_model:
     #   print("Now saving the model... at {}".format(args.model_fname))
      #  model.save(args.model_fname)
    ## evalute all test data
    #print ('/n/n******************/n/n')
    #print('evalute all test data')
    #evalute_model(Model,[test_c, test_r])

    ## evalute subsampled test data

    if args.mode in ['test','train']:
        print ('/n/n******************/n/n')
        print('evalute subsampled test data at rate %s'%subsample_Percent)

        ## reload the weight from previous saved model
        if args.mode == 'test':

            model.load_weights(args.pretrained_model_fname)

        res = evalute_model(model,[test_c_new, test_r_new])

        write_evaluation_result(res, args.evaluation_file_path)

    ## evaluate performance purely based on random predictor:
    #evaluate_random_predictor(358)

    ## plot the loss and save to figures
    #plot_loss(history, args.loss_plot)





    ## evalute all test data
    #print ('/n/n******************/n/n')
    #print('evalute all test data')
    #evalute_model(model,[test_c, test_r])

if __name__ == "__main__":
    main()


