# -*- encoding:utf-8 -*-
from __future__ import print_function

import os
import sys
import numpy as np



def evaluate_recall(predicted_prob, y_test, k=3):
    num_examples = float(len(predicted_prob))
    num_correct = 0
    y=np.argsort(-np.array(predicted_prob))  ## reverse the predicted class rank,
    for predictions, label in zip(y, y_test):
        if label in predictions[:k]:
            num_correct += 1
    return num_correct/num_examples


# def compute_recall_ks(probas):
#     recall_k = {}
#     for group_size in [2, 5, 10]:
#         recall_k[group_size] = {}
#         print ('group_size: %d' % group_size)
#         for k in [1, 2, 5]:
#             if k < group_size:
#                 recall_k[group_size][k] = recall(probas, k, group_size)
#                 print ('recall@%d' % k, recall_k[group_size][k])
#     return recall_k

def compute_recall_ks(probas):
    recall_k = {}
    for group_size in [ 5, 10, 30, 61]:
        recall_k[group_size] = {}
        print ('group_size: %d' % group_size)
        for k in [3, 5]:
            if k < group_size:
                recall_k[group_size][k] = recall(probas, k, group_size)
                print ('recall@%d' % k, recall_k[group_size][k])
    return recall_k



def recall(probas, k, group_size):
    test_size = 61
    ## this is because the test/validation data is formulated as  context, groundTruth, distractor0, distractor1, ,,,,,distractor 59, in total 61
    ## so every 61 row use the same context and among them row #0 is always the pair of context with ground truth
    n_batches = len(probas) // test_size
    n_correct = 0
    for i in range(n_batches):
       # batch = np.array(probas[i*test_size:(i+1)*test_size])[:group_size]
        batch = np.array(probas[i * test_size:(i + 1) * test_size])[:group_size]
        #print ('predicted probs for batch %d is : \n'%i)
        #print (batch)
        #print ('shape of batch is ')
        #print (batch.shape)
        indices = np.argpartition(batch, -k)[-k:]
        if 0 in indices:
            n_correct += 1
    return float(n_correct) / (len(probas) / test_size)


#
# def recall_randomsampling(probas, k, group_size):
#     test_size = 61
#     ## this is because the test/validation data is formulated as  context, groundTruth, distractor0, distractor1, ,,,,,distractor 59, in total 61
#     ## so every 61 row use the same context and among them row #0 is always the pair of context with ground truth
#     n_batches = len(probas) // test_size
#     n_correct = 0
#     for i in range(n_batches):
#        # batch = np.array(probas[i*test_size:(i+1)*test_size])[:group_size]
#         batch = np.array(probas[i * test_size:(i + 1) * test_size])[:group_size]
#         #print ('predicted probs for batch %d is : \n'%i)
#         #print (batch)
#         #print ('shape of batch is ')
#         #print (batch.shape)
#         indices = np.argpartition(batch, -k)[-k:]
#         if 0 in indices:
#             n_correct += 1
#     return float(n_correct) / (len(probas) / test_size)




def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def subsample_Train_idx(data,percent):
    idx=np.random.choice(len(data), int(round(len(data)*percent)),replace=False)
    return idx

def subsample_Dev_idx(data,percent):
    Count_set=len(data)//10
    #tem_idx=np.random.choice(Count_set, int(round(Count_set*percent)),replace=False)
    #idx =range(int(round(Count_set*percent))*10)
    idx =[i for i in range(int(round(Count_set*percent))*10)]
    #idx=[data[j*10:j*10+10] for j in tem_idx]
    return idx