# -*- encoding:utf-8 -*-
from __future__ import print_function

import os
import sys
import numpy as np
import time
from utilities.data_helper import compute_recall_ks, str2bool,subsample_Train_idx,subsample_Dev_idx


def evalute_model(model,test_data):
    start_time = time.time()
    y_pred = model.predict(test_data)
    print("---model inference time takes %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    result = compute_recall_ks(y_pred[:,0])
    print("---model evaluation time takes %s seconds ---" % (time.time() - start_time))
    return result

def write_evaluation_result(recall_dic, result_file_path):
    fo= open(result_file_path,'w')
    for group, sub_dic in recall_dic. items():
        fo.write('\n\n\n group_size: %d\n'%group)
        for k,v in sub_dic.items():
            fo.write('recall @%d : %s \n' %(k,str(v)))
    fo.close()




def evaluate_recall_randomsampling(y, y_test, k=1):
    num_examples = float(len(y))
    num_correct = 0
    for predictions, label in zip(y, y_test):
        if label in predictions[:k]:
            num_correct += 1
    return num_correct/num_examples

def predict_random(template_pool_size):
    return np.random.choice(template_pool_size, template_pool_size, replace=False)
# Evaluate Random predictor

def evaluate_random_predictor(test_sample_size):
    y_random = [predict_random(61) for _ in range(test_sample_size)]
    y_test = np.zeros(len(y_random))
    for n in [3, 5]:
        print("Recall @ ({}, 61): {:g}".format(n, evaluate_recall_randomsampling(y_random, y_test, n)))



