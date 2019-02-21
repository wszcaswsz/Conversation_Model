
import keras

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

import matplotlib.pyplot as plt

plt.style.use('ggplot')
from sklearn.model_selection import train_test_split
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.layers import Embedding, Flatten, Dense, Dropout, Activation
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras import metrics



import numpy as np



def read_file(inputfile):
    labels = []
    texts = []
    f = open(inputfile, 'r')
    k = 0
    r = 0
    record = set()
    skip = 0
    for line in f:
        # print(line)
        k += 1
        if k == 1:
            continue

        split_line = line.rstrip().replace('"', '').replace('\xa0', '  ').split(",")
        cat = split_line[-1]
        text = ','.join(split_line[:len(split_line) - 1])
        # print (cat)
        # print (text)
        if (cat, text) not in record:
            if cat != '00163E3CD6251EE7B7E962A036702F73':
                record.add((cat, text))
                texts.append(text)
                labels.append(cat)
                r += 1
            else:
                skip += 1
        else:
            skip += 1
    print('read in total lines')
    print(r)
    print('skip duplicated lines')
    print(skip)
    f.close()
    return labels, texts


# In[5]:

## import template info
template_text = {}
f = open('Template_withText_Parsed_out_61template.csv', encoding="utf8")
for line in f:
    tem = line.rstrip().split(',')
    UUID = tem[-2]
    count = tem[-1]
    template_text_str = ','.join(tem[:-2])
    template_text[UUID] = template_text_str
f.close()




def read_file_toPrepareData_for_DualEncoder(inputfile, template_text_dic):
    labels = []
    texts = []
    f = open(inputfile, 'r')
    k = 0
    record = set()
    skip = 0
    missing = 0
    # generic = 0
    for line in f:
        # print(line)
        k += 1
        if k == 1:
            continue
        print(line)
        split_line = line.rstrip().replace('"', '').replace('\xa0', '  ').split(",")
        cat = split_line[-1]
        text = ','.join(split_line[:len(split_line) - 1])
        # print (cat)
        # print (text)
        # if cat in ['00163E3CD6251EE7B7E962A036702F73','00163E3CD6251EE7B8B4AEC37397BDF3']:
        #   generic +=1
        #  continue
        if (cat, text) not in record:
            # if cat != '00163E3CD6251EE7B7E962A036702F73':

            if cat in template_text_dic:
                record.add((cat, text))
                # if cat in ['00163E3CD6251EE7B7E962A036702F73','00163E3CD6251EE7B8B4AEC37397BDF3']:
                #   cat = 'generic'
                # else:
                #   cat = 'specific'
                texts.append(text)
                labels.append(cat)

            else:
                missing += 1
        else:
            skip += 1
    print('read in total lines')
    print(k - 1)
    print('skip duplicated lines')
    print(skip)
    # print ('skip generic lines')
    # print (generic)
    print('missing template content lines')
    print(missing)

    f.close()
    return labels, texts



label, text = read_file_toPrepareData_for_DualEncoder(
    'Filtered_IncomingEmailText_with_correspondingTemplateUsed_InteractionNumber1and0only.csv', template_text)
len(label)
len(text)



texts = text

# In[30]:

## check the length of the text
length_text = [len(s) for s in texts]
# length_text
print('Maximum length of texts is:')
max(length_text)

import statistics

print('Average length of texts is:')
sum(length_text) / len(length_text)

print('Median length of texts is:')
statistics.median(length_text)

total_categories = len(list(set(label)))
print('total L1 categories are:')
print(total_categories)



np.random.seed(0)
indices = np.arange(len(texts))
print(indices[:10])
np.random.shuffle(indices)
print(indices[:10])
texts_shuffled = [texts[i] for i in indices]
# L1_label = [L1_label[i] for i in indices]
label_shuffled = [label[i] for i in indices]



training_samples = round(len(texts_shuffled) * 0.7)
validation_samples = round(len(texts_shuffled) * 0.15)



texts_train = texts_shuffled[:training_samples]
label_train = label_shuffled[:training_samples]
texts_val = texts_shuffled[training_samples:training_samples + validation_samples]
label_val = label_shuffled[training_samples:training_samples + validation_samples]
texts_test = texts_shuffled[training_samples + validation_samples:]
label_test = label_shuffled[training_samples + validation_samples:]



import re


import random
import re
import unicodecsv
import csv


def clean_str(text):
    text = text.encode('ascii', 'ignore').decode('utf-8')
    # print (text)
    # text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    # text = re.sub(r'http\S+', '', text)
    # print (text)
    # cleanString = re.sub('[^A-Za-z0-9]+', ' ', text)
    cleanString = re.sub('\W+', ' ', text)
    return cleanString


def write_file_for_DualEncoder(email_text, template_text, template_dic, mode):
    ## email_text is email, template_text is corrresponding used template ID for each email, template_dic is the dic that { templateID: templatetext}
    template_collection = [clean_str(v) for k, v in template_dic.items()]
    f = unicodecsv.writer(open('%s.csv' % mode, 'wb'), encoding='utf-8')
    # f= open('%s.csv'%mode,'w',encoding="utf8")
    if mode == 'train':
        header = ["Context", "Utterance", "Label"]
        # f.write('Context,Utterance,Label\n')
    else:
        header = ["Context", "Ground Truth Utterance"]
        header.extend(map(lambda x: "Distractor_{}".format(x), range(len(template_collection) - 1)))
    f.writerow(header)
    # f.writerow('Context,Ground Truth Utterance,Distractor_0,Distractor_1,Distractor_2,Distractor_3,Distractor_4,Distractor_5,Distractor_6,Distractor_7,Distractor_8\n')
    assert len(email_text) == len(template_text)
    for i in range(len(email_text)):
        Context = clean_str(email_text[i]).replace('#', '')
        Utterance = clean_str(template_dic[template_text[i]]).replace('/#', '')
        template_other = [clean_str(v) for k, v in template_dic.items() if k != template_text[i]]
        print('context is : \n %s \n\n template is : \n %s' % (Context, Utterance))
        if mode == 'train':
            # f.write('"%s","%s",%s\n'%(Context,Utterance,str(1)))
            row_positive = [Context, Utterance, str(1)]
            f.writerow(row_positive)
            negative_template = clean_str(random.choice(template_other))
            row_negative = [Context, negative_template, str(0)]
            # print ('negative sampled template is %s \n'%negative_template)
            f.writerow(row_negative)
            # f.write('"%s","%s",%s\n'%(Context,negative_template,str(0)))
        if mode in ['val', 'test']:
            assert len(template_collection) - 1 == len(template_other)
            # negative_templates = [clean_str(s) for s in random.sample(template_other)]
            row = [Context, Utterance]
            print(row)
            row.extend(template_other)
            f.writerow(row)

            # f.write('"%s","%s",%s\n'%(Context,Utterance,','.join(negative_template))

            # f.write()
    f.close()


# In[96]:

write_file_for_DualEncoder(texts_train, label_train, template_text, 'train')

# In[97]:

write_file_for_DualEncoder(texts_val, label_val, template_text, 'val')

# In[98]:

write_file_for_DualEncoder(texts_test, label_test, template_text, 'test')

