This is a neural conversation model that rank best response from a pool of candidate, for an incoming utterance or email.

You need pretrained word embedding for this repo:
Download Glove Embeddings into embeddings directory cd embeddings wget http://nlp.stanford.edu/data/glove.840B.300d.zip unzip glove.840B.300d.zip

You can use your own chat/email exchange corpus or use Ubuntu corpus for this repo, Download the dataset (train, test and dev): You can download it directly in format .pkl into dataset folder: These pkl files were generated using the utilities/prepare_dataset_from_csv.py script which separate the context and response from each of the train.csv, test.csv and dev.csv https://drive.google.com/file/d/1VjVzY0MqKj0b-q_KXnaHC49qCw3iDqEY/view?usp=sharing

Or you can download raw files in csv format from here https://github.com/rkadlec/ubuntu-ranking-dataset-creator and run the utilities/prepare_dataset_from_csv.py script by yourself. cd utilities python prepare_dataset_from_csv.py

To prepare training data, each incoming email is paired with the actual used response template with label 1; and paired with random sampled template from the candidate pool with label 0.  So the ratio of positive pair (label of 1) vs negative pair (label of 0 ) is 1:1.

To prepare testing data, each email is paired with ground truth response at position #1, and also paired with randomly sampled templates as distractors.

Recall @K with specific group size is used to measure the model performance. 

The neural network is approached as a dual-encoder strategy, similar as those listed in the reference. There are a few encoder is considered here:  CNN with a single filter size,  CNN with multiple filter size, LSTM, Transformer.

To run the model with training: python main.py --mode train
To evalute performance of pretrained model on testing data: python main.py --mode test


reference:
https://github.com/dennybritz/chatbot-retrieval/
https://github.com/basma-b/dual_encoder_udc
https://arxiv.org/pdf/1506.08909.pdf 
https://github.com/foamliu/Self-Attention-Keras

