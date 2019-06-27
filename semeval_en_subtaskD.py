#!/usr/bin/env python
# coding: utf-8
import os
import random as rn
import subprocess
import keras
import tensorflow as tf
import pickle
from gensim.models.wrappers import FastText
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import class_weight
from loading_preprocessing_TC import *
from ast import literal_eval
import re

## Setup

# Session settings
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.per_process_gpu_memory_fraction = 0.2
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# Global settings

EMBEDDING_DIM = 300
EMBEDDING_PATH = "../../resources/"
DATA_PATH = "data/processed/semeval-subtaskD/"
EXTRA_DATA_PATH = "data/external/"
SCORER_PATH = "scorer/"
PREDICTION_PATH = "predictions/"
MODEL_PATH = "models/semeval/"
SEED = 42
MAX_LENGTH = 200
epochs = 20
batch_size = 20
model_name = 'keras_semeval_en_subtaskD'


def clean_text(text):
    clean_text = re.sub('(\d+)', r' \1 ', text)
    clean_text = re.sub('\s{2,}', ' ', clean_text)
    clean_text = re.sub(r'[^\w\s]', '', clean_text)
    return clean_text

## Loading data

train = pd.read_csv(DATA_PATH + 'train-expanded_NoLing.csv')
validation = pd.read_csv(DATA_PATH + 'dev_expanded_NoLing.csv')
answer_texts_train = pd.read_csv(DATA_PATH + 'answer_texts_train_NoLing.csv')
answer_texts_validation = pd.read_csv(DATA_PATH + 'answer_texts_dev_NoLing.csv')
test = pd.read_csv(DATA_PATH + 'test_NoLabels_NoLing.csv')
answer_texts_test = pd.read_csv(DATA_PATH + 'answer_texts_test_NoLing.csv')
translations = pickle.load(open(EXTRA_DATA_PATH + 'arabic_english_translations.p', 'rb'))
embeddings_index = FastText.load_fasttext_format(EMBEDDING_PATH + 'cc.en.300.bin')

## Preparing data

train['pool'] = train['pool'].apply(literal_eval)
train['answer_ids'] = train['answer_ids'].apply(literal_eval)
validation['pool'] = validation['pool'].apply(literal_eval)
validation['answer_ids'] = validation['answer_ids'].apply(literal_eval)
answer_texts_test.set_index('answer_id', drop=False, inplace=True)
test['candidates'] = test['candidates'].apply(literal_eval)
test.set_index('Unnamed: 0', drop=True, inplace=True)
for idx, row in test.iterrows():
    test.set_value(idx, 'candidates', sorted([int(x) for x in set(row['candidates'])]))
answer_texts_test['clean_answer'] = answer_texts_test['answer'].apply(clean_text)
test['clean_question'] = test['question'].apply(clean_text)
answer_texts_train.set_index('answer_id', drop=False, inplace=True)
answer_texts_validation.set_index('answer_id', drop=False, inplace=True)

train['question'] = [translations[x] for x in train['question'].values]
train['answer'] = [translations[x] for x in train['answer'].values]
validation['question'] = [translations[x] for x in validation['question'].values]
validation['answer'] = [translations[x] for x in validation['answer'].values]
test['question'] = [translations[x] for x in test['question'].values]
answer_texts_train['answer'] = [translations[x] for x in answer_texts_train['answer'].values]
answer_texts_validation['answer'] = [translations[x] for x in answer_texts_validation['answer'].values]
answer_texts_test['answer'] = [translations[x] for x in answer_texts_test['answer'].values]

texts = list([x for x in train['clean_question'].values]) + list([x for x in answer_texts_train['clean_answer'].values])
tokenizer = Tokenizer(oov_token='#OOV#')
tokenizer.fit_on_texts(texts)
vocabulary = tokenizer.word_index
vocabulary_inv = {v: k for k, v in vocabulary.items()}

embedding_matrix = np.zeros((len(vocabulary) + 1, EMBEDDING_DIM))
oov_vector = np.random.rand(EMBEDDING_DIM)
oovs_num = 0
for word, i in vocabulary.items():
    if word in embeddings_index.wv.vocab:
        embedding_vector = embeddings_index[word]
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = oov_vector
        oovs_num += 1
embedding_matrix[vocabulary['#OOV#']] = oov_vector

answer_texts = pd.concat([answer_texts_validation, answer_texts_train])
answer_texts = answer_texts[~answer_texts.index.duplicated(keep='first')]

questions = []
wrong_answers = []
for idx, row in train.iterrows():
    for y in row['pool']:
        wrong_answers.append(answer_texts.loc[int(y)]['clean_answer'])
        questions.append(row['clean_question'])
correct_answers = []
for idx, row in train.iterrows():
    correct_answers.append(row['clean_answer'])
    questions.append(row['clean_question'])

data = [(x, 0) for x in wrong_answers] + [(x, 1) for x in correct_answers]

data_answers = [x[0] for x in data]
data_questions = questions
data_targets = [x[1] for x in data]

questions_validation = []
wrong_answers_validation = []
for idx, row in validation.iterrows():
    for y in row['pool']:
        wrong_answers_validation.append(answer_texts.loc[int(y)]['clean_answer'])
        questions_validation.append(row['clean_question'])
correct_answers_validation = []
for idx, row in validation.iterrows():
    correct_answers_validation.append(row['clean_answer'])
    questions_validation.append(row['clean_question'])

data_validation = [(x, 0) for x in wrong_answers_validation] + [(x, 1) for x in correct_answers_validation]

data_answers_validation = [x[0] for x in data_validation]
data_questions_validation = questions_validation
data_targets_validation = [x[1] for x in data_validation]

X_train_a = tokenizer.texts_to_sequences(data_answers)
X_train_q = tokenizer.texts_to_sequences(data_questions)

X_train_a = pad_sequences(X_train_a, maxlen=MAX_LENGTH, value=0.0)
X_train_q = pad_sequences(X_train_q, maxlen=MAX_LENGTH, value=0.0)

Y_train = np.array(data_targets)

X_validation_a = tokenizer.texts_to_sequences(data_answers_validation)
X_validation_q = tokenizer.texts_to_sequences(data_questions_validation)

X_validation_a = pad_sequences(X_validation_a, maxlen=MAX_LENGTH, value=0.0)
X_validation_q = pad_sequences(X_validation_q, maxlen=MAX_LENGTH, value=0.0)

Y_validation = np.array(data_targets_validation)


## Training

class Metrics(Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.last_val_f1 = 1

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict([self.validation_data[0], self.validation_data[1]]))).round()
        val_targ = self.validation_data[2]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.last_val_f1 = _val_f1
        print("— val_f1: %f — val_precision: %f — val_recall %f" % (_val_f1, _val_precision, _val_recall))
        return


class EarlyStopByF1(keras.callbacks.Callback):
    def __init__(self, value=0, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        predict = (np.asarray(self.model.predict([self.validation_data[0], self.validation_data[1]]))).round()
        target = self.validation_data[2]
        score = f1_score(target, predict)
        if score <= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping Threshold" % epoch)
            self.model.stop_training = True
        else:
            self.value = score


def get_model(embedding_matrix, M=96, N=64, gaussian_noise=0, unidirectional=False,
              trainable_embeddings=False, mean_pooling=False, initializer=keras.initializers.he_normal(seed=SEED),
              dropout=False):
    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=MAX_LENGTH, trainable=trainable_embeddings)

    a_input = Input(shape=(MAX_LENGTH,), dtype='int32')
    q_input = Input(shape=(MAX_LENGTH,), dtype='int32')

    embedded_a = embedding_layer(a_input)
    embedded_q = embedding_layer(q_input)

    if gaussian_noise != 0:
        embedded_a = keras.layers.GaussianNoise(gaussian_noise)(embedded_a)
        embedded_q = keras.layers.GaussianNoise(gaussian_noise)(embedded_q)

    if unidirectional:
        if dropout:
            shared_lstm = keras.layers.LSTM(M, return_sequences=True, recurrent_dropout=0.2, dropout=0.5,
                                            kernel_initializer=initializer)
            shared_lstm2 = keras.layers.LSTM(N, return_sequences=True, recurrent_dropout=0.2, dropout=0.5,
                                             kernel_initializer=initializer)
        else:
            shared_lstm = keras.layers.CuDNNLSTM(M, return_sequences=True, kernel_initializer=initializer)
            shared_lstm2 = keras.layers.CuDNNLSTM(N, return_sequences=True, kernel_initializer=initializer)
        N_output = N
    else:
        if dropout:
            shared_lstm = Bidirectional(keras.layers.LSTM(M, return_sequences=True, recurrent_dropout=0.2, dropout=0.5,
                                                          kernel_initializer=initializer))
            shared_lstm2 = Bidirectional(keras.layers.LSTM(N, return_sequences=True, recurrent_dropout=0.2, dropout=0.5,
                                                           kernel_initializer=initializer))
        else:
            shared_lstm = Bidirectional(
                keras.layers.CuDNNLSTM(M, return_sequences=True, kernel_initializer=initializer))
            shared_lstm2 = Bidirectional(
                keras.layers.CuDNNLSTM(N, return_sequences=True, kernel_initializer=initializer))
        N_output = 2 * N

    a_lstm_intermediate = shared_lstm(embedded_a)
    a_lstm_intermediate = keras.layers.BatchNormalization()(a_lstm_intermediate)
    a_lstm_output = shared_lstm2(a_lstm_intermediate)

    q_lstm_intermediate = shared_lstm(embedded_q)
    q_lstm_intermediate = keras.layers.BatchNormalization()(q_lstm_intermediate)
    q_lstm_output = shared_lstm2(q_lstm_intermediate)

    O_q = GlobalMaxPooling1D(name='max_pool_q')(q_lstm_output)
    q_vec = Dense(N_output, name='W_qm')(O_q)
    q_vec = RepeatVector(200)(q_vec)

    a_vec = TimeDistributed(Dense(N_output, name='W_am'))(a_lstm_output)

    m = Add()([q_vec, a_vec])
    m = Activation(activation='tanh')(m)

    s = TimeDistributed(Dense(N_output, name='w_ms'))(m)
    s = keras.layers.Softmax(axis=1, name='attention_scores')(s)
    h_hat_a = Multiply(name='attended_a')([a_lstm_output, s])

    O_a = GlobalMaxPooling1D(name='max_pool_attended_a')(h_hat_a)
    x = Dot(axes=-1, normalize=True)([O_q, O_a])

    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(12345)
    tf.set_random_seed(1234)

    model = Model([a_input, q_input], x)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])

    model.summary()
    return model


print('Started training.')
adam = keras.optimizers.Adam(clipnorm=1.)
class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
class_weights = {i: x for i, x in enumerate(list(class_weights))}
print('Class weights:', class_weights)
metrics = Metrics()
model = get_model(embedding_matrix, M=96, N=64, gaussian_noise=0, unidirectional=False,
                  trainable_embeddings=False, mean_pooling=False, initializer=keras.initializers.he_normal(seed=SEED),
                  dropout=False)
checkpoint = ModelCheckpoint(MODEL_PATH + model_name + ".h5", monitor='val_loss', verbose=2, save_best_only=True,
                             mode='auto')
early_stopping = EarlyStopByF1()
model.fit([X_train_a, X_train_q], Y_train, validation_data=([X_validation_a, X_validation_q], Y_validation),
          epochs=epochs, batch_size=batch_size, callbacks=[checkpoint, early_stopping, metrics],
          class_weight=class_weights)
print('Finished training.')


## Evaluation

def evaluate(model, pred_mode, padding='pre'):
    threshold = 0.5
    pred_filename = PREDICTION_PATH + pred_mode + '_' + model_name + '.pred'

    if pred_mode == 'test':
        dataset = test
        answer_texts = answer_texts_test
        gold_labels = SCORER_PATH + '/SemEval2017-Task3-CQA-MD-test.xml.subtaskD.relevancy'

    pred = []
    for idx, row in dataset.iterrows():
        answers = []
        answer_ids = []
        for c in row['candidates']:
            answers.append(answer_texts.loc[c]['clean_answer'])
            answer_ids.append(c)
        x_test_batch_a = tokenizer.texts_to_sequences(answers)
        x_test_batch_a = pad_sequences(x_test_batch_a, maxlen=MAX_LENGTH, value=0.0, padding=padding)
        x_test_batch_q = tokenizer.texts_to_sequences([row['clean_question']] * len(answers))
        x_test_batch_q = pad_sequences(x_test_batch_q, maxlen=MAX_LENGTH, value=0.0, padding=padding)
        scores = list(model.predict([x_test_batch_a, x_test_batch_q]).flatten())
        for i in range(0, len(answers)):
            pred.append((idx, answer_ids[i], scores[i]))

    with open(pred_filename, 'w') as f:
        for val in pred:
            if np.round(val[2]) == 1:
                label = 'true'
            else:
                label = 'false'
            f.write('\t'.join([str(val[0]), str(val[1]), '0', str(val[2]), label]) + "\n")

    # subprocess.run(['d:'], shell=True)
    # subprocess.run(['cd', 'D:/Documents/dfki/MLQA'], shell=True)
    PYTHON2_PATH = 'C:/Users/Kate/.conda/envs/py2/python.exe'
    command = ' '.join(
        [PYTHON2_PATH, SCORER_PATH + '/MAP_scripts/ev.py', gold_labels, pred_filename])
    print(command)

    result = subprocess.run([command], shell=True, stdout=subprocess.PIPE)
    for x in result.stdout.decode('utf-8').split('\n'):
        print(x)
    return result


print('Evaluating on test from SemEval 2017.')
evaluate(model, 'test')
