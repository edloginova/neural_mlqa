#!/usr/bin/env python
# coding: utf-8


import codecs
import datetime
import gc
import pickle
import random
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from gensim.models.wrappers import FastText

np.random.seed(42)
random.seed(12345)
EMBEDDING_PATH = "../../resources/"
MODEL_PATH = "models/insuranceqa/"
K_TRAIN = 50
K_TEST = 100

model_name = "torch_insqav2_de"
torch.utils.backcompat.broadcast_warning.enabled = True

print("Torch version: ", torch.__version__)
print("Model: attention, dropout, no learning rate decay.")

# # Loading data


corpus_dir = "data/processed/insuranceqa-v2/de/"

data_a_file = "InsuranceQA.German.answers.inwords.decomp"
answers = pd.read_table(corpus_dir + data_a_file, header=None, names=['answer_id', 'answer'],
                        dtype={'answer_id': str, 'answer': str})
answers.head()

dfs_tmp = []
for option in ['train', 'test', 'valid']:
    file_tmp = corpus_dir + "InsuranceQA.German." + option + '.' + str(500) + ".inwords.decomp"
    df_tmp = pd.read_table(file_tmp, header=None, names=['domain', 'question', 'answer_ids', 'pool'])
    if option != 'valid':
        df_tmp['split_type'] = option
    else:
        df_tmp['split_type'] = 'dev'
    dfs_tmp.append(df_tmp)
split_tmp = pd.concat(dfs_tmp)

# handle multiple correct answers
split_tmp['answer_ids'] = split_tmp['answer_ids'].apply(lambda x: [i for i in x.split()])
lst_col = 'answer_ids'
split_tmp_expanded = pd.DataFrame({col: np.repeat(split_tmp[col].values, split_tmp[lst_col].str.len())
                                   for col in split_tmp.columns.difference([lst_col])
                                   }).assign(**{lst_col: np.concatenate(split_tmp[lst_col].values)})[
    split_tmp.columns.tolist()]
split_tmp_expanded.rename(columns={'answer_ids': 'answer_id'}, inplace='True')

data = split_tmp_expanded.merge(answers, on='answer_id', how='left')

for idx, row in data.iterrows():
    answer_ids = list(data[data.question == row['question']]['answer_id'].values)
    data.set_value(idx, 'answer_ids', answer_ids)

dev_data = data[data.split_type == 'dev']
train_data = data[data.split_type == 'train']


def code2word(code):
    return vocab[code]


def decode_text(text):
    decoded_tokens = [code2word(word) for word in text.split()]
    return ' '.join(decoded_tokens)


with codecs.open(corpus_dir + "vocabulary", encoding='utf8') as input_file:
    content = input_file.readlines()
vocab = {x.split()[0].lower(): x.split()[1] for x in content}
print(list(vocab.keys())[:10])
data['answer'] = data['answer'].apply(str.lower)
data['question'] = data['question'].apply(str.lower)
# data['answer'] = data['answer'].apply(decode_text)
# data['question'] = data['question'].apply(decode_text)
dev_data = data[data.split_type == 'dev']
train_data = data[data.split_type == 'train']
print(train_data.head())
params = pickle.load(open('INSQA_params', 'rb'))

print(params)
params['batch_size'] = 2
print("Loaded data")

# ## Embeddings & OOV


fasttext_embeddings = FastText.load_fasttext_format(EMBEDDING_PATH + 'cc.de.300.bin')
tokens = data.question.apply(str.split).values + data.answer.apply(str.split).values
tokens = [x for y in tokens for x in y]
tokens = set(tokens)
vocabulary_encoded = {k: i for i, k in enumerate(tokens)}
print('# tokens:', len(tokens))

EMBEDDING_DIM = 300
embedding_matrix = np.zeros((len(vocabulary_encoded) + 1, EMBEDDING_DIM))
oov_vector = np.random.rand(EMBEDDING_DIM)
oovs = []
for word, i in vocabulary_encoded.items():
    if word in fasttext_embeddings.wv.vocab:
        embedding_vector = fasttext_embeddings[word]
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = oov_vector
        oovs.append(word)
embedding_matrix = torch.FloatTensor(embedding_matrix)

print("Loaded embeddings")
print('# embedding_matrix:', embedding_matrix.shape)
print('# OOVs:', len(oovs))

# ## Randomizing pools

pool = set(train_data['answer_id'].values)
for idx, row in train_data.iterrows():
    real_pool = list(pool - set([str(x) for x in row['answer_ids']]))
    real_small_pool = random.sample(real_pool, K_TRAIN)
    train_data.set_value(idx, 'pool', ' '.join(real_small_pool))
pool = set(dev_data['answer_id'].values)
for idx, row in dev_data.iterrows():
    real_pool = list(pool - set([str(x) for x in row['answer_ids']]))
    real_random_pool = random.sample(real_pool, K_TEST)
    dev_data.set_value(idx, 'pool', ' '.join(real_random_pool))

print("Randomized pools")

# # Model

cuda_option = True
if torch.cuda.is_available():
    if not cuda_option:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


def transfer_data(x):
    if cuda_option:
        return x.cuda()
    else:
        return x


def print_used_gpu():
    sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8").split('\n')
    print(out_list[66].split()[2])


'''
1. Take list of sequences (texts) in the following format: [ [word1, ... wordN], [word1, ... wordM], ...] (list of lists)
2. Encode them using the vocabulary (words to numbers). 
3. Pad with #size_vocab up to max_len in this particular batch.
#size_vocab corresponds to an artificial padding token, which respective word embeddings consists of -10.
4. Sort by original (before padding) sequence lengths.
5. Look up word embeddings.
6. Use pack_padded_sequence function from PyTorch to create a required PyTorch representation to feed into LSTM
Documentation on pack_padded and pad_packed: http://pytorch.org/docs/master/_modules/torch/nn/utils/rnn.html
'''


def prepare_batch(texts, vocabulary_encoded, embeddings, max_len, volatile=False):
    vectorized_seqs = [[vocabulary_encoded[w] for w in text if w in vocabulary_encoded.keys()] for text in texts]
    seq_lengths = torch.LongTensor([len(x) for x in vectorized_seqs])
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())) + params["vocab_size"]
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        if seqlen < max_len:
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq[:seqlen])
        else:
            seq_tensor[idx, :max_len] = torch.LongTensor(seq[:max_len])
    seq_lengths[seq_lengths > max_len] = max_len
    seq_lengths_sorted, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor_sorted = seq_tensor[perm_idx]
    return seq_tensor_sorted, perm_idx, seq_lengths_sorted.numpy()


class QALSTM(nn.Module):
    def __init__(self, hidden_dim, embedding_size, vocab_size, cuda_option, embeddings):
        super(QALSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size + 1, embedding_size)
        self.embeddings.weight.data.copy_(embeddings)
        np.random.seed(1)
        torch.manual_seed(2)
        torch.cuda.manual_seed(2)
        torch.cuda.manual_seed_all(2)
        self.lstm = nn.LSTM(embedding_size, hidden_dim, bidirectional=True)  # (seq_len, batch, input_size) ->
        # (seq_len, batch, hidden_size * num_directions)
        self.cuda_option = cuda_option
        self.hidden = self.init_hidden()
        self.lin1 = nn.Linear(params['rnn_size'] * 2, params['rnn_size'] * 2, False)
        self.lin2 = nn.Linear(params['rnn_size'] * 2, params['rnn_size'] * 2, False)
        self.lin3 = nn.Linear(params['rnn_size'] * 2, 1, False)
        self.tahn = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (
            autograd.Variable(transfer_data(torch.zeros(2, params['batch_size'], self.hidden_dim)), requires_grad=True),
            autograd.Variable(transfer_data(torch.zeros(2, params['batch_size'], self.hidden_dim)), requires_grad=True))

    def forward(self, sentence, lens, perm_idx, is_eval=False, volatile=False, attention=(False, None),
                wrong_answer_mode=False):
        # unpack output, transfer it to GPU and pack again (not done in batchify() to save memory)
        #         attention = (False, None)
        verbose = False
        if volatile:
            with torch.no_grad():
                sentence = self.embeddings(autograd.Variable(transfer_data(sentence)).long())
        else:
            sentence = self.embeddings(autograd.Variable(transfer_data(sentence)).long())
        sentence = sentence.transpose(0, 1)
        packed_input = nn.utils.rnn.pack_padded_sequence(sentence, lens)

        # apply QALSTM
        if not is_eval:
            packed_output, self.hidden = self.lstm(packed_input, self.hidden)
        else:
            packed_output, _ = self.lstm(packed_input)
        del packed_input

        # unpack output and transpose it to be batch_size x rnn_size*2 x max_len
        unpacked_output, lengths = nn.utils.rnn.pad_packed_sequence(packed_output, padding_value=-10.0)
        unpacked_output = torch.transpose(torch.transpose(unpacked_output, 0, 2), 0, 1)
        if verbose:
            print('unpacked_output: ', unpacked_output)
        del packed_output
        if attention[0]:

            # mask padding for LSTM output
            mask = autograd.Variable(torch.ones(unpacked_output.size())).cuda()
            for i, l in enumerate(lengths):
                if l < unpacked_output.size(2):
                    mask[i, :, l:] = 0
            if verbose:
                print('mask: ', mask)
            # apply W_{am} and W_{qm}
            a_lin = self.lin1(torch.transpose(unpacked_output * mask, 1, 2))
            if verbose:
                print('a_lin: ', a_lin)
            q_lin = self.lin2(torch.transpose(attention[1].repeat(1, 1, a_lin.size(1)), 1, 2))
            if verbose:
                print('q_lin: ', q_lin)
            # obtain m_{a, q}(t)
            if wrong_answer_mode:
                m = q_lin.repeat(a_lin.size(0), 1, 1) + a_lin
            else:
                m = q_lin + a_lin
            # if wrong_answer_mode:
            m = m * torch.transpose(mask, 1, 2)
            if verbose:
                print('m: ', m)
            attentions = self.lin3(self.tahn(m))
            if verbose:
                print('attentions: ', attentions)

            mask_ = (mask[:, 0:1, :] == 0)
            mask_ = torch.transpose(mask_, -1, -2)

            padded_attention = attentions.clone()

            padded_attention.masked_fill_(mask_, -float('inf'))

            softmax_attentions = model.softmax(padded_attention)
            # change padding value from 0 to -10 again to correctly maxpool
            unpacked_output = unpacked_output * torch.transpose(softmax_attentions.repeat(1, 1, 2 * params['rnn_size']),
                                                                1, 2)
            unpacked_output.masked_fill_((mask == 0), -10.0)

        # restore original order
        if self.cuda_option:
            perm_idx = perm_idx.cuda()
            unpacked_output = unpacked_output[perm_idx, :, :]
            perm_idx = perm_idx.cpu()
        else:
            unpacked_output = unpacked_output[perm_idx, :, :]

        # maxpool
        result = self.dropout(unpacked_output)
        result, _ = unpacked_output.max(2, keepdim=True)
        del unpacked_output

        if verbose:
            print('result: ', result)
        return result


def check_accuracy(model, all_questions, all_single_correct_answers_texts, all_wrong_answer_texts, batch_size=2,
                   verbose=False):
    begin_time = datetime.datetime.now()
    volatile = True
    len_data = len(all_questions)
    num_batches = int(np.ceil(len_data / batch_size))

    num_correct = 0
    num_instances = 0
    print("num_batches: ", num_batches)

    for batch_id in range(0, num_batches):
        wa_batches = [None] * 1

        if (batch_id + 1) * batch_size < len_data:
            questions = all_questions[(batch_id) * batch_size:(batch_id + 1) * batch_size]
            answers = all_single_correct_answers_texts[(batch_id) * batch_size:(batch_id + 1) * batch_size]
            wrong_answer_texts = all_wrong_answer_texts[(batch_id) * batch_size:(batch_id + 1) * batch_size]
        else:
            questions = all_questions[batch_id * batch_size:]
            answers = all_single_correct_answers_texts[batch_id * batch_size:]
            wrong_answer_texts = all_wrong_answer_texts[batch_id * batch_size:]

        unique_questions = {}
        questions_tmp = [' '.join(x) for x in questions]
        for uq in list(set(questions_tmp)):
            unique_questions[uq] = []
        for num, uq in enumerate(questions_tmp):
            unique_questions[uq] = unique_questions[uq] + [num]

        questions_input_packed, perm_idx_q, lens_q = prepare_batch(questions, vocabulary_encoded, model.embeddings,
                                                                   params['max_len'], volatile)
        _, perm_idx_q = perm_idx_q.sort(0)

        answers_input_packed, perm_idx_a, lens_a = prepare_batch(answers, vocabulary_encoded, model.embeddings,
                                                                 params['max_len'], volatile)
        _, perm_idx_a = perm_idx_a.sort(0)

        wa_batches[0] = [None] * len(questions)
        for i, wa in enumerate(wrong_answer_texts):
            wa_input_packed, perm_idx_wa, seq_lengths_sorted = prepare_batch(wa, vocabulary_encoded, model.embeddings,
                                                                             params['max_len'], volatile)
            _, perm_idx_wa = perm_idx_wa.sort(0)
            wa_batches[0][i] = (wa_input_packed, perm_idx_wa, seq_lengths_sorted)

        questions_output = transfer_data(model(questions_input_packed, lens_q, perm_idx_q, True, True))
        answers_output = transfer_data(
            model(answers_input_packed, lens_a, perm_idx_a, True, True, (True, questions_output)))

        wa_outputs_all = [None] * len(wrong_answer_texts)
        for i, wa in enumerate(wrong_answer_texts):
            (wa_input_packed, perm_idx_wa, lens_wa) = wa_batches[0][i]
            wa_outputs_all[i] = model(wa_input_packed, lens_wa, perm_idx_wa, True, True,
                                      (True, questions_output[i, :, :]), True)

        scores_cos_q_a = cos_qa(questions_output, answers_output)
        with torch.no_grad():
            scores_max_cos_q_wa = autograd.Variable(torch.zeros(len(wa_batches[0])))
        for ind, wa_output in enumerate(wa_outputs_all):
            scores_max_cos_q_wa[ind], idx_max = torch.max(
                cos_qa(torch.t(questions_output[ind, :, :].repeat(1, wa_output.size(0))), wa_output[:, :, 0]), 0)

        scores_max_cos_q_wa = transfer_data(scores_max_cos_q_wa)
        if verbose:
            print('scores_cos_q_a:\n', scores_cos_q_a)
            print('scores_max_cos_q_wa:\n', scores_max_cos_q_wa)
        binary_scores = scores_cos_q_a.squeeze(-1) > scores_max_cos_q_wa
        for key, val in unique_questions.items():
            for elem in val:
                if binary_scores[elem] == 1:
                    num_correct += 1
                    break
                    #         num_correct += torch.sum().data.cpu().numpy()[0]
        num_instances += len(unique_questions)
        scores_max_cos_q_wa = scores_max_cos_q_wa.cpu()

        del scores_cos_q_a, scores_max_cos_q_wa
        del questions_output, answers_output, wa_outputs_all

        print("\rBatch %d / %d. Time since beginning: %s" % (
            batch_id + 1, num_batches, str(datetime.datetime.now() - begin_time)), end='')

    accuracy = num_correct / num_instances
    print("\nFinal accuracy: %0.4f" % accuracy)

    finish_time = datetime.datetime.now() - begin_time
    print(finish_time)

    return accuracy


def save_checkpoint(state, filename=MODEL_PATH + model_name):
    torch.save(state, filename)


# # Preparing data


answer_texts = data[['answer_id', 'answer']]
answer_texts = answer_texts.set_index('answer_id')
answer_texts = answer_texts[~answer_texts.index.duplicated(keep='first')]
answer_texts['answer'] = answer_texts['answer'].apply(str.split)

all_questions_dev = dev_data['question'].apply(str.split).values
all_answers_dev = dev_data['answer'].apply(str.split).values
all_multiple_correct_answers_dev = dev_data['answer_ids'].values
all_single_correct_answers_dev = dev_data['answer_id'].values
all_wrong_answer_pools_dev = dev_data['pool'].apply(str.split)
all_wrong_answer_pools_dev = [[x for x in indices if x not in all_multiple_correct_answers_dev[ind]] for ind, indices in
                              enumerate(all_wrong_answer_pools_dev)]
all_wrong_answer_texts_dev = [answer_texts.loc[wa_pool]['answer'].values for wa_pool in all_wrong_answer_pools_dev]
all_single_correct_answers_texts_dev = [answer_texts.loc[ca_ids]['answer'] for ca_ids in all_single_correct_answers_dev]

print(dev_data.head())
test_data = data[data['split_type'] == 'test']
pool = set(test_data['answer_id'].values)
for idx, row in test_data.iterrows():
    real_pool = list(pool - set([str(x) for x in row['answer_ids']]))
    real_random_pool = random.sample(real_pool, 100)
    test_data.set_value(idx, 'pool', ' '.join(real_random_pool))
all_questions_test = test_data['question'].apply(str.split).values
all_answers_test = test_data['answer'].apply(str.split).values
all_multiple_correct_answers_test = test_data['answer_ids'].values
all_single_correct_answers_test = test_data['answer_id'].values
all_wrong_answer_pools_test = test_data['pool'].apply(str.split)
all_wrong_answer_pools_test = [[x for x in indices if x not in all_multiple_correct_answers_test[ind]] for ind, indices
                               in enumerate(all_wrong_answer_pools_test)]
all_wrong_answer_texts_test = [answer_texts.loc[wa_pool]['answer'].values for wa_pool in all_wrong_answer_pools_test]
all_single_correct_answers_texts_test = [answer_texts.loc[ca_ids]['answer'] for ca_ids in
                                         all_single_correct_answers_test]

print("Prepared data")

# # Training


params['vocab_size'] = len(vocabulary_encoded)
params['embedding_size'] = EMBEDDING_DIM
model = QALSTM(params['rnn_size'], params['embedding_size'], params['vocab_size'], cuda_option, embedding_matrix)
if cuda_option:
    model.cuda()
else:
    model.cpu()
np.random.seed(1)
torch.manual_seed(2)
torch.cuda.manual_seed(2)
torch.cuda.manual_seed_all(2)
cos_qa = nn.CosineSimilarity(dim=1, eps=1e-8)
loss_function = nn.MarginRankingLoss(margin=params['margin'], size_average=False)
optimizer = optim.SGD(model.parameters(), lr=params['lr'])
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience = 2)

print("Created model")
print(model)

# print("Checking accuracy on validation before training:")
# model.eval()
# acc = check_accuracy(model, all_questions_dev, all_single_correct_answers_texts_dev, all_wrong_answer_texts_dev, 2)

batch_size = params['batch_size']
len_data = len(train_data)
num_batches = int(np.ceil(len_data / batch_size))
all_questions = train_data['question'].apply(str.split).values
all_answers = train_data['answer'].apply(str.split).values
all_multiple_correct_answers = train_data['answer_ids'].values
all_single_correct_answers = train_data['answer_id'].values
all_wrong_answer_pools = train_data['pool'].apply(str.split)
all_wrong_answer_pools = [[x for x in indices if x not in all_multiple_correct_answers[ind]][:params['anspool']] for
                          ind, indices in enumerate(all_wrong_answer_pools)]
all_wrong_answer_texts = [answer_texts.loc[wa_pool]['answer'].values for wa_pool in all_wrong_answer_pools]
all_single_correct_answers_texts = [answer_texts.loc[ca_ids]['answer'] for ca_ids in all_single_correct_answers]

train_loss = [-1] * params['max_epoch']
validation_accuracies = [-1] * params['max_epoch']
a_attention_weights_stat = [None] * params['max_epoch']
wa_attention_weights_stat = [None] * params['max_epoch']
cur_max_acc = 0
begin_time = datetime.datetime.now()
finish_time = begin_time
wa_batches = [None] * num_batches
volatile = False
print_used_gpu()
for epoch in range(0, params['max_epoch']):
    print("Epoch %d/%d" % (epoch + 1, params['max_epoch']))
    batch_loss_value = 0
    for batch_id in range(0, num_batches - 1):

        if (batch_id + 1) * batch_size < len_data:
            questions = all_questions[(batch_id) * batch_size:(batch_id + 1) * batch_size]
            answers = all_single_correct_answers_texts[(batch_id) * batch_size:(batch_id + 1) * batch_size]
            wrong_answer_texts = all_wrong_answer_texts[(batch_id) * batch_size:(batch_id + 1) * batch_size]
        else:
            questions = all_questions[(batch_id) * batch_size:]
            answers = all_single_correct_answers_texts[(batch_id) * batch_size:]
            wrong_answer_texts = all_wrong_answer_texts[(batch_id) * batch_size:]

        questions_input_packed, perm_idx_q, lens_q = prepare_batch(questions, vocabulary_encoded, model.embeddings,
                                                                   params['max_len'], volatile)
        _, perm_idx_q = perm_idx_q.sort(0)

        answers_input_packed, perm_idx_a, lens_a = prepare_batch(answers, vocabulary_encoded, model.embeddings,
                                                                 params['max_len'], volatile)
        _, perm_idx_a = perm_idx_a.sort(0)

        wa_batches[batch_id] = [None] * len(questions)
        for i, wa in enumerate(wrong_answer_texts):
            wa_input_packed, perm_idx_wa, seq_lengths_sorted = prepare_batch(wa, vocabulary_encoded, model.embeddings,
                                                                             params['max_len'], volatile)
            _, perm_idx_wa = perm_idx_wa.sort(0)
            wa_batches[batch_id][i] = (wa_input_packed, perm_idx_wa, seq_lengths_sorted)

        # calculate output for all wrong answers and find the most similar one for each question
        model.eval()
        questions_output = transfer_data(model(questions_input_packed, lens_q, perm_idx_q, True, False))
        wa_outputs_all = torch.zeros(params['batch_size'], params['anspool'], 2 * params['rnn_size'])
        for i, wa in enumerate(wa_batches[batch_id]):
            (wa_input_packed, perm_idx_wa, lens_wa) = wa_batches[batch_id][i]
            wa_outputs_all[i, :, :] = model(wa_input_packed, lens_wa, perm_idx_wa, True, False,
                                            (True, questions_output[i, :, :]), True).data.squeeze(-1)
        with torch.no_grad():
            wa_outputs_all = autograd.Variable(transfer_data(wa_outputs_all))
        idx_maxs = [None] * len(wa_outputs_all)
        with torch.no_grad():
            scores_max_cos_q_wa_before = autograd.Variable(torch.zeros(len(wa_batches[batch_id])))
        for i in range(0, params['batch_size']):
            scores_cos_q_wa = cos_qa(torch.t(questions_output[i, :, :].repeat(1, params['anspool'])),
                                     wa_outputs_all[i, :, :])
            scores_max_cos_q_wa_before[i], idx_max = torch.max(scores_cos_q_wa, 0)
            idx_maxs[i] = idx_max.item()
        del scores_max_cos_q_wa_before

        max_wa = []
        for i, idx_max in enumerate(idx_maxs):
            max_wa.append(wrong_answer_texts[i][idx_max])
        max_wa = np.array(max_wa)
        wa_input_packed, perm_idx_wa, lens_wa = prepare_batch(max_wa, vocabulary_encoded, model.embeddings,
                                                              params['max_len'], False)
        _, perm_idx_wa = perm_idx_wa.sort(0)
        # calculate output for questions, correct answers and most similar wrong answers in training mode
        model.train()
        model.zero_grad()

        #         print('Q - TRAIN')
        # init_hidden() creates new initial states for new sequences
        model.hidden = model.init_hidden()
        questions_output = transfer_data(model(questions_input_packed, lens_q, perm_idx_q))
        model.hidden = model.init_hidden()
        #         print('A - TRAIN')
        answers_output = model(answers_input_packed, lens_a, perm_idx_a, False, False, (True, questions_output))
        answers_output = transfer_data(answers_output)
        #         print('WA - TRAIN')
        model.hidden = model.init_hidden()
        wrong_answers_output = transfer_data(
            model(wa_input_packed, lens_wa, perm_idx_wa, False, False, (True, questions_output), False))

        # calculate cosine similarities
        scores_cos_q_a = cos_qa(questions_output, answers_output).squeeze(-1)
        scores_max_cos_q_wa = cos_qa(questions_output, wrong_answers_output).squeeze(-1)
        # calculate loss
        loss = loss_function(scores_cos_q_a, scores_max_cos_q_wa,
                             autograd.Variable(transfer_data(torch.ones(params["batch_size"])), requires_grad=False))

        # update parameters
        loss.backward()
        optimizer.step()
        batch_loss_value += loss.data.cpu().numpy()

        print("\rBatch %d / %d. Time: %s. ETA: %s" % (batch_id, num_batches, str(datetime.datetime.now() - begin_time),
                                                      str(((datetime.datetime.now() - begin_time) * (
                                                              num_batches / (batch_id + 1)) * params['max_epoch']))),
              end='')

        gc.collect()

        del questions_input_packed, questions_output, lens_q, perm_idx_q
        del answers_input_packed, answers_output, lens_a, perm_idx_a
        del wa_outputs_all, wa_input_packed, lens_wa, perm_idx_wa
        del idx_maxs
        del wrong_answers_output, scores_cos_q_a, scores_max_cos_q_wa
    print('\n Average batch loss: ', batch_loss_value / num_batches)
    train_loss[epoch] = batch_loss_value / num_batches
    gc.collect()
    finish_time = datetime.datetime.now() - begin_time
    print(finish_time)
    model.eval()
    print("Checking accuracy on validation:")
    acc = check_accuracy(model, all_questions_dev, all_single_correct_answers_texts_dev, all_wrong_answer_texts_dev, 2)
    #     scheduler.step(acc)
    validation_accuracies[epoch] = acc
    if acc > cur_max_acc:
        cur_max_acc = acc
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'acc': acc,
            'optimizer': optimizer.state_dict(),
        })
        print('Current maximum accuracy updated: ', cur_max_acc)
        print('Current best model saved.')

    print("Checking accuracy on train:")
    acc = check_accuracy(model, all_questions, all_single_correct_answers_texts, all_wrong_answer_texts, 2)

    print("Checking accuracy on test:")
    check_accuracy(model, all_questions_test, all_single_correct_answers_texts_test, all_wrong_answer_texts_test, 2)
    if validation_accuracies[-1] < validation_accuracies[-2] and validation_accuracies[-2] < validation_accuracies[-3]:
        print("Early stopping.")
        break

    print('----------------------------------')

checkpoint = torch.load(MODEL_PATH + model_name)
start_epoch = checkpoint['epoch']
best_acc = checkpoint['acc']
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

print('Validation best accuracy:', best_acc)
model.eval()
acc = check_accuracy(model, all_questions_dev, all_single_correct_answers_texts_dev, all_wrong_answer_texts_dev, 2)

print("Checking accuracy on test:")
check_accuracy(model, all_questions_test, all_single_correct_answers_texts_test, all_wrong_answer_texts_test, 2)
