import re
import string
import time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer

stemmer = EnglishStemmer()
nlp = spacy.load('en')
eng_stopwords = set(stopwords.words("english"))
num_str = [str(i) for i in range(0, 9)]

emoticons = {":-)": "happy", ":)": "happy", ":-]"":]": "happy", ":-3": "happy", ":3": "happy", ":->": "happy",
             ":>": "happy", \
             "8-)": "happy", "8)": "happy", ":-}": "happy", ":}": "happy", ":o)": "happy", ":c)": "happy",
             ":^)": "happy", \
             "=]": "happy", "=)": "happy", ":-D": "happy", ":D": "laugh", "8-D": "laugh", "8D": "laugh", "x-D": "laugh", \
             "xD": "laugh", "X-D": "laugh", "XD": "laugh", "=D": "laugh", "=3": "happy", "B^D": "laugh", ":-(": "sad", \
             ":(": "sad", ":-c": "sad", ":c": "sad", ":-<": "sad", ":<": "sad", ":-[": "sad", ":[": "sad",
             ":-||": "sad", \
             ">:[": "angry", ":{": "sad", ":@": "sad", ">:(": "angry", ";-)": "wink", ";)": "wink", "*-)": "wink", \
             "*)": "wink", ";-]": "wink", ";]": "wink", ";^)": "wink", ":-,": "wink", ";D": "laugh", \
             ":-/": "scepticism", ":/": "scepticism", ":-.": "scepticism", ">:\\": "angry", ">:/": "angry", \
             ":\\": "scepticism", "=/": "scepticism", "=\\": "scepticism", ":L": "scepticism", "=L": "scepticism", \
             ":S": "scepticism"}
emoticons_re = {}
for key, val in emoticons.items():
    new_key = key
    for c in new_key:
        if c in ['[', '\\', '^', '$', '.', '|', '?', '*', '+', '(', ')']:
            new_key = new_key.replace(c, "\\" + c)
        new_key = new_key.replace("\\\|", "\\|")
    regex = re.compile(new_key + "+")
    emoticons_re[regex] = val


class Text():
    text_id = -1
    text_type = ''
    text = ''
    clean_text = ''
    heavy_clean_text = ''
    spellchecked_text = ''
    placeholders_text = ''
    named_entities = []
    pos_tags = []
    lemmata = []
    stems = []
    tokens = []
    clean_tokens = []
    heavy_clean_tokens = []
    placeholders_tokens = []
    spellchecked_tokens = []
    doc = None
    punct = re.compile("(\.){2,}|(\?){2,}|(,){2,}|(-){2,}|(\"){2,}|(\$){2,}|(\*){2,}|(\'){2,}|(!){2,}")
    tags2words = {'GPE': 'country', 'ORDINAL': 'number', 'LAW': 'law', 'CARDINAL': 'number',
                  'LOC': 'location', 'EVENT': 'event', 'DATE': 'date', 'QUANTITY': 'quantity', 'NOT_NE': 'None',
                  'PERCENT': 'percent', 'PRODUCT': 'product', 'MONEY': 'money', 'FAC': 'facility',
                  'NORP': 'nationality',
                  'TIME': 'time', 'WORK_OF_ART': 'art', 'PERSON': 'person',
                  'LANGUAGE': 'language', 'ORG': 'organization'}

    def __init__(self, text: str, text_type: str, text_id: str):
        self.text = text
        self.text_type = text_type
        self.text_id = text_id

    def tokenize(self):
        if self.doc is None:
            self.doc = nlp(self.text)
        self.tokens = [str(token.text) for token in self.doc]
        return self.tokens

    def lemmatize(self):
        if self.doc is None:
            self.doc = nlp(self.text)
        self.lemmata = [str(token.lemma_) for token in self.doc]
        return self.lemmata

    def pos_tag(self):
        if self.doc is None:
            self.doc = nlp(self.text)
        self.pos_tags = [str(token.pos_) for token in self.doc]
        return self.pos_tags

    def stemmatize(self):
        if self.doc is None:
            self.doc = nlp(self.text)
        self.stems = [stemmer.stem(token.text) for token in self.doc]
        return self.stems

    def ner(self):
        if self.doc is None:
            self.doc = nlp(self.text)
        ne_texts = [ent.text for ent in self.doc.ents]
        ne = [(str(ent.text), str(ent.label_)) for ent in self.doc.ents]
        self.named_entities = [(token.text, "NOT_NE") if token.text not in ne_texts else ne[ne_texts.index(token.text)]
                               for token in self.doc]
        return self.named_entities

    def spell_check(self):
        if len(self.tokens) == 0:
            self.tokenize()
        self.spellchecked_text = []
        for token in self.tokens:
            # if len(token) > 2 and not d.check(token) and len(d.suggest(token)) > 0:
            #    self.spellchecked_text.append(d.suggest(token)[0])
            # else:
            #    self.spellchecked_text.append(token)
            self.spellchecked_text.append(token)
        self.spellchecked_text = ' '.join(self.spellchecked_text)
        spellchecked_tokens = [str(token.text) for token in nlp(self.spellchecked_text)]
        return self.spellchecked_text

    def replace_ne(self):
        if len(self.named_entities) == 0:
            self.ner()
        self.placeholders_text = self.text
        for ent in self.named_entities:
            if ent[1] != 'NOT_NE':
                self.placeholders_text = self.placeholders_text.replace(ent[0], self.tags2words[ent[1]])
        placeholders_tokens = [str(token.text) for token in nlp(self.placeholders_text)]
        return self.placeholders_text

    def clean(self):
        self.clean_text = self.extract_emoticons(self.text)
        self.clean_text = self.clean_punctuation(self.clean_text)
        self.clean_tokens = [str(token.text) for token in nlp(self.clean_text)]
        return self.clean_text

    def extract_emoticons(self, text, tag=0):
        transformed_text = text
        try:
            for emoticon in emoticons_re.keys():
                if emoticon.search(text):
                    for m in emoticon.finditer(text):
                        if tag:
                            placeholder = " [EMOTICON:" + emoticons_re[emoticon] + "] "
                        else:
                            placeholder = " " + emoticons_re[emoticon] + " "
                        transformed_text = transformed_text.replace(m.group(), placeholder)
        except Exception as e:
            print(text)
        return transformed_text

    def clean_punctuation(self, text):
        clean_text = text
        while self.punct.search(clean_text):
            repeated_character = self.punct.search(clean_text).group(0)
            if "." in repeated_character:
                repeated_character_regex = "\." + "{2,}"
                repeated_character = "."
            elif "?" in repeated_character or "*" in repeated_character or "$" in repeated_character:
                repeated_character_regex = "\\" + repeated_character[0] + "+"
                repeated_character = repeated_character[0]
            else:
                repeated_character_regex = repeated_character[0] + "+"
                repeated_character = repeated_character[0]
            clean_text = re.sub(repeated_character_regex, repeated_character, clean_text)
        clean_text = re.sub('([.,!?()*\\\\"\'-:;0-9=\$%\&_])', r' \1 ', clean_text)
        clean_text = re.sub('\s{2,}', ' ', clean_text)
        return clean_text

    def heavy_clean(self):
        self.heavy_clean_text = ' '.join([y.lower() for y in self.text.split() if
                                          not y.lower() in eng_stopwords and not y in num_str and not y in string.punctuation])
        self.heavy_clean_tokens = [str(token.text) for token in nlp(self.heavy_clean_text)]
        return self.heavy_clean_text


def transform_dataset(dataset_original, transformation):
    dataset = dataset_original.copy(deep=True)
    begin = time.time()
    fields = list(set(dataset.columns) & set(['question', 'answer']))
    for field in fields:
        column_name = field + '_' + transformation[0]
        dataset[column_name] = ''
        dataset[column_name] = dataset[column_name].astype(object)
        dataset[column_name] = dataset[field].apply(transformation[1])
    end = time.time()
    print('Transformation:', transformation[0], '\t Time elapsed:', (end - begin))
    return dataset


class OrgQuestion():
    id_q = -1
    subj = ""
    body = ""
    thread = []

    def __init__(self, id_q, subj, body):
        self.id_q = id_q
        self.subj = subj
        self.body = body

    def add_to_thread(self, elem):
        self.thread = self.thread + [elem]

    def pprint(self):
        print('OrgQuestion:\n \tORGQ_ID = %s, \n \tOrgQSubject = %s, \n \tOrgQBody = %s' % (
        self.id_q, self.subj, self.body))
        for question in self.thread:
            question.pprint()


class RelQuestion():
    id_rq = -1
    subj = ""
    body = ""
    relevance = 0
    rank_order = -1
    category = ""
    rel_comments = []

    def __init__(self, id_rq, subj, body, relevance, rank_order, category):
        self.id_rq = id_rq
        self.subj = subj
        self.body = body
        self.relevance = convert_score(relevance)
        self.rank_order = int(rank_order)
        self.category = category

    def add_to_rel_comments(self, elem):
        self.rel_comments = self.rel_comments + [elem]

    def pprint(self):
        print('\tRelQuestion:\n \t\t RELQ_ID = %s, \n \t\t RelQSubject = %s, \n \t\t RelQBody = %s' % (
        self.id_rq, self.subj, self.body))
        print('\n\t\t RELQ_RANKING_ORDER = %d, \n \t\t RELQ_CATEGORY = %s, \n \t\t RELQ_RELEVANCE2ORGQ = %d' % (
        self.rank_order, self.category, self.relevance))
        for comment in self.rel_comments:
            comment.pprint()


class RelComment():
    id_rc = -1
    text = ""
    relevance = 0

    def __init__(self, id_rc, text, relevance):
        self.id_rc = id_rc
        self.text = text
        self.relevance = convert_score(relevance)

    def pprint(self):
        print(
            '\t\t--- RelComment:\n \t\t\t RELC_ID = %s, \n \t\t\t RelCText = %s, \n \t\t\t RELC_RELEVANCE2RELQ = %d' % (
            self.id_rc, self.text, self.relevance))


def convert_score(s):
    if s == 'Bad' or s == 'PotentiallyUseful' or s == 'Irrelevant':
        return -1
    elif s == 'Good' or s == 'PerfectMatch' or s == 'Relevant':
        return 1
    else:
        return 0


def read_xml(files):
    data = []
    thread_count = 0
    rel_q_count = 0
    rel_c_count = 0
    for file in files:
        tree = ET.parse(file)
        root = tree.getroot()
        for thread in root:
            thread_count += 1
            rel_q = thread[0]
            rel_q_body = ''
            rel_q_subj = ''
            for datum in rel_q:
                if datum.tag == 'RelQSubject':
                    rel_q_subj = datum.text
                elif datum.tag == 'RelQBody':
                    if datum.text:
                        rel_q_body = datum.text
            rel_q = RelQuestion(rel_q.attrib['RELQ_ID'], rel_q_subj, rel_q_body, None, \
                                0, rel_q.attrib['RELQ_CATEGORY'])
            for idx, comment in enumerate(thread[1:]):
                rel_c = RelComment(comment.attrib['RELC_ID'], comment[0].text, comment.attrib['RELC_RELEVANCE2RELQ'])
                rel_q.add_to_rel_comments(rel_c)
                rel_c_count += 1
            data.append(rel_q)
            rel_q_count += 1
    print("Threads: ", thread_count)
    print("Questions: ", rel_q_count)
    print("Comments: ", rel_c_count)
    return data


def xml2dataframe_NoLabels(dataset, split_type=''):
    tmp = {}
    for obj in dataset:
        candidates = []
        for c in obj.rel_comments:
            candidates.append(c.id_rc)
        tmp[obj.id_rq] = (' '.join([obj.subj, obj.body]), candidates, split_type)
    dataset_dataframe = pd.DataFrame.from_dict(tmp, orient='index').rename(
        columns={0: 'question', 1: 'candidates', 2: 'split_type'})
    for ind, row in dataset_dataframe.iterrows():
        dataset_dataframe.set_value(ind, 'qid', int(ind.split('_')[0][1:]))
        dataset_dataframe.set_value(ind, 'rid', int(ind.split('_')[1][1:]))
    dataset_dataframe = dataset_dataframe.sort_values(['qid', 'rid'])
    answer_texts_dataset = {}
    for obj in dataset:
        for c in obj.rel_comments:
            answer_texts_dataset[c.id_rc] = c.text
    answer_texts_dataset = pd.DataFrame.from_dict(answer_texts_dataset, orient='index')
    answer_texts_dataset.reset_index(inplace=True)
    answer_texts_dataset = answer_texts_dataset.rename(columns={'index': 'answer_id', 0: 'answer'})
    answer_texts_dataset.head()
    return dataset_dataframe, answer_texts_dataset


def xml2dataframe_Labels(dataset, split_type):
    tmp = {}
    for obj in dataset:
        pool_pos = []
        pool_neg = []
        for c in obj.rel_comments:
            if c.relevance == -1:
                pool_neg.append(c.id_rc)
            else:
                pool_pos.append(c.id_rc)
        tmp[obj.id_rq] = (' '.join([obj.subj, obj.body]), pool_pos, pool_neg, split_type)
    dataset_dataframe = pd.DataFrame.from_dict(tmp, orient='index').rename(
        columns={0: 'question', 1: 'answer_ids', 2: 'pool', 3: 'split_type'})
    for ind, row in dataset_dataframe.iterrows():
        dataset_dataframe.set_value(ind, 'qid', int(ind.split('_')[0][1:]))
        dataset_dataframe.set_value(ind, 'rid', int(ind.split('_')[1][1:]))
    dataset_dataframe = dataset_dataframe.sort_values(['qid', 'rid'])
    answer_texts_dataset = {}
    for obj in dataset:
        for c in obj.rel_comments:
            answer_texts_dataset[c.id_rc] = c.text
    answer_texts_dataset = pd.DataFrame.from_dict(answer_texts_dataset, orient='index')
    answer_texts_dataset.reset_index(inplace=True)
    answer_texts_dataset = answer_texts_dataset.rename(columns={'index': 'answer_id', 0: 'answer'})
    return dataset_dataframe, answer_texts_dataset


def add_answers(dataset, answer_texts, expanded=True):
    dataset['answer_id'] = dataset['answer_ids']
    lst_col = 'answer_id'
    dataset_expanded = pd.DataFrame({col: np.repeat(dataset[col].values, dataset[lst_col].str.len())
                                     for col in dataset.columns.difference([lst_col])
                                     }).assign(**{lst_col: np.concatenate(dataset[lst_col].values)})[
        dataset.columns.tolist()]
    dataset_expanded = dataset_expanded.merge(answer_texts, on='answer_id', how='left')
    return dataset_expanded


def transform_dataset(dataset_original, transformation):
    dataset = dataset_original.copy(deep=True)
    begin = time.time()
    fields = list(set(dataset.columns) & set(['question', 'answer']))
    for field in fields:
        column_name = field + '_' + transformation[0]
        dataset[column_name] = ''
        dataset[column_name] = dataset[column_name].astype(object)
        dataset[column_name] = dataset[field].apply(transformation[1])
    end = time.time()
    print('Transformation:', transformation[0], '\t Time elapsed:', (end - begin))
    return dataset


def objlist2dataframe(obj_list, split_type):
    tmp = {}
    for t in obj_list:
        pool_pos = []
        pool_neg = []
        for c in t.rel_comments:
            if c.relevance == -1:
                pool_neg.append(c.id_rc)
            else:
                pool_pos.append(c.id_rc)
        tmp[t.id_rq] = (' '.join([t.subj, t.body]), pool_pos, ' '.join(pool_neg), split_type)
    dataframe = pd.DataFrame.from_dict(tmp, orient='index').rename(
        columns={0: 'question', 1: 'answer_ids', 2: 'pool', 3: 'split_type'})
    return dataframe
