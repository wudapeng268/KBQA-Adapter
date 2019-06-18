# coding:utf-8
# author:wup
# description: prepare train test data pre batch
# input:
# output:
# e-mail:wup@nlp.nju.cn
# date:
import pdb
import pickle as pkl

import numpy as np
import time
from src.util import FileUtil
import random
import tensorflow as tf
stop_words = set(
    [u'all', u'just', u"don't", u'being', u'over', u'both', u'through', u'yourselves', u'its', u'before', u'o', u'don',
     u'hadn', u'herself', u'll', u'had', u'should', u'to', u'only', u'won', u'under', u'ours', u'has', u"should've",
     u"haven't", u'do', u'them', u'his', u'very', u"you've", u'they', u'not', u'during', u'now', u'him', u'nor',
     u"wasn't", u'd', u'did', u'didn', u'this', u'she', u'each', u'further', u"won't", u'where', u"mustn't", u"isn't",
     u'few', u'because', u"you'd", u'doing', u'some', u'hasn', u"hasn't", u'are', u'our', u'ourselves', u'out', u'what',
     u'for', u"needn't", u'below', u're', u'does', u"shouldn't", u'above', u'between', u'mustn', u't', u'be', u'we',
     u'who', u"mightn't", u"doesn't", u'were', u'here', u'shouldn', u'hers', u"aren't", u'by', u'on', u'about',
     u'couldn', u'of', u"wouldn't", u'against', u's', u'isn', u'or', u'own', u'into', u'yourself', u'down', u"hadn't",
     u'mightn', u"couldn't", u'wasn', u'your', u"you're", u'from', u'her', u'their', u'aren', u"it's", u'there',
     u'been', u'whom', u'too', u'wouldn', u'themselves', u'weren', u'was', u'until', u'more', u'himself', u'that',
     u"didn't", u'but', u"that'll", u'with', u'than', u'those', u'he', u'me', u'myself', u'ma', u"weren't", u'these',
     u'up', u'will', u'while', u'ain', u'can', u'theirs', u'my', u'and', u've', u'then', u'is', u'am', u'it', u'doesn',
     u'an', u'as', u'itself', u'at', u'have', u'in', u'any', u'if', u'again', u'no', u'when', u'same', u'how', u'other',
     u'which', u'you', u"shan't", u'shan', u'needn', u'haven', u'after', u'most', u'such', u'why', u'a', u'off', u'i',
     u'm', u'yours', u"you'll", u'so', u'y', u"she's", u'the', u'having', u'once']
)
stop_words.add("'s")  # add 's


class SimpleQA:
    def __init__(self, config, kbqa_flag=False):
        self.config = config
        self.n_step = config['model']['n_step']
        self.type_len = 500

        self.fix_random_seed = config['run_op']['fix_random_seed']
        self.get_word_emb(config['data']['word_emb_path'], config['data']['word_voc_path'])
        self.word_embedding_len = len(self.word_embedding)

        self.rel_voc = pkl.load(open(config['data']['rel_voc_path'], "rb"))
        self.relation_size = int(len(self.rel_voc) / 2)


        self.negtivate_relation_sample = config['model']['negtivate_relation_sample']
        self.all_relation = [i for i in range(self.relation_size)]

        self.subject2relation = pkl.load(open(config['data']['subject2relation_path'], "rb"))
        self.dev_data = pkl.load(open(config['data']['dev_path'], "rb"))
        self.train_data = pkl.load(open(config['data']['train_path'], "rb"))
        self.train_relation = set([t.relation for t in self.train_data])
        self.test_data = pkl.load(open(config['data']['test_path'], "rb"))
        self.max_candidate_relation = 0
        if kbqa_flag or self.config['test_wq'] or self.config['train_wq']:
            tt=self.test_data
            tf.logging.info("kbqa flag: {}".format(kbqa_flag))
            for t in tt:
                self.max_candidate_relation = max(len(t.cand_rel), self.max_candidate_relation)
        else:
            tt = self.dev_data + self.test_data
            for t in tt:
                if t.subject in self.subject2relation:
                    if len(self.subject2relation[t.subject]) > self.max_candidate_relation:
                        self.max_candidate_relation = len(self.subject2relation[t.subject])
        tf.logging.info("max_candidate_relation {}".format(self.max_candidate_relation))
        
        tf.logging.info("test data lens: {}".format(len(self.test_data)))
        if "fix_negtive_sampling" in self.config['model'] and self.config['model']['fix_negtive_sampling']:
            tf.logging.info("Fix negtivate sampling!!")
            self.old_negtivate = {}

    def get_word_emb(self, emb_path, voc_path):
        tf.logging.info("===>load pre-trained word emb and voc<===")
        self.word_embedding = np.load(emb_path)
        self.word_vocabulary = pkl.load(open(voc_path, "rb"))

    itemIndexTrain = 0
    itemIndexDev = 0
    itemIndexTest = 0

    def word2id(self, query):
        vector = np.zeros(self.n_step)
        for i, word in enumerate(query.split(" ")):
            if i >= self.n_step:
                break
            if word in self.word_vocabulary and self.word_vocabulary[word] < self.word_embedding_len:
                vector[i] = self.word_vocabulary[word]
            else:
                vector[i] = self.word_embedding_len - 1
        return vector

    def load_train_data(self, batch_size):
        if self.fix_random_seed:
            random.seed(1234)
        if self.itemIndexTrain >= len(self.train_data):
            tf.logging.ERROR("########bigger?!")
            self.itemIndexTrain = 0
        if self.itemIndexTrain + batch_size > len(self.train_data):
            batch_size = len(self.train_data) - self.itemIndexTrain

        batch_x_anonymous = np.zeros((batch_size, self.n_step), dtype=np.int32)
        batch_x_anonymous_lens = np.zeros((batch_size), dtype=np.float32)

        batch_relation_index = np.zeros((batch_size, 1 + self.negtivate_relation_sample))
        batch_relation_lens = np.zeros((batch_size))

        for ind, it in enumerate(self.train_data[self.itemIndexTrain:self.itemIndexTrain + batch_size]):
            anonymous_question = it.anonymous_question
            batch_x_anonymous_lens[ind] = len(anonymous_question.split(" "))
            batch_x_anonymous[ind, :] = self.word2id(anonymous_question)
                
            self.all_relation.remove(it.relation)
            tt = random.sample(self.all_relation, self.negtivate_relation_sample)
            self.all_relation.append(it.relation)
            tt = [it.relation] + tt
            batch_relation_lens[ind] = len(tt)
            batch_relation_index[ind, :] = tt

        self.itemIndexTrain += batch_size
        return_value = {}
        value_name = ['batch_x_anonymous', 'batch_x_anonymous_lens', 'batch_relation_index', 'batch_relation_lens']

        for name in value_name:
            return_value[name] = eval(name)
        return return_value

    def load_test_data(self, batch_size, dataset="test", kbqa_flag=False):
        if dataset == "test":
            runitem = self.test_data
            run_item_index = self.itemIndexTest
        else:
            runitem = self.dev_data
            run_item_index = self.itemIndexDev
        if run_item_index + batch_size > len(runitem):
            batch_size = len(runitem) - run_item_index

        batch_x_anonymous = np.zeros((batch_size, self.n_step), dtype=np.int32)
        batch_x_anonymous_lens = np.zeros((batch_size), dtype=np.float32)

        batch_relation_index = np.zeros((batch_size, self.max_candidate_relation))
        batch_relation_lens = np.zeros((batch_size))
        qids = []
        questions = []
        gold_relation = []
        gold_subject = []
        cand_rel_list = []

        for ind, it in enumerate(runitem[run_item_index:run_item_index + batch_size]):
            question = it.question
            questions.append(question)
            qids.append(it.qid)

            anonymous_question = it.anonymous_question
            batch_x_anonymous_lens[ind] = len(anonymous_question.split(" "))
            batch_x_anonymous[ind, :] = self.word2id(anonymous_question)

            gold_relation.append(it.relation)
            gold_subject.append(it.subject)
            num = 0
            temp_cand = []
            if it.subject in self.subject2relation:
                for t in self.subject2relation[it.subject]:
                    batch_relation_index[ind, num] = t
                    temp_cand.append(t)
                    num += 1

            cand_rel_list.append(temp_cand)
            batch_relation_lens[ind] = num

        if dataset == "test":
            self.itemIndexTest += batch_size
        else:
            self.itemIndexDev += batch_size
        return_value = {}

        value_name = ['batch_x_anonymous', 'batch_x_anonymous_lens',
                      'batch_size', 'questions', 'cand_rel_list', 'batch_relation_lens',
                      'gold_relation',
                      ]
        value_name.append('gold_subject')
        value_name.append("batch_relation_index")
        value_name.append("qids")
        for name in value_name:
            return_value[name] = eval(name)
        return return_value
