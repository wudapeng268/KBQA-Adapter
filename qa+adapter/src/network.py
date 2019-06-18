# coding:utf-8
# author:wup
# description: 
# e-mail:
# date: 
import tensorflow as tf
from src.load_data import SimpleQA
from src.util import NN
from src.util import FileUtil
from src.adapter import adapter
import numpy as np
import pdb
import pickle as pkl


class BiGRU(object):
    def __init__(self, sess, config, kbqa_flag):
        self.sess = sess
        self.config = config
        self.n_steps = config['model']['n_step']
        self.n_hidden = config['model']['n_hidden']
        self.fix_batch_size = config['model']['batch_size']

        self.display_step = 100
        self.dev_batch_size = config['model']['dev_batch_size']

        self.adapter_keep_prob = config['model']['adapter_dropout']
        self.gama_r = config['model']['gama_r']
        print("gama_r {}".format(self.gama_r))
        self.train_word_embedding = config['model']['train_word_embedding']
        print("train word emb {}".format(self.train_word_embedding))
        self.fine_tune_part_emb = config['model']['fine_tune_part_emb']
        self.transX_embedding_path = config['data']['transX_embedding_path']
        self.relation_rep = config['model']['relation_rep']

        self.qa = SimpleQA(config, kbqa_flag)

        self.deving_iters = len(self.qa.dev_data)
        self.training_iters = len(self.qa.train_data)
        self.testing_iters = len(self.qa.test_data)

        # for test in run_op.py
        self.train_relation = set([t.relation for t in self.qa.train_data])


        self.relation_vocabulary_size = int(self.qa.relation_size)
        self.relation_embedding_size = int(config['model']['relation_embedding_size'])

        self.relation_word_id_path = config['data']['relation_word_id_path']
        self.relation_word_len_path = config['data']['relation_word_len_path']
        self.relation_part_id_path = config['data']['relation_part_id_path']
        self.relation_part_len_path = config['data']['relation_part_len_path']

        self.build_model()

    def matmul_query_relation(self, query_vec, rel_vec):

        query_vec = tf.expand_dims(query_vec, 1)
        # [batch,1,cand_rel]
        score = tf.matmul(query_vec, tf.transpose(rel_vec, [0, 2, 1]))
        score = tf.squeeze(score, 1)
        query_vec = tf.squeeze(query_vec)
        score /= (tf.expand_dims(tf.sqrt(tf.reduce_sum(query_vec ** 2, 1)), 1))
        score /= (tf.sqrt(tf.reduce_sum(rel_vec ** 2, 2)))
        # [batch,cand_rel]
        return score

    def load_relation_emb(self, path):
        if path.endswith("npy"):
            return np.load(path)
        context = FileUtil.readFile(path)
        rel_embedding = []
        for t in context:
            data = t.split("\t")
            data = [float(x) for x in data]
            rel_embedding.append(data)
        return np.array(rel_embedding, dtype=np.float32)

    def relation_network(self, x, relation_index, relation_len, rel_vec, share_name):
        '''
        :param x: query embedding after look-up
        :param relation_index: cand relation by sample or gold subject
        :param relation_len: lens of relation_index only use in test
        :param rel_vec: all relation vector look up embedding in this funcation
        :return: rel_score:[batch_size,relation_voc_size], loss:[1]
        '''
        self.debug_x = x
        query4relation, self.output1, self.output2 = NN.short_conn_bi_lstm(x, self.x_lens, self.n_hidden,"bi_gru4relation_query", share_name,self.config)

        m = tf.sequence_mask(self.x_lens,self.n_steps,dtype=tf.float32)
        query4relation = query4relation+tf.expand_dims((1-m)*-1e12,2)
        query4relation = tf.reduce_max(query4relation, 1)


        self.debug_q = query4relation
        with tf.variable_scope("relation_network"):
            rel_emb = tf.nn.embedding_lookup(rel_vec, relation_index)

            rel_score = self.matmul_query_relation(query4relation, rel_emb)
            rel_score = tf.exp(rel_score - tf.reduce_max(rel_score, 1, keepdims=True))
            relation_mask = tf.sequence_mask(relation_len, self.qa.max_candidate_relation, dtype=tf.float32)
            rel_score_test = rel_score + (1-relation_mask)*-1e12

            only_rel_predict = tf.argmax(rel_score_test, 1)
            # [1]
            loss_relation = tf.maximum(rel_score[:, 1:] - rel_score[:, 0][:, None] + self.gama_r, 0)
            loss_relation = tf.reduce_mean(tf.reduce_sum(
                loss_relation, axis=1))
        return rel_score, loss_relation, only_rel_predict

    def relation_network4test(self, x, relation_index, relation_len, rel_vec, share_name):
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope, reuse=True):
            return self.relation_network(x, relation_index, relation_len, rel_vec, share_name)

    def get_relation_vector(self, relation_word_id, relation_word_len, word_emb, share_name):
        # share lstm
        with tf.variable_scope("get_realtion_vec_network", reuse=tf.AUTO_REUSE):
            relation_word_vec = tf.nn.embedding_lookup(word_emb, relation_word_id)

            rel_vec, rel_vec_state = NN.bi_lstm(relation_word_vec, relation_word_len, self.n_hidden, share_name)

            # [6700,n_step,n_hidden*2]
            rel_vec = tf.concat(rel_vec, 2)

           

            # [6700,1,n_hidden*2]
            rel_vec_state = tf.expand_dims(tf.concat(rel_vec_state, 1), 1)
            return rel_vec, rel_vec_state


    def build_model(self):
        self.question_ids = tf.placeholder(tf.int32, [None, self.n_steps])
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.word_embedding = tf.Variable(self.qa.word_embedding, trainable=self.train_word_embedding,name="word_embedding")

        # self.word_embedding=self.word_embedding

        t_word_embedding_train = self.word_embedding
        t_word_embedding_test = self.word_embedding
    
        relation_word_id = tf.constant(np.load(self.relation_word_id_path), dtype=tf.int32)
        relation_word_len = tf.constant(np.load(self.relation_word_len_path), dtype=tf.int32)

        # use random or transX for target
        if self.config['model']['target_method'] == "random":
            self.rel_embedding = tf.Variable(
                tf.random_uniform(shape=[self.relation_vocabulary_size, self.relation_embedding_size],
                                  minval=-0.08,
                                  maxval=0.08), trainable=self.fine_tune_part_emb, name="relation_part_embedding")
        elif self.config['model']['target_method'] == "nre_rel":
            self.rel_embedding = tf.Variable(
                self.load_relation_emb(self.transX_embedding_path), trainable=self.fine_tune_part_emb,
                name="relation_part_embedding")
        else:
            raise Exception("error in model.target_method of config")

        relation_part_id = tf.constant(np.load(self.relation_part_id_path), dtype=tf.int32)
        relation_part_len = tf.constant(np.load(self.relation_part_len_path), dtype=tf.int32)

        self.x_lens = tf.placeholder(tf.int32, [None])
        self.relation_index = tf.placeholder(tf.int32, [None, None])

        share_name = "word_relation"
        share_name2 = "deep_abstract"

        
        rel_word_vec_train, rel_word_vec_train_state = self.get_relation_vector(relation_word_id, relation_word_len,t_word_embedding_train,share_name)
        rel_word_vec_test, rel_word_vec_test_state = self.get_relation_vector(relation_word_id, relation_word_len,t_word_embedding_test,share_name)

        m = tf.sequence_mask(relation_word_len,relation_word_id.shape[-1],dtype=tf.float32)
        rel_word_vec_train = rel_word_vec_train+tf.expand_dims((1-m)*-1e12,2)
        rel_word_vec_test = rel_word_vec_test+tf.expand_dims((1-m)*-1e12,2)

            
        if not self.config['model']['relation_adapter_flag']:
            # not adapter
            rel_part_train, _ = self.get_relation_vector(relation_part_id, relation_part_len,self.rel_embedding,share_name)
            rel_part_test = rel_part_train
        else:
            # pre-trained emb
            
            global_emb = tf.constant(self.load_relation_emb(self.transX_embedding_path), dtype=tf.float32,name="relation_part_embedding")
            self.general_emb = global_emb

            with tf.variable_scope("adapter"):

                self.relation_adapter = adapter(global_emb, self.rel_embedding, self.adapter_keep_prob,self.relation_index[:, 0], "relation")
                self.relation_adapter.forward(self.config['model']['relation_adapter']['forward_method'])

                adapter_output = self.relation_adapter.adapter_output
                self.temp_mapping = adapter_output
                if 'dual_loss_alpha' in self.config['model']['relation_adapter']:
                    dual_loss_alpha = self.config['model']['relation_adapter']['dual_loss_alpha']
                else:
                    dual_loss_alpha = 1
                print("relation dual loss alpha {}".format(dual_loss_alpha))

                self.relation_adapter.define_loss(self.is_training,
                                                  self.config['model']['relation_adapter']['loss_method'],
                                                  dual_loss_alpha)

                self.E_o = adapter_output
                if self.config['model']['relation_adapter']['train_method'] != "None":
                    rel_part_train, _ = self.get_relation_vector(relation_part_id, relation_part_len,self.rel_embedding,share_name)
                    rel_part_test, _ = self.get_relation_vector(relation_part_id, relation_part_len, adapter_output,share_name)
                else:
                    rel_part_test, _ = self.get_relation_vector(relation_part_id, relation_part_len, adapter_output,share_name)
                    rel_part_train = rel_part_test

        self.word_test = rel_word_vec_test
        self.part_test = rel_part_test
        # [6700,n_step,n_hidden*2]
        print("use relation rep {}".format(self.relation_rep))
        rel_vec_train = tf.reduce_max(
            tf.concat([rel_word_vec_train, rel_part_train], 1), 1)
        rel_vec_test = tf.reduce_max(
            tf.concat([rel_word_vec_test, rel_part_test], 1), 1)
        

        self.relation_lens = tf.placeholder(tf.int32, [None])
        self.cand_rel = tf.placeholder(tf.float32, [None, self.relation_vocabulary_size])
        x_train = tf.nn.embedding_lookup(t_word_embedding_train, ids=self.question_ids)
        x_test = tf.nn.embedding_lookup(t_word_embedding_test, ids=self.question_ids)

        _, self.score_loss, self.rel4train = self.relation_network(x_train, self.relation_index, self.relation_lens, rel_vec_train, share_name)
        self.rel_score, self.score_loss_test, self.rel_pred = self.relation_network4test(x_test, self.relation_index,self.relation_lens,rel_vec_test, share_name)
        
        self.saver = tf.train.Saver()
