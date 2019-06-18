#coding:utf-8
#author:wup
#description: 10 fold dataset
#e-mail:wup@nlp.nju.cn
#date:
import pickle as pkl
import os
import pdb
import random
import numpy as np

def avg(l):
    return sum(l)/len(l)

prefix = "fold"

index = 0
rel_voc = pkl.load(open(os.path.join("/home/user_data/wup/fb2m_data","FB2M.rel_voc.pickle"),"rb"))
stat = []
while(index<10):
    train_data = pkl.load(open("/home/user_data/wup/fb2m_data/train.mix.pickle","rb"))
    dev_data = pkl.load(open("/home/user_data/wup/fb2m_data/vaild.mix.pickle","rb"))
    test_data = pkl.load(open("/home/user_data/wup/fb2m_data/test.mix.pickle","rb"))

    train_data.extend(dev_data)
    train_data.extend(test_data)
    all_data=train_data

    all_relation={}

    for t in all_data:
        if t.relation not in all_relation:
            all_relation[t.relation]=[t]
        else:
            all_relation[t.relation].append(t)

    relation_num=len(all_relation)
    relation_key = list(all_relation.keys())
    final_train=[]
    final_dev=[]
    final_test=[]
    try_num=0
    dataset_prefix = "{}-{}".format(prefix,index)
    while(True):
        print("try num: {}".format(try_num))
        try_num+=1
        random.shuffle(relation_key)
        a = len(relation_key)
        relation_key = list(relation_key)
        
        relation_key_num = [len(all_relation[t])*1.0 for t in relation_key]

        test_num=0
        test_relation=[]
        k=0
        while(k<len(relation_key)):
            if test_num<10000:
                test_num+=relation_key_num[k]
                test_relation.append(relation_key[k])
            else:
                break
            k+=1
        vaild_num=0
        vaild_relation = []
        while (k < len(relation_key)):
            if vaild_num < 5000:
                vaild_num += relation_key_num[k]
                vaild_relation.append(relation_key[k])
            else:
                break
            k += 1
        print("vaild unseen relation num: {}".format(len(vaild_relation)))
        print("test unseen relation num: {}".format(len(test_relation)))

        train_relation = []
        for k in all_relation:
            if k not in test_relation and k not in vaild_relation:
                train_relation.append(k)

        vaild_dataset = []
        test_dataset =  []
        last_dataset =[]
        for t in all_data:
            if t.relation in vaild_relation:
                vaild_dataset.append(t)
            elif t.relation in test_relation:
                test_dataset.append(t)
            else:
                last_dataset.append(t)
        
        vaild_unseen_relation = len(set([t.relation for t in vaild_dataset]))
        test_unseen_relation = len(set([t.relation for t in test_dataset]))


        random.shuffle(last_dataset)
        last_num=len(last_dataset)

        train_dataset=[]
        train_split = int(last_num*7/8.5)
        vaild_split = int(last_num*7.5/8.5)
        vaild_seen_part = last_dataset[train_split:vaild_split]
        test_seen_part = last_dataset[vaild_split:]
        vaild_seen_relation = len(set([t.relation for t in vaild_seen_part]))
        test_seen_relation = len(set([t.relation for t in test_seen_part]))


        train_dataset.extend(last_dataset[:train_split])
        vaild_dataset.extend(last_dataset[train_split:vaild_split])
        test_dataset.extend(last_dataset[vaild_split:])


        
        if len(test_relation)>300: 
            final_train=train_dataset
            final_dev=vaild_dataset
            final_test = test_dataset
            break

    train_relation = set([t.relation for t in final_train])
    seen_vaild = [t for t in final_dev if t.relation in train_relation]
    unseen_vaild = [t for t in final_dev if t.relation not in train_relation]
    seen_test = [t for t in final_test if t.relation in train_relation]
    unseen_test = [t for t in final_test if t.relation not in train_relation]

    stat.append("Samples\ttrain\tdev-seen\tdev-unseen\ttest-seen\ttest-unseen")
    stat.append("{}\t{}\t{}\t{}\t{}\t{}\t".format(dataset_prefix,len(final_train),len(seen_vaild),len(unseen_vaild),len(seen_test),len(unseen_test)))
    stat.append("Relation\ttrain\tdev-seen\tdev-unseen\ttest-seen\ttest-unseen")
    stat.append("{}\t{}\t{}\t{}\t{}\t{}\t".format(dataset_prefix,len(train_relation), vaild_seen_relation, vaild_unseen_relation, test_seen_relation,
                                        test_unseen_relation))
    stat.append("")
    ll = len(stat)
    for t in stat[ll-5:-1]:
        print(t)

    pkl.dump(final_train,open("{}.train.pickle".format(dataset_prefix),"wb"))
    pkl.dump(final_dev,open("{}.vaild.pickle".format(dataset_prefix),"wb"))
    pkl.dump(final_test,open("{}.test.pickle".format(dataset_prefix),"wb"))

    index+=1

with open("stat.txt","w") as fout:
    for t in stat:
        fout.write(t+"\n")
