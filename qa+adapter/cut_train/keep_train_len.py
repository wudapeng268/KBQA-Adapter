import pickle as pkl

fold = 3
train = pkl.load(open("/home/user_data/wup/10-fold-dataset/fold-{}.train.pickle".format(fold),"rb"))
import os
train_len = 60000
dir_path = "same-tl-{}-train-divide".format(train_len)
# os.rmdir(dir_path)
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

train_relation = set([t.relation for t in train])

relation_num=range(300,900,100)

import random
import collections
import pdb

train_map = collections.defaultdict(list)

for t in train:
    train_map[t.relation].append(t)

relation_sort = sorted(train_map.items(),key=lambda item:len(item[1]),reverse=True)

ss=100
fix = relation_sort[-ss:]
fix_key = set([t[0] for t in fix])
fix_num = sum([len(t[1]) for t in fix])
fix_train = [t for t in train if t.relation in fix_key]
print("fix num {}".format(fix_num))
fre_key = []
fre_relation_size=100
for i in range(fre_relation_size):
    fre_key.append(relation_sort[i][0])
fre_train = [t for t in train if t.relation in fre_key]

last_train = [t for t in train if t.relation not in fix_key]
last_relation = set([t.relation for t in train if t.relation not in fix_key])

last_sample_num = train_len-fix_num
for i,n in enumerate(relation_num):
    last_key_num = n - ss
    # print("create relation size {}".format(n))

    sample_relation = random.sample(last_relation, last_key_num)
    sample_train = set([t for t in last_train if t.relation in sample_relation])
    # pdb.set_trace()
    small_train = []
    if len(sample_train)>last_sample_num:
        # print("big")
        small_train.extend([train_map[t][0] for t in sample_relation])
        # pdb.set_trace()
        small_train.extend(random.sample(sample_train,last_sample_num-len(sample_relation)))
        final_train = small_train+fix_train
    else:
        small_train = list(sample_train)

        small_train.extend(random.sample(fre_train,last_sample_num-len(sample_train)))
        final_train = small_train+fix_train

    print("train size {}".format(len(final_train)))

    relation_len = len(set([t.relation for t in final_train]))
    print("train relation size {}".format(relation_len))

    new_path = os.path.join(dir_path,"train-relation-{}.pickle".format(n))

    # print("ok!")

    pkl.dump(final_train,open(new_path,"wb"))

'''
train size 60000
train relation size 383
train size 60000
train relation size 473
train size 60000
train relation size 561
train size 60000
train relation size 661
train size 60000
train relation size 749
train size 60000
train relation size 845    
'''
