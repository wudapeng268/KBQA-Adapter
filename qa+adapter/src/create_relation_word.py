import pickle as pkl
rel_voc = pkl.load(open("/home/user_data/wup/fb2m_data/FB2M.rel_voc.pickle","rb"))
word_voc = pkl.load(open("/home/user_data/wup/fb2m_data/nre_data/word.voc.nre.pickle","rb"))
import numpy as np
new_word_voc={}
for w in word_voc:
    if word_voc[w]<100003:
        new_word_voc[w] = word_voc[w]
pkl.dump(new_word_voc,open("/home/user_data55/wup/new.word.voc.pickle","wb"))
word_voc  =new_word_voc


relation_size = int(len(rel_voc)/2)
relation_id2words = {}
for i in range(relation_size):
    relation_name = rel_voc[i]
    words = relation_name.replace("_"," ").replace("."," ").split(" ")
    relation_id2words[i]=words
max_len = max([len(relation_id2words[t]) for t in relation_id2words])
print("max_len {}".format(max_len))

relation_word_id = np.zeros([relation_size,max_len])
relation_word_len= np.zeros([relation_size])

for i in range(relation_size):
    words = relation_id2words[i]
    for j,w in enumerate(words):
        if w in word_voc:
            relation_word_id[i,j]=word_voc[w]
    relation_word_len[i] = len(words)
# import pdb
# pdb.set_trace()
#check!

np.save("/home/user_data55/wup/new.word.id.npy",relation_word_id)
np.save("/home/user_data55/wup/new.word.len.npy",relation_word_len)

##change word voc to 100003
