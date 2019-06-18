# coding:utf-8
# author:wup
# description:
# input:
# output:
# e-mail:wup@nlp.nju.cn
# date:

import yaml
def readFile(filename):
    context = open(filename).readlines()
    return [c.strip() for c in context]


def writeFile(context, filename, append=False):
    if not append:
        with open(filename, 'w+') as fout:
            for co in context:
                fout.write(co + "\n")
    else:
        with open(filename, 'a+') as fout:
            for co in context:
                fout.write(co + "\n")

def load_from_config(path):
    with open(path,"r") as f:
        xx = yaml.load(f.read())
    return xx

def write_config(config,path):
    with open(path,"w") as f:
        yaml.dump(config,f)


def list2str(l, split=" "):
    a = ""
    for li in l:
        a += (str(li) + split)
    a = a[:-len(split)]
    return a
