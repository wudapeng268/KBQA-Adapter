# coding:utf-8
# author:wup
# description:
# input:
# output:
# e-mail:wup@nlp.nju.cn
# date:

import yaml
import io

def readFile(filename):
    context = open(filename).readlines()
    return [c.strip() for c in context]


def writeFile(context, filename, append=False):
    if not append:
        with io.open(filename, 'w+',encoding="utf-8") as fout:
            for co in context:
                fout.write(co + "\n")
    else:
        with io.open(filename, 'a+',encoding = "utf-8") as fout:
            for co in context:
                fout.write(co + "\n")

def load_from_config(path):
    with io.open(path,"r",encoding="utf-8") as f:
        xx = yaml.load(f.read())
    return xx

def write_config(config,path):
    with io.open(path,"w",encoding="utf-8") as f:
        yaml.dump(config,f)


def list2str(l, split=" "):
    a = ""
    for li in l:
        a += (str(li) + split)
    a = a[:-len(split)]
    return a
