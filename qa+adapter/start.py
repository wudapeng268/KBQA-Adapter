#author:wup
#description: start file
#e-mail:wup@nlp.nju.cn
#date: 2018.4.4

import os
import tensorflow as tf
import time


GPUCARD = os.environ.get("CUDA_VISIBLE_DEVICES")
if GPUCARD == None:
    tf.logging.info("forget choose gpu")
    tf.logging.info("export CUDA_VISIBLE_DEVICES=")
    os._exit(1)




from util import FileUtil
import pickle as pkl
import argparse
parse = argparse.ArgumentParser()
parse.add_argument("--run", type=str, default="run")
parse.add_argument("--config", type=str, default="")
parse.add_argument("--fold",type=int,help="10-fold")
parse.add_argument("--key_num",type=int,help="exp key")
# vaild experiment

parse.add_argument("--same_tl",action="store_true",help="run same train len")
parse.add_argument("--star_model",action="store_true",help="run same relation size")

parse.add_argument("--test_raw_model",action="store_true",help="test_raw_model")
parse.add_argument("--test_wq",type=str,default=None,help="test_raw_model")
parse.add_argument("--train_wq",action="store_true",help="test_raw_model")
parse.add_argument("--train_new_wq",action="store_true",help="test_raw_model")

args = parse.parse_args()

if args.train_new_wq:
    args.train_wq= True

tf.logging.info("Use tf")
from src.network import BiGRU
from src import run_op as run_op

run = args.run
config = FileUtil.load_from_config(args.config)

def create_dir(path):
    if not os.path.exists(path):
        os.system("mkdir -p {}".format(path))

default_model_dir = "/home/wup/qa+adapter_new_dev/model"
if "model_path" in config['run_op']:
    default_model_dir = config['run_op']['model_dir']


default_log_dir = "/home/wup/qa+adapter_new_dev/log"
if "log_path" in config['run_op']:
    default_log_dir = config['run_op']['log_dir']


if not os.path.exists(default_model_dir):
    os.system("mkdir -p {}".format(default_model_dir))
if not os.path.exists(default_log_dir):
    os.system("mkdir -p {}".format(default_log_dir))

tf.logging.set_verbosity(tf.logging.INFO)

def print_config(config):
    if type(config)==dict:
        for k in config:
            if type(config[k])==dict:
                print_config(config[k])
            else:
                tf.logging.info("{}:\t {}".format(k,config[k]))
    else:
        return



def create_and_train_model(config,tfconfig,fold):
    with tf.Session(config=tfconfig) as sess:
        # if args.tf:
        if config['run_op']['fix_random_seed']:
            tf.set_random_seed(12345)  # set random seed
            tf.logging.info("Fix random seed")
        model = BiGRU(sess, config, kbqa_flag)
        model.model_path = model_path
        model.log_path = log_path

        tf.logging.info("Print logging")
        print_config(config)
        time.sleep(3)

        if run == "train":
            run_op.train(model, config)
        elif run == "test":
            # pass
            run_op.test(model, config, True, True)
        elif run == "test_kbqa":
            run_op.test(model, config, True, True, kbqa_flag=True)
        elif run == "vis_emb":
            run_op.visualization_emb(model, config)
        else:
            tf.logging.info("error in run! only accept train or test")
            exit(1)

        FileUtil.writeFile([model.model_path], "current_train_model_name.txt", True)


fold = args.fold
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
tf.logging.info("10-fold:\t {}".format(fold))

key_num = args.key_num
exp = ""

# create dataset path
if args.divide_train:
    exp = "divide_train"
    config['data']['train_path'] = "/home/user_data/wup/fold-3-train-divide/train-sample-{}.pickle".format(key_num)
    config['model']['name'] = config['model']['name'] + "-{}-{}".format(fold,key_num)
elif args.same_rs_new:
    config['data']['train_path'] = "/home/user_data55/wup/same-rs-new-900-train-divide/train-sample-{}.pickle".format(key_num)
    config['model']['name'] = config['model']['name'] + "-rs-new-{}".format(key_num)
elif args.same_tl:
    config['data']['train_path'] = "/home/user_data/wup/same-tl-60000-train-divide/train-relation-{}.pickle".format(key_num)
    config['model']['name'] = config['model']['name'] + "-tl-{}".format(key_num)
elif args.train_wq:
    pass
else:
    config['model']['name'] = config['model']['name'] + "-{}".format(fold)
    config['data']['train_path'] = "/home/user_data/wup/10-fold-dataset/fold-{}.train.pickle".format(fold)

tf.logging.info("train path:\t {}".format(config['data']['train_path']))
tf.logging.info("model name:\t {}".format(config['model']['name']))

if not args.train_wq:
    config['data']['dev_path'] = "/home/user_data/wup/10-fold-dataset/fold-{}.vaild.pickle".format(fold)
    config['data']['test_path'] = "/home/user_data/wup/10-fold-dataset/fold-{}.test.pickle".format(fold)
    config['data']['seen_word_path'] = "/home/user_data/wup/10-fold-dataset/fold-{}.seen_word.test.pickle".format(fold)
    config['data']['unseen_word_path'] = "/home/user_data/wup/10-fold-dataset/fold-{}.unseen_word.test.pickle".format(fold)

kbqa_flag=False


if run=="test_kbqa":
    kbqa_flag=True
    config['data']['dev_path'] = config['data']['dev_path']+".cand"
    config['data']['test_path'] = config['data']['test_path']+".cand"


baseline_path = {}
pretrain_path = "/home/user_data/wup/10-fold-baseline"
baseline_path[0] = os.path.join(pretrain_path,"paper.baseline-0/model.ckpt-14950")
baseline_path[1] = os.path.join(pretrain_path,"paper.baseline-1/model.ckpt-10030")
baseline_path[2] = os.path.join(pretrain_path,"paper.baseline-2/model.ckpt-17043")
baseline_path[3] = os.path.join(pretrain_path,"paper.baseline-3/model.ckpt-6090")

baseline_path[4] = os.path.join(pretrain_path,"paper.baseline-4/model.ckpt-14800")
baseline_path[5] = os.path.join(pretrain_path,"paper.baseline-5/model.ckpt-6622")
baseline_path[6] = os.path.join(pretrain_path,"paper.baseline-6/model.ckpt-9207")
baseline_path[7] = os.path.join(pretrain_path,"paper.baseline-7/model.ckpt-11248")
baseline_path[8] = os.path.join(pretrain_path,"paper.baseline-8/model.ckpt-12980")
baseline_path[9] = os.path.join(pretrain_path,"paper.baseline-9/model.ckpt-6556")

model_path = os.path.join(default_model_dir,config['model']['name'])
log_path = os.path.join(default_log_dir,config['model']['name'])

create_dir(model_path)
create_dir(log_path)


# train model with adapter
if 'all' in config['model']['name']:
    if args.same_tl:
        tf.logging.info("run:\t {}".format("same_tl"))
        curr_baseline_path = os.path.join(default_model_dir, "paper.baseline-tl-{}".format(key_num))
        restore_file = tf.train.latest_checkpoint(curr_baseline_path)
    elif args.star_model:
        tf.logging.info("run:\t {}".format("star_model"))
        curr_baseline_path = os.path.join(default_model_dir, "paper.baseline.alpha.gan-{}".format(fold))
        restore_file = tf.train.latest_checkpoint(curr_baseline_path)
    elif args.train_new_wq:
        tf.logging.info("run:\t {}".format("train new wq"))
        curr_baseline_path = os.path.join(default_model_dir, "new.webq.baseline")
        restore_file = tf.train.latest_checkpoint(curr_baseline_path)
    elif (args.train_wq and not args.train_new_wq):
        tf.logging.info("run:\t {}".format("train wq"))
        curr_baseline_path = os.path.join(default_model_dir, "webq.baseline")
        restore_file = tf.train.latest_checkpoint(curr_baseline_path)
    else:
        tf.logging.info("run:\t {}".format("adapter"))
        curr_baseline_path = os.path.join(default_model_dir, "paper.baseline-{}".format(key_num))
        restore_file = tf.train.latest_checkpoint(curr_baseline_path)
        # restore_file = baseline_path[fold]
    config['run_op']['restore_file'] = restore_file


if run=="train":
    FileUtil.write_config(config,"{}/config".format(model_path))

config['fold'] = fold

if args.test_raw_model:
    model_path = os.path.join("/home/user_data55/wup/acl_submit/model/", config['model']['name'])
    tf.logging.info("now model.model_path {}".format(model_path))

if args.test_wq!=None:
    config['data']['test_path'] = args.test_wq
    config['test_wq'] = True
    tf.logging.info("now test path is {}".format(args.test_wq))
else:
    config['test_wq'] = False

if args.train_wq:
    config['train_wq'] = args.train_wq
else:
    config['train_wq'] = False


create_and_train_model(config,tfconfig,fold)


