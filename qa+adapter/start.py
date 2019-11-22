#author:wup
#description: start file
#e-mail:wup@nlp.nju.cn
#date: 

import os
import tensorflow as tf
import time


GPUCARD = os.environ.get("CUDA_VISIBLE_DEVICES")
if GPUCARD == None:
    tf.logging.info("forget choose gpu")
    tf.logging.info("export CUDA_VISIBLE_DEVICES=")
    os._exit(1)




from src.util import FileUtil
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
args = parse.parse_args()



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
if args.same_tl:
    config['data']['train_path'] = "/home/user_data/wup/same-tl-60000-train-divide/train-relation-{}.pickle".format(key_num)
    config['model']['name'] = config['model']['name'] + "-tl-{}".format(key_num)
else:
    config['model']['name'] = config['model']['name'] + "-{}".format(fold)
    config['data']['train_path'] = "/home/user_data/wup/10-fold-dataset/fold-{}.train.pickle".format(fold)

config['data']['dev_path'] = "/home/user_data/wup/10-fold-dataset/fold-{}.vaild.pickle".format(fold)
config['data']['test_path'] = "/home/user_data/wup/10-fold-dataset/fold-{}.test.pickle".format(fold)

tf.logging.info("train path:\t {}".format(config['data']['train_path']))
tf.logging.info("model name:\t {}".format(config['model']['name']))


kbqa_flag=False


if run=="test_kbqa":
    kbqa_flag=True
    config['data']['dev_path'] = config['data']['dev_path']+".cand"
    config['data']['test_path'] = config['data']['test_path']+".cand"



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
        curr_baseline_path = os.path.join(default_model_dir, "paper.baseline.star-{}".format(fold))
        restore_file = tf.train.latest_checkpoint(curr_baseline_path)
    else:
        tf.logging.info("run:\t {}".format("adapter"))
        curr_baseline_path = os.path.join(default_model_dir, "paper.baseline-{}".format(key_num))
        restore_file = tf.train.latest_checkpoint(curr_baseline_path)

    config['run_op']['restore_file'] = restore_file


if run=="train":
    FileUtil.write_config(config,"{}/config".format(model_path))

config['fold'] = fold


create_and_train_model(config,tfconfig,fold)


