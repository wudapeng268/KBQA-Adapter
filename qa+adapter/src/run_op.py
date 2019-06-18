# coding:utf-8
# author:wup
# description: 
# e-mail:
# date: 
import os
import pickle as pkl
import random
import time

import numpy as np
import tensorflow as tf

import src.util.result_util as result_util
from src.summary_add import SummaryWriter
from src.util import FileUtil

write_log = False
import pdb

def cal_macro_acc(output):
    '''
    :param output: str gold_relation\t predict_relation
    :return: relation score
    '''
    rel_map = {}
    rel_correct = {}

    for t in output:
        r = t.split("\t")
        gold = r[0]
        pre = r[1]
        if gold in rel_map:
            rel_map[gold] += 1
        else:
            rel_map[gold] = 1
        if pre == gold:
            if gold in rel_correct:
                rel_correct[gold] += 1
            else:
                rel_correct[gold] = 1
    true_relation_score = 0
    for rel in rel_map:
        all_num = rel_map[rel]
        if rel in rel_correct:
            true_num = rel_correct[rel]
            true_relation_score += true_num * 1.0 / all_num
    if len(rel_map)==0:
        return 0
    true_relation_score /= len(rel_map)
    return true_relation_score

def cal_macro_acc_seen_rate(output):
    '''
    :param output: str gold_relation\t state(1/0: predict seen or not)
    :return: relation score
    '''
    rel_map = {}
    rel_state = {}

    for t in output:
        r = t.split("\t")
        gold = r[0]
        state = int(r[1])
        if gold in rel_map:
            rel_map[gold] += 1
        else:
            rel_map[gold] = 1

        if gold in rel_state:
            rel_state[gold] += state
        else:
            rel_state[gold] = state
    true_relation_score = 0
    for rel in rel_map:
        all_num = rel_map[rel]
        if rel in rel_state:
            true_num = rel_state[rel]
            true_relation_score += true_num * 1.0 / all_num
    if len(rel_map)==0:
        return 0
    true_relation_score /= len(rel_map)
    return true_relation_score


def dev(model, config, current_epoch):
    step = 1
    acc = 0
    qid = 0
    seen_acc = 0
    seen_num = 0
    unseen_acc = 0

    seen_relation_output = []
    unseen_relation_output = []
    pred4seen_acc = 0
    pred4seen_seen_acc = 0
    pred4seen_unseen_acc = 0
    model.qa.itemIndexDev = 0

    while (step - 1) * model.dev_batch_size < model.deving_iters:
        model.dev_global_step += 1
        value = model.qa.load_test_data(
            model.dev_batch_size, "dev")

        feed = {model.question_ids: value['batch_x_anonymous'],
                model.relation_index: value['batch_relation_index'],
                model.relation_lens: value['batch_relation_lens'],
                model.x_lens: value['batch_x_anonymous_lens'],
                model.is_training: False, }

        score_loss_test, predict_relation, rel_score, rel_pred4seen = model.sess.run(
            [model.score_loss_test, model.rel_pred, model.rel_score, model.rel_pred4seen],
            feed_dict=feed)

        if write_log:
            model.writer.add_summary("dev/score_loss", score_loss_test,
                                     model.dev_global_step * 1.0)

        for i in range(value['batch_size']):
            temp_gold_relation = value['gold_relation'][i]

            if temp_gold_relation in model.train_relation:
                seen_num += 1

            if predict_relation[i] >= len(value['cand_rel_list'][i]):
                qid += 1
                continue

            ans_relation_id = value['cand_rel_list'][i][predict_relation[i]]
            rel_pred4seen_id = value['cand_rel_list'][i][rel_pred4seen[i]]

            if i >= len(value['questions']):
                continue
            if temp_gold_relation in model.train_relation:
                seen_relation_output.append("{}\t{}".format(temp_gold_relation, ans_relation_id))
            else:
                unseen_relation_output.append("{}\t{}".format(temp_gold_relation, ans_relation_id))

            if temp_gold_relation == ans_relation_id:
                acc += 1
                if temp_gold_relation in model.train_relation:
                    seen_acc += 1
                else:
                    unseen_acc += 1

            if temp_gold_relation == rel_pred4seen_id:
                pred4seen_acc += 1
                if temp_gold_relation in model.train_relation:
                    pred4seen_seen_acc += 1
                else:
                    pred4seen_unseen_acc += 1

            qid += 1
        step += 1

    acc = (acc * 1.0 / qid)

    if seen_num == 0:
        seen_acc = 0
    else:
        pred4seen_seen_acc = pred4seen_seen_acc * 1. / seen_num
        seen_acc = seen_acc * 1.0 / seen_num

    pred4seen_unseen_acc = pred4seen_unseen_acc * 1. / (qid - seen_num)
    pred4seen_acc = pred4seen_acc * 1.0 / qid
    unseen_acc = unseen_acc * 1.0 / (qid - seen_num)
    seen_macro = cal_macro_acc(seen_relation_output)
    unseen_macro = cal_macro_acc(unseen_relation_output)
    all_macro = cal_macro_acc(seen_relation_output+unseen_relation_output)
    if write_log:
        model.writer.add_summary("dev/all_mapping_acc", pred4seen_acc, current_epoch)
        model.writer.add_summary("dev/all_mapping_seen_acc", pred4seen_seen_acc, current_epoch)
        model.writer.add_summary("dev/all_mapping_unseen_acc", pred4seen_unseen_acc, current_epoch)

    return acc, seen_acc, unseen_acc, all_macro


def train_relation_flag(model):
    flags = np.zeros(model.qa.relation_size)
    for t in model.qa.train_data:
        flags[int(t.relation)] = 1
    return flags


need_all = 0  # after tune unseen train seen and unseen


class dev_struct():
    best_acc = -1
    patience = 10


def run_dev2(model, config, current_step, all_dev_stage, current_stage, all_dev_struct, seen_dev_struct,
             unseen_dev_struct):
    new_acc, seen_acc, unseen_acc, all_macro = dev(model, config, current_step)
    if write_log:
        model.writer.add_summary("dev/acc", new_acc, current_step)
        model.writer.add_summary("dev/seen_acc", seen_acc, current_step)
        model.writer.add_summary("dev/unseen_acc", unseen_acc, current_step)
        model.writer.add_summary("dev/true relation score", all_macro, current_step)

    tf.logging.info("step:\t%d,new_acc_relation:\t%f" % (current_step, new_acc))
    tf.logging.info("step:\t%d,seen acc in dev:\t%f" % (current_step, seen_acc))
    tf.logging.info("step:\t%d,unseen acc in dev:\t%f" % (current_step, unseen_acc))
    tf.logging.info("step:\t%d,true relaiton score in dev:\t%f" % (current_stage,all_macro))

    if all_dev_stage[current_stage] == "all":
        if new_acc > all_dev_struct.best_acc:
            all_dev_struct.best_acc = new_acc
            save_model(model, config, current_step)
            all_dev_struct.patience = 10
        else:
            all_dev_struct.patience -= 1
        tf.logging.info("all dev struct patience: {}".format(all_dev_struct.patience))

    elif all_dev_stage[current_stage] == "seen":
        if seen_acc > seen_dev_struct.best_acc:
            seen_dev_struct.best_acc = seen_acc
            save_model(model, config, current_step)
            seen_dev_struct.patience = 10
        else:
            seen_dev_struct.patience -= 1
        tf.logging.info("seen dev struct patience: {}".format(seen_dev_struct.patience))

    elif all_dev_stage[current_stage] == "unseen":
        tf.logging.info("old best unseen acc {} now acc {}".format(unseen_dev_struct.best_acc, unseen_acc))
        if unseen_acc > unseen_dev_struct.best_acc:
            unseen_dev_struct.best_acc = unseen_acc
            save_model(model, config, current_step)
            unseen_dev_struct.patience = 10
        else:
            unseen_dev_struct.patience -= 1
        tf.logging.info("unseen dev struct patience: {}".format(unseen_dev_struct.patience))

    return all_dev_struct, seen_dev_struct, unseen_dev_struct


def train(model, config):
    # dev struct
    all_train_stage = config['run_op']['all_train_stage'].split("-")
    all_dev_stage = config['run_op']['all_dev_stage'].split("-")
    current_stage = 0
    all_data_dev_struct = dev_struct()
    seen_data_dev_struct = dev_struct()
    unseen_data_dev_struct = dev_struct()


    #check config
    if len(all_train_stage) != len(all_dev_stage):
        tf.logging.info("train stage not equal with dev stage")
        exit(1)

    if config['run_op']['fix_random_seed']:
        random.seed(1234)

    if 'D_iter' in config['model']:
        tf.logging.info("D_iter {}".format(config['model']['D_iter']))

    model.global_step = 0
    model.dev_global_step = 0

    # op
    if config['run_op']['optimizer'] == "adam":
        score_optimizer = tf.train.AdamOptimizer(
            learning_rate=config['run_op']['learning_rate'] / (1 + 0.0001 * model.global_step),
            name="rel_optimer").minimize(model.score_loss)
    elif config['run_op']['optimizer'] == "sgd":
        score_optimizer = tf.train.GradientDescentOptimizer(learning_rate=config['run_op']['learning_rate'],name="rel_optimer").minimize(model.score_loss)

    

    if hasattr(model, "relation_adapter") and config['model']['relation_adapter']['train_method'] != None:
        if "clip_c" in config['model']['relation_adapter']:
            clip_c = config['model']['relation_adapter']['clip_c']
        else:
            clip_c = 0
        model.relation_adapter.define_train_op(config['model']['relation_adapter']['train_method'],config['run_op']['adapter_lr'], model.global_step, clip_c,
                                               "relation_part_embedding")
    
    adapter_output = np.zeros(model.qa.word_embedding.shape)
    temp_word = tf.placeholder(tf.float32,model.word_embedding.shape,name="word_emb_placeholder")
    assign_op = tf.assign(model.word_embedding,temp_word)

    init = tf.global_variables_initializer()
    model.sess.run(init)

    # begin train!
    current_epoch = 0

    early_stop = False
    tf.logging.info("early_stop {}".format(config['run_op']['early_stop']))


    def check_load_name(name):
        not_load_name = ["op", "Discrimeter", "back", "beta1_power", "beta2_power", "optimer", "adapter"]

        for t in not_load_name:
            if t in name:
                return False
        return True

    if write_log:
        model.writer = SummaryWriter(model.log_path)

    # restore
    if config['run_op']['continue_train']:
        variables = tf.contrib.slim.get_variables_to_restore()
        variables_to_restore = [v for v in variables if check_load_name(v.name)]
        saver_restore = tf.train.Saver(variables_to_restore)
        saver_restore.restore(model.sess, config['run_op']['restore_file'])
        model.global_step = int(config['run_op']['restore_file'].split('-')[-1]) + 1
        current_epoch = int(model.global_step / (model.training_iters / model.fix_batch_size))

        model.dev_global_step = model.global_step

        tf.logging.info("restor trained seen model in {}".format(config['run_op']['restore_file']))
        tt_acc, tt_seen_acc, tt_unseen_acc, _ = dev(model, config, current_epoch)
        tf.logging.info("epoch: {},previous acc in dev: {}, seen acc: {}, unseen acc: {} ".format(current_epoch - 1, tt_acc,tt_seen_acc, tt_unseen_acc))

    else:
        tf.logging.info("new train!")
        


    while not early_stop:

        ss = time.time()

        model.qa.itemIndexTrain = 0
        model.qa.itemIndexTest = 0
        model.qa.itemIndexDev = 0

        random.shuffle(model.qa.train_data)
        step = 1
        mm = time.time()

        while (step - 1) * model.fix_batch_size < model.training_iters:

            model.global_step += 1
            value = model.qa.load_train_data(model.fix_batch_size)

            feed = {model.question_ids: value['batch_x_anonymous'],
                    model.relation_index: value['batch_relation_index'],
                    model.x_lens: value['batch_x_anonymous_lens'],
                    model.is_training: True, }

            if all_train_stage[current_stage] == "all" or all_train_stage[current_stage] == "seen":
                train_seen = True
            else:
                train_seen = False
            if all_train_stage[current_stage] == "all" or all_train_stage[current_stage] == "unseen":
                train_unseen = True
            else:
                train_unseen = False

            if train_seen:
                _, score_loss = model.sess.run(
                    [score_optimizer, model.score_loss],
                    feed_dict=feed)
                if write_log:
                    model.writer.add_summary("train/score_loss", score_loss,
                                             model.global_step * 1.0)

            # run adapter
            
            if hasattr(model, "relation_adapter"):
                if train_seen and "unseen_loop" in config['data']:
                    unseen_loop = config['data']['unseen_loop']
                else:
                    unseen_loop = 1
                for _ in range(unseen_loop):
                    model.relation_adapter.run(train_unseen, model, config, feed)

            if (step - 1) % model.display_step == 0:
                if train_seen:
                    tf.logging.info("=======now train seen!=======")
                if train_unseen:
                    tf.logging.info("=======now train unseen!=========")
                tf.logging.info("epoch:\t%d,rate:\t%d/%d,time:\t%f" % (
                    current_epoch, step, model.training_iters // model.fix_batch_size, time.time() - mm))
                mm = time.time()

            step += 1

        if current_stage >= len(all_train_stage) and config['run_op']['early_stop']:
            break
        if current_epoch < 5 or current_epoch % config['run_op']['dev_epoch'] == 0 and config['run_op']['early_stop']:
            all_data_dev_struct, seen_data_dev_struct, unseen_data_dev_struct = run_dev2(model, config,model.global_step, all_dev_stage, current_stage,all_data_dev_struct,seen_data_dev_struct,unseen_data_dev_struct)

            if all_data_dev_struct.patience == 0 or \
                            seen_data_dev_struct.patience == 0 or \
                            unseen_data_dev_struct.patience == 0:

                tf.logging.info("restore model!")
                model_path = model.model_path
                model.saver.restore(model.sess, tf.train.latest_checkpoint(model_path))
                model.global_step = int(tf.train.latest_checkpoint(model_path).split('-')[-1])

                #dev it
                tt_acc, tt_seen_acc, tt_unseen_acc, _ = dev(model, config, model.global_step)
                tf.logging.info(
                    "train stage {}\n dev_stage {} \n best all acc {}\n best seen acc {}\n best unseen acc {}\n".format(
                        all_train_stage[current_stage], all_dev_stage[current_stage], tt_acc, tt_seen_acc,
                        tt_unseen_acc))

                tf.logging.info("clear stage!")
                all_data_dev_struct = dev_struct()
                seen_data_dev_struct = dev_struct()
                unseen_data_dev_struct = dev_struct()
                current_stage += 1

                tf.logging.info("run a test!")
                test(model, config, True, False)

            if current_stage >= len(all_train_stage) and config['run_op']['early_stop']:
                break

        ee = time.time()
        tf.logging.info("epoch:\t%d,time:\t%f" % (current_epoch, ee - ss))
        current_epoch += 1

    tf.logging.info("Finish train")
    tf.logging.info("Start test")
    test(model, config, True, True)


def save_model(model, config, e):
    mm = time.time()
    fold_path = os.path.join(model.model_path)

    model_path = os.path.join(fold_path, "model.ckpt")
    save_path = model.saver.save(model.sess, model_path, e)
    tf.logging.info("Model saved in file: %s" % save_path)
    mm2 = time.time()
    tf.logging.info("Save model time: %f" % (mm2 - mm))



def test(model, config, final_test, write_file, kbqa_flag=False):

    model_location = None
    if final_test:
        init = tf.global_variables_initializer()
        model.sess.run(init)

        model_location = tf.train.latest_checkpoint((model.model_path))
        tf.logging.info("restore model {}".format(model_location))
        model.saver.restore(model.sess, model_location)

    # begin test!
    step = 1
    relation_detection_output = []
    qid = 0
    acc = 0

    model.qa.itemIndexTest = 0
    unseen_relation_output = []
    unseen_all = 0
    unseen_acc = 0

    relation_output = []

    word_big_part = 0
    word_small_part = 0
    word_equal_part = 0
    all_para_num = 0

    choose_seen_error = 0
    choose_unseen_error = 0

    kbqa_acc = 0
    unseen_kbqa_acc = 0

    
    seen_macro_output=[]
    unseen_macro_output=[]

    macro_output_for_seen_rate=[]
    
    while (step - 1) * model.dev_batch_size < model.testing_iters:

        ss = time.time()
        if kbqa_flag:
            value = model.qa.load_test_data(
                model.dev_batch_size, "test", kbqa_flag)
        else:
            value = model.qa.load_test_data(
                model.dev_batch_size)

        feed = {model.question_ids: value['batch_x_anonymous'],
                model.relation_index: value['batch_relation_index'],
                model.relation_lens: value['batch_relation_lens'],
                model.x_lens: value['batch_x_anonymous_lens'],
                model.is_training: False, }

        wo_cand_rel, score, rel_pred4seen, rel_word_vec, rel_part_vec = model.sess.run(
            [model.rel_pred, model.rel_score, model.rel_pred4seen, model.word_test, model.part_test],
            feed_dict=feed)

        rel_word_vec = np.max(rel_word_vec, 1)
        row, col = rel_word_vec.shape
        all_para_num += row * col
        rel_part_vec = np.max(rel_part_vec, 1)
        word_big_part += np.sum(rel_word_vec > rel_part_vec)
        word_small_part += np.sum(rel_word_vec < rel_part_vec)
        word_equal_part += np.sum(rel_word_vec == rel_part_vec)

        for i in range(value['batch_size']):
            temp_gold_relation = value['gold_relation'][i]
            
            if wo_cand_rel[i] >= len(value['cand_rel_list'][i]):
                qid += 1
                relation_output.append("{}\t{}".format(temp_gold_relation, "oov"))
                continue

            qid += 1
            pre = value['cand_rel_list'][i][wo_cand_rel[i]]
            current_query = value['questions'][i]
            out_str = "{}\t{}\t{}\t{}".format(
                value['qids'][i], current_query,
                model.qa.rel_voc[pre],
                model.qa.rel_voc[temp_gold_relation])

            relation_detection_output.append(out_str)
            unseen = False

            if temp_gold_relation not in model.train_relation:
                unseen_all += 1
                unseen = True
                unseen_relation_output.append(out_str)

            if seen_word_flag and unseen and current_query in query4seen_word:
                if pre == temp_gold_relation:
                    seen_word_acc += 1
            if seen_word_flag and unseen and current_query in query4unseen_word:
                if pre == temp_gold_relation:
                    unseen_word_acc += 1

            if temp_gold_relation in model.train_relation:
                seen_macro_output.append("{}\t{}".format(temp_gold_relation, pre))
            else:
                unseen_macro_output.append("{}\t{}".format(temp_gold_relation, pre))


            
            if unseen and pre in model.train_relation:
                choose_seen_error += 1

            if unseen:
                if pre in model.train_relation:
                    macro_output_for_seen_rate.append("{}\t{}".format(temp_gold_relation,1))
                else:
                    macro_output_for_seen_rate.append("{}\t{}".format(temp_gold_relation,0))

            if (not unseen) and (pre not in model.train_relation):
                choose_unseen_error += 1
            if temp_gold_relation == pre:
                acc += 1
                if unseen:
                    unseen_acc += 1
            else:
                pass

            if kbqa_flag:
                gold_subject = value['gold_subject'][i]
                if temp_gold_relation == pre and gold_subject in model.qa.subject2relation and  pre in model.qa.subject2relation[gold_subject]:
                    kbqa_acc += 1
                    if unseen:
                        unseen_kbqa_acc += 1

        if (step - 1) % model.display_step == 0:
            tf.logging.info("rate:\t%d/%d" % (step, (model.testing_iters / model.dev_batch_size)))
            ee = time.time()
            tf.logging.info("time:\t" + str(ee - ss))

        step += 1
    unseen_error = unseen_all - unseen_acc
    seen_all = qid - unseen_all
    seen_error = (qid - unseen_all) - (acc - unseen_acc)
    assert all_para_num == (word_big_part + word_small_part + word_equal_part)

    # prepare result
    output_dic = {}
    output_columns = ["model_name", "test_seen","seen_macro_acc",
                      "test_unseen","unseen_macro_acc",
                      "all","all_macro_acc", "location",
                      "server", "kbqa_acc"
                      ,'unseen_choose_seen_macro'
                      , "unseen_relation_num", "time"]

    
    output_dic['unseen_choose_seen_macro'] = cal_macro_acc_seen_rate(macro_output_for_seen_rate)

    tf.logging.info("para composition big:equal:small = {}:{}:{}".format(word_big_part, word_equal_part, word_small_part))
    
    tf.logging.info("qid {} test_len {}".format(qid, len(model.qa.test_data)))
    assert qid == len(model.qa.test_data)

    now = time.asctime(time.localtime(time.time()))
    output_dic['time'] = (now)

    true_relation_score = cal_macro_acc(relation_output)
    output_dic['seen_macro_acc'] = cal_macro_acc(seen_macro_output)
    output_dic['unseen_macro_acc'] = cal_macro_acc(unseen_macro_output)
    output_dic['all_macro_acc'] = cal_macro_acc(seen_macro_output+unseen_macro_output)

    all_acc = acc * 1.0 / qid
    output_dic['all'] = all_acc

    kbqa_acc = kbqa_acc * 1. / qid
    output_dic['kbqa_acc'] = kbqa_acc

    output_dic['unseen_relation_num'] = unseen_all

    if qid - unseen_all == 0:
        seen_acc = 0
    else:
        seen_acc = (acc - unseen_acc) * 1.0 / (qid - unseen_all)
    output_dic['test_seen'] = seen_acc

    if unseen_all != 0:
        unseen_acc = unseen_acc * 1.0 / unseen_all
    output_dic['test_unseen'] = unseen_acc

    tf.logging.info("all {} seen {} unseen acc {}".format(all_acc,seen_acc,unseen_acc))

    output_dic['location'] = model_location

    model_name = config['model']['name']
    output_dic['model_name'] = model_name

    csv_path = "{}/result.csv".format(model.model_path)
    if final_test:
        result_util.write_result(output_dic, output_columns, csv_path)

    if write_file:
        FileUtil.writeFile(unseen_relation_output, "{}/unseen.output.txt".format(model.model_path))
        FileUtil.writeFile(relation_detection_output, "{}/all.output.txt".format(model.model_path))

    if not final_test:
        return acc * 1.0 / qid, seen_acc, unseen_acc, true_relation_score


