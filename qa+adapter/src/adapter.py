import tensorflow as tf
from src.util import NN


class adapter:

    def __init__(self,input,target,keep_prob,gather_index,name):
        '''
        :param input: a tensor
        :param target: a tensor
        :param keep_prob: dropout layer in adapter
        :param gather_index: the seen index of input
        :param name: relation or word
        '''
        self.input=input
        self.target = target
        self.keep_prob=keep_prob
        self.gather_index=gather_index
        self.name=name
        self.write_log = False

        self.target_size = self.target.shape[-1]


    def adapter_nn(self,input,target_size,name,method):
        '''
        build adapter nn
        :param input:
        :param target_size: output size
        :param name:
        :param method: forward_method see README.md
        :return:
        '''
        linear = method.split("-")[0]
        layer_num = int(method.split("-")[1])
        layer_norm = method.split("-")[2]
        print("{} adapter forward method: {} layer {} with {}".format(self.name,layer_num,linear,layer_norm))
        if linear in ["relu","tanh"]:
            if linear=="relu":
                adapter_activate=tf.nn.relu
            elif linear=="tanh":
                adapter_activate=tf.nn.tanh
            else:
                print("config error in model.{}.activate".format(self.name))
                exit(-1)
            print("add {} non linear".format(self.name))
            adapter_output = NN.fflayer(input, target_size,bias=False, activation=adapter_activate, name="trains_0"+name)


            if layer_norm=="layer_norm":
                adapter_output = tf.contrib.layers.layer_norm(adapter_output)
            if layer_num >= 2:
                for i in range(layer_num-1):
                    adapter_output = NN.fflayer(adapter_output,target_size, activation=adapter_activate, name="trans_{}".format(i+1)+name)
                    if layer_norm=="layer_norm":
                        adapter_output = tf.contrib.layers.layer_norm(adapter_output)

        elif linear=="linear":
            adapter_output = NN.fflayer(input,target_size, name="trans_0"+name)
        else:
            print("the first parameter in forward method error!")
            exit(1)
        return adapter_output

    def forward(self,method):
        '''
        :param method: forward_method see README.md
        :return:
        '''
        self.forward_method=method
        self.adapter_output = self.adapter_nn(self.input,self.target_size,self.name,method)

    def dual_learning(self,dual_loss_alpha):
        '''
        dual learning for training adapter
        :return:
        '''
        print("#####add mse back!########")
        back_size = self.input.shape[-1]
        back_output = self.adapter_nn(self.adapter_output, back_size, self.name + "_back",self.forward_method)
        back_loss = tf.reduce_mean((back_output - self.input) ** 2, 1)
        self.back_loss = dual_loss_alpha*tf.reduce_mean(back_loss, 0)



    def D(self,target,adapter_output,with_sigmoid,add_last=None):
        '''
        release code!
        Discrimeter for adapter
        :param target: pseudo target emb
        :param adapter_output: adapter output
        :param with_sigmoid: use sigmoid our paper is false
        :return:
        '''
        row,col = target.shape
        all = tf.concat([target,adapter_output],0)
        # linear!
        if add_last!=None:
            o=NN.fflayer(all,col,name="D_adapter"+add_last)
            s = NN.fflayer(o,1,name="D_final"+add_last)
        else:
            o = NN.fflayer(all, col, name="D_adapter")
            s = NN.fflayer(o,1,name="D_final")

        if with_sigmoid:
            s=tf.sigmoid(s)
        output_target= s[:row,:]
        output_adapter = s[row:,:]
        return output_target,output_adapter

    

    def define_loss(self,is_training,method,dual_loss_alpha=1):
        '''

        :param is_training:
        :param method: loss_method see README.md
        :return:
        '''
        if "dual" in method:
            self.dual_function = method.split("-")[-1]
            self.dual_learning(dual_loss_alpha)
        self.loss_method=method

        mm= method.split("-")[0]

        # loss
        if mm == "mse" :
            if self.keep_prob != 1:
                self.adapter_output = tf.layers.dropout(self.adapter_output, 1.0 - self.keep_prob,training=is_training)
            temp = tf.reduce_mean((self.adapter_output - self.target) ** 2, 1)
            temp = tf.gather(temp, self.gather_index)

            adapter_loss = tf.reduce_mean(temp)
            self.adapter_loss=adapter_loss
            

        elif mm == "gan":
            with tf.variable_scope("Discrimeter" + self.name):
                o_target, o_adapter = self.D(self.target, self.adapter_output, False)
            o_target = tf.gather(o_target, self.gather_index)
            o_adapter = tf.gather(o_adapter, self.gather_index)
            self.loss_D = tf.reduce_mean(-(o_target) + (o_adapter), 0)
            self.loss_G = tf.reduce_mean(-(o_adapter))

            error = (tf.reduce_mean(tf.ones_like(o_target)-tf.sigmoid(o_target), 0) + tf.reduce_mean(
                tf.sigmoid(o_adapter) - tf.zeros_like(o_adapter), 0)) / 2
            self.D_acc = 1 - error

        else:
            print("loss method is error only accept mse dual gan ")

    def define_train_op(self,method,learning_rate,global_step,clip_c,target_name):

        '''

        :param method: train_method see README.md
        :param learning_rate:
        :param global_step:
        :param clip_c: c in wgan
        :param target_name: target name in graph
        :return:
        '''
        self.train_method = method
        with tf.variable_scope("adapter"):
            if method!="None":
                adapter_para = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)

                key = None
                for t in adapter_para:
                    if t.name == "{}:0".format(target_name):
                        key = t
                adapter_para.remove(key)

                if "dual" in self.loss_method:
                    self.back_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='back_op').minimize(self.back_loss)


                mm = self.loss_method.split("-")[0]
                if "gan" in mm and clip_c==0:
                    raise Exception("use gan in {} but clip_c=0".format(self.name))

                if mm == "mse":
                    if method == "sgd":
                        self.adapter_op = tf.train.GradientDescentOptimizer(
                            learning_rate=learning_rate/ (1 + 0.0001 * global_step),
                            name="optimizer_map").minimize(self.adapter_loss,
                                                           var_list=adapter_para)
                    elif method== "RMSProp":
                        self.adapter_op = tf.train.RMSPropOptimizer(
                            learning_rate=learning_rate / (1 + 0.0001 *global_step),
                            name="optimizer_map").minimize(self.adapter_loss,
                                                           var_list=adapter_para)
                    elif method== "adam":
                        self.adapter_op = tf.train.AdamOptimizer(
                            learning_rate=learning_rate / (1 + 0.0001 * global_step),
                            name="optimizer_map").minimize(self.adapter_loss,
                                                           var_list=adapter_para)
                    else:
                        print("error config in run_op.adapter_op")
                        exit(-1)
                    
                elif mm == "gan":

                    self.D_op = tf.train.RMSPropOptimizer(
                        learning_rate=learning_rate,
                        name="D_op").minimize(self.loss_D, var_list=adapter_para)
                    # import pdb;pdb.set_trace()
                    self.clip = [var.assign(tf.clip_by_value(var, -float(clip_c), float(clip_c))) for var in tf.global_variables() if "Discrimeter" in var.name]

                    self.G_op = tf.train.RMSPropOptimizer(
                        learning_rate=learning_rate,
                        name="G_op").minimize(self.loss_G, var_list=adapter_para)
                else:
                    print("config model._adapter.loss_method error")
                    exit(-1)

    def run(self,train_unseen,model,config,feed):
        '''
        :param train_unseen: flag of train adapter or not
        :param model:
        :param config:
        :param feed: feed_dict
        :return:
        '''
        if train_unseen and self.train_method!=None:
            mm = self.loss_method.split("-")[0]
            if mm == "mse":
                _, adapter_loss = model.sess.run(
                    [self.adapter_op, self.adapter_loss],
                    feed_dict=feed)
                if self.write_log:
                    model.writer.add_summary("train/{}_adapter_loss".format(self.name), adapter_loss, model.global_step * 1.0)

            elif mm == "gan" :
                if 'D_iter' not in config['model']:
                    config['model']['D_iter'] = 1
                    config['model']['G_iter'] = 1

                if config['model']['D_iter'] < 1:
                    tt = config['model']['D_iter']
                    config['model']['D_iter'] = 1
                    config['model']['G_iter'] = int(1.0 / tt)

                for _ in range(config['model']['D_iter']):
                    _, _, D_loss, D_acc = model.sess.run(
                        [self.D_op, self.clip, self.loss_D,
                         self.D_acc], feed_dict=feed)
                # print("D_acc: {}".format(D_acc))
                if self.write_log:
                    model.writer.add_summary("train/{}_D_loss".format(self.name), D_loss,model.global_step * 1.0)
                    model.writer.add_summary("train/{}_D_acc".format(self.name), D_acc,model.global_step * 1.0)
                for _ in range(config['model']['G_iter']):
                    _, G_loss = model.sess.run([self.G_op, self.loss_G],feed_dict=feed)
                if self.write_log:
                    model.writer.add_summary("train/{}_G_loss".format(self.name), G_loss,model.global_step * 1.0)

            else:
                print("config model._adapter.loss_method error")
                exit(-1)

            if "dual" in self.loss_method:
                _, back_loss = model.sess.run([self.back_op, self.back_loss], feed_dict=feed)
                if self.write_log:
                    model.writer.add_summary("train/{}_back_loss".format(self.name), back_loss,model.global_step * 1.0)
