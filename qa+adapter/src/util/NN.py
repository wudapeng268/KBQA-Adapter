from __future__ import absolute_import

import tensorflow as tf
# from tensorflow.contrib import rnn
from tensorflow import nn
from tensorflow.python.util import nest  # pylint: disable=E0611

import pdb
rnn=tf.nn.rnn_cell

def BiGRU(x, x_len, n_hidden, biRnnScopeName):

    gru_fw_cell = rnn.GRUCell(n_hidden)

    gru_bw_cell = rnn.GRUCell(n_hidden)

    outputs, output_states = nn.bidirectional_dynamic_rnn(gru_fw_cell, gru_bw_cell, x, dtype=tf.float32,
                                                          sequence_length=x_len,
                                                          scope=biRnnScopeName)
    return outputs, output_states


def BiLSTM(x, x_len, n_hidden, biRnnScopeName):


    lstm_fw_cell = rnn.LSTMCell(n_hidden)

    lstm_bw_cell = rnn.LSTMCell(n_hidden)


    outputs, output_states = nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32,
                                                          sequence_length=x_len,
                                                          scope=biRnnScopeName)
    #output_states is a tuple (fw,bw), fw contain (c,h) bw too.
    return outputs, output_states


def bi_gru(x, x_lens, n_hidden, name, keep_prob,istraining):
    output1, output1_state = BiGRU(x, x_lens, n_hidden, name + "_1")
    middle1 = tf.concat(output1, 2)
    middle1 = tf.layers.dropout(middle1, 1.-keep_prob,istraining)
    output2,output2_state = BiGRU(middle1, x_lens, n_hidden, name + "_2")
    output2 = tuple_dropout(output2,1.-keep_prob,istraining)
    output2_state = tuple_dropout(output2_state,1.-keep_prob,istraining)

    return output2,output2_state

def bi_lstm(x, x_lens, n_hidden, name):
    output1, output1_state = BiLSTM(x, x_lens, n_hidden, name)
    # pdb.set_trace()
    output1_state = (output1_state[0][0], output1_state[1][0])
    return output1,output1_state

def tuple_dropout(x,prob,istraining):
    x1,x2=x
    x1=tf.layers.dropout(x1,prob,istraining)
    x2=tf.layers.dropout(x2,prob,istraining)
    x=(x1,x2)
    return x

def short_conn_bi_gru(x, x_lens, n_hidden, name, keep_prob, is_training):
    output1, output1_state = BiGRU(x, x_lens, n_hidden, name + "_1")
    middle1 = tf.concat(output1, 2)
    middle1 = tf.layers.dropout(middle1, 1 - keep_prob, is_training)
    output2, output2_state = BiGRU(middle1, x_lens, n_hidden, name + "_2")

    return (output1[0] + output2[0],output1[1] + output2[1]), (output1_state[0] + output2_state[0],output1_state[1] + output2_state[1])

def short_conn_bi_lstm(x, x_lens, n_hidden, name,share_name,config=None):
    '''
    release first add then max res-Net
    :param x:
    :param x_lens:
    :param n_hidden:
    :param name:
    :param share_num:
    :param keep_prob:
    :param is_training:
    :return: query4relation embedding
    '''

    with tf.variable_scope("get_realtion_vec_network", reuse=tf.AUTO_REUSE):
        output1, output1_state = BiLSTM(x, x_lens, n_hidden, share_name)
    middle1 = tf.concat(output1, 2)

    output2, output2_state = BiLSTM(middle1, x_lens, n_hidden, name+"_2")

    output1=tf.concat(output1,-1)
    output2=tf.concat(output2,-1)
    if 'single_lstm' in config['model'] and config['model']['single_lstm']:
        # pass
        query4relation = output1
        print("single layer")
    else:
        query4relation = output2+output1
    # query4relation = tf.reduce_max(query4relation, 1)
    return query4relation,output1,output2


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
def linear(args,
           output_size,
           bias=True,
           bias_initializer=None,
           kernel_initializer=None,
           name=""):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D/3D Tensor or a list of 2D/3D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      kernel_initializer: starting value to initialize the weight.
    Returns:
      A 2D/3D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    kernel_name = name + _WEIGHTS_VARIABLE_NAME
    bias_name = name + _BIAS_VARIABLE_NAME
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):  # added by Chengqi
        ## capable for 3D tensor
        shape = args.get_shape()
        if shape.ndims > 2:
            scope = tf.get_variable_scope()
            with tf.variable_scope(scope) as outer_scope:
                weights = tf.get_variable(
                    kernel_name, [shape[-1].value, output_size],
                    dtype=args.dtype,
                    initializer=kernel_initializer)
                res = tf.tensordot(args, weights, [[shape.ndims - 1], [0]])
                res.set_shape([args.get_shape()[0].value,args.get_shape()[1].value,output_size])

                if not bias:
                    return res
                with tf.variable_scope(outer_scope) as inner_scope:
                    inner_scope.set_partitioner(None)
                    if bias_initializer is None:
                        bias_initializer = tf.constant_initializer(0.0, dtype=args.dtype)
                    biases = tf.get_variable(
                        bias_name, [output_size],
                        dtype=args.dtype,
                        initializer=bias_initializer)
                return tf.nn.bias_add(res, biases)
    total_arg_size = args.get_shape()[-1].value

    # total_arg_size = args.get_shape()[-1].value
    dtype = args.dtype
    # Now the computation.
    scope = tf.get_variable_scope()
    # args = [args]
    with tf.variable_scope(scope) as outer_scope:
        weights = tf.get_variable(
            kernel_name, [total_arg_size, output_size],
            dtype=dtype,
            initializer=kernel_initializer)

        # if len(args) == 1:
        res = tf.matmul(args, weights)
        # else:
        #     res = tf.matmul(tf.concat(args, 1), weights)
        if not bias:
            return res
        with tf.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            if bias_initializer is None:
                bias_initializer = tf.constant_initializer(0.0, dtype=dtype)
            biases = tf.get_variable(
                bias_name, [output_size],
                dtype=dtype,
                initializer=bias_initializer)
        return tf.nn.bias_add(res, biases)

def dropout_wrapper(x, keep_prob, seed=None):
    """ A wrapper function for `tf.nn.dropout`
    Args:
        x: A Tensor.
        keep_prob: A float, the probability that each
          element is kept.
        seed: A Python integer. Used to create random seeds.
    Returns: A `tf.Tensor` of the same shape of `x`.
    """
    if keep_prob < 1.0:
        return tf.nn.dropout(x, keep_prob=keep_prob, seed=seed)
    return x

def fflayer(inputs,
            output_size,
            activation=None,
            dropout_input_keep_prob=1.0,
            dropout_seed=None,
            bias=True,
            kernel_initializer=None,
            bias_initializer=None,
            name=None):
    """ Applies feed forward transform for a 2-d matrix or
     3-d tensor.
    Args:
        inputs: A Tensor of 2-d or 3-d, [..., dim]
        output_size: An integer.
        handle: A Tensor. If provided, use it as the weight matrix.
        activation: The activation function.
        dropout_input_keep_prob: A float, the probability that each
          element in `inputs` is kept.
        dropout_seed: A Python integer. Used to create random seeds.
        bias: Whether to add a bias vector.
        kernel_initializer: The initializer of kernel weight.
        bias_initializer: The initializer of bias vector.
        name: A string.
    Returns: A Tensor of shape [..., `output_size`]
    """
    scope = tf.get_variable_scope()
    with tf.variable_scope(name or scope):
        inputs = dropout_wrapper(inputs, keep_prob=dropout_input_keep_prob,seed=dropout_seed)

        preact = linear(inputs, output_size=output_size,bias=bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer)

    if activation is None:
        return preact
    return activation(preact)