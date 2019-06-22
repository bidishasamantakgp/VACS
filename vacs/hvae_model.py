from __future__ import print_function, division, absolute_import
import tensorflow as tf
import numpy as np
import os
from utils import parameters
import utils.model as model
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import xavier_initializer
params = parameters.Parameters()

def rnn_placeholders(state):
    """Convert RNN state tensors to placeholders with the zero state as default."""
    if isinstance(state, tf.contrib.rnn.LSTMStateTuple):
        c, h = state
        c = tf.placeholder_with_default(c, c.shape, c.op.name)
        h = tf.placeholder_with_default(h, h.shape, h.op.name)
        return tf.contrib.rnn.LSTMStateTuple(c, h)
    elif isinstance(state, tf.Tensor):
        h = state
        h = tf.placeholder_with_default(h, h.shape, h.op.name)
        return h
    else:
        structure = [rnn_placeholders(x) for x in state]
        return tuple(structure)


##mu_ml and logvar_nl---- non linearity for mu and logvar 
def gauss_layer(inp, dim, mu_nl=None, logvar_nl=None, scope=None):
    """
    Gaussian layer
    Args:
        inp(tf.Tensor): input to Gaussian layer
        dim(int): dimension of output latent variables
        mu_nl(callable): nonlinearity for Gaussian mean
        logvar_nl(callable): nonlinearity for Gaussian log variance
        scope(str/VariableScope): tensorflow variable scope
    """
    with tf.variable_scope(scope, "gauss") as sc:
        mu = fully_connected(inp, dim, activation_fn=mu_nl,
                weights_initializer=xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                scope="mu")
        logvar = fully_connected(inp, dim, activation_fn=logvar_nl,
                weights_initializer=xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                scope="logvar")
        eps = tf.random_normal(tf.shape(logvar), name='eps',dtype=tf.float64)
        sample = mu + logvar * eps
    return mu, logvar, sample

def zglobal_encoder(label_input,zsent_sample,seq_len, batch_size):
    """
    Pre-stochastic layer encoder for z1 (latent segment variable)
    Args:
        x(tf.Tensor): tensor of shape (bs, T, F)
        z2(tf.Tensor): tensor of shape (bs, D1)
        rhus(list): list of numbers of LSTM layer hidden units
    Return:
        out(tf.Tensor): concatenation of hidden states of all LSTM layers
    """
    # prepare input

    # print("------------",label_input.shape)
    # print("********",zsent_sample.shape)
    # # zsent_sample=[zsent_sample]*(tf.shape(label_input)[1])
    # z_dash=tf.tile(zsent_sample,[tf.shape(label_input.shape)[1],1])
    # z_dash=tf.split(z_dash,tf.shape(label_input.shape)[1], axis=0)
    # zsent_sample_1=tf.stack(z_dash,axis=0)
    # # for i in range(int(label_input.shape[1])):
    # # zsent_sample_1=tf.stack(zsent_sample,axis=1)
    # l_zsent = tf.concat([label_input,zsent_sample_1],axis=-1) ##MIGHT NEED MODIFICATION
    # print(l_zsent.shape)
    # encoder_input=l_zsent

    bs, T = tf.shape(label_input)[0], tf.shape(label_input)[1]
    zsent_sample = tf.tile(tf.expand_dims(zsent_sample, 1), (1, T, 1))
    x_z2 = tf.concat([label_input, zsent_sample], axis=-1)
    encoder_input=x_z2

    if params.base_cell == 'lstm':
      base_cell = tf.contrib.rnn.LSTMCell
    elif params.base_cell == 'rnn':
      base_cell = tf.contrib.rnn.RNNCell
    else:
      base_cell = tf.contrib.rnn.GRUCell

    cell = model.make_rnn_cell([params.encoder_hidden for _ in range(
        params.decoder_rnn_layers)], base_cell=base_cell)

    #cell2=model.make_rnn_cell([params.decoder_hidden for _ in range(params.decoder_rnn_layers)], base_cell=base_cell)

    initial = cell.zero_state(batch_size, dtype=tf.float64)
    #initial2=cell.zero_state(batch_size, dtype=tf.float64)

    if params.keep_rate < 1:
        encoder_input = tf.nn.dropout(encoder_input, params.keep_rate)
        #encoder_input2=tf.nn.dropout(encoder_input2, params.keep_rate)        
    outputs, final_state = tf.nn.dynamic_rnn(cell,
                                             inputs=encoder_input,
                                             sequence_length=seq_len,
                                             initial_state=initial,
                                             swap_memory=True,
                                             dtype=tf.float64,
                                             scope="zglobal_encoder_rnn")
    final_state = tf.concat(final_state[0], 1)
    return final_state

def zsent_encoder(encoder_input, seq_len, batch_size):
    """
    Pre-stochastic layer encoder for z2 (latent sequence variable)
    Args:
        x(tf.Tensor): tensor of shape (bs, T, F)
        rhus(list): list of numbers of LSTM layer hidden units
    Return:
        out(tf.Tensor): concatenation of hidden states of all LSTM layers
    """
    # construct lstm
    # cell = tf.nn.rnn_cell.BasicLSTMCell(params.cell_hidden_size)
    # cells = tf.nn.rnn_cell.MultiRNNCell([cell]*params.rnn_layers)
    if params.base_cell == 'lstm':
      base_cell = tf.contrib.rnn.LSTMCell
    elif params.base_cell == 'rnn':
      base_cell = tf.contrib.rnn.RNNCell
    else:
      base_cell = tf.contrib.rnn.GRUCell

    cell = model.make_rnn_cell([params.encoder_hidden for _ in range(params.decoder_rnn_layers)], base_cell=base_cell)

    #cell2=model.make_rnn_cell([params.decoder_hidden for _ in range(params.decoder_rnn_layers)], base_cell=base_cell)

    initial = cell.zero_state(batch_size, dtype=tf.float64)
    #initial2=cell.zero_state(batch_size, dtype=tf.float64)

    if params.keep_rate < 1:
        encoder_input = tf.nn.dropout(encoder_input, params.keep_rate)
        #encoder_input2=tf.nn.dropout(encoder_input2, params.keep_rate)
    # print(encoder_input.shape)
    outputs, final_state = tf.nn.dynamic_rnn(cell,
                                             inputs=encoder_input,
                                             sequence_length=seq_len,
                                             initial_state=initial,
                                             swap_memory=True,
                                             dtype=tf.float64,
                                             scope="zsent_encoder_rnn")
    final_state = tf.concat(final_state[0], 1)
    return final_state



def encoder(encoder_input, label_input, seq_len, batch_size):
    with tf.variable_scope("encoder"):

        zsent_pre_out = zsent_encoder(encoder_input,seq_len,batch_size)
        zsent_mu, zsent_sigma, zsent_sample = gauss_layer(zsent_pre_out, params.latent_size, scope="zsent_enc_gauss")
        Zsent_distribution = [zsent_mu, zsent_sigma]

        zglobal_pre_out = zglobal_encoder(label_input, zsent_sample, seq_len,batch_size)
        zglobal_mu, zglobal_sigma, zglobal_sample = gauss_layer(zglobal_pre_out, params.latent_size, scope="zglobal_enc_gauss")
        Zglobal_distribition = [zglobal_mu, zglobal_sigma]
        
        # x_pre_out, px_z, x_sample = decoder(z1_sample, z2_sample, xout, x_rhus)
    return Zsent_distribution, zsent_sample, Zglobal_distribition, zglobal_sample
    # , px_z, x_sample

def lstm_decoder_labels(z, d_inputs, d_seq_l, batch_size, embed, vocab_size,gen_mode=False,scope=None):

    with tf.variable_scope(scope, "decoder") as sc:
        with tf.device("/cpu:0"):
            dec_inps = tf.nn.embedding_lookup(embed, d_inputs)
        # turn off dropout for generation:
        if params.dec_keep_rate < 1 and not gen_mode:
            dec_inps = tf.nn.dropout(dec_inps, params.dec_keep_rate)
        
        max_sl = tf.shape(dec_inps)[1]
        # define cell
        if params.base_cell == 'lstm':
          base_cell = tf.contrib.rnn.LSTMCell
        elif params.base_cell == 'rnn':
            base_cell = tf.contrib.rnn.RNNCell
        else:
          # not working for now
          base_cell = tf.contrib.rnn.GRUCell
        cell = model.make_rnn_cell([params.decoder_hidden for _ in range(params.decoder_rnn_layers)], base_cell=base_cell)
        
        if params.decode == 'hw':
            # Higway network [S.Sementiuta et.al]
            for i in range(params.highway_lc):
                with tf.variable_scope("hw_layer_dec{0}".format(i)) as scope:
                    z_dec = fully_connected(z, params.decoder_hidden * 2,activation_fn=tf.nn.sigmoid,
                            weights_initializer=xavier_initializer(),
                            biases_initializer=tf.zeros_initializer(),
                            scope="decoder_inp_state")

            inp_h, inp_c = tf.split(z_dec, 2, axis=1)
            initial_state = rnn_placeholders(
                (tf.contrib.rnn.LSTMStateTuple(inp_c, inp_h), ))
        elif params.decode == 'concat':
            z_out = tf.reshape(
              tf.tile(tf.expand_dims(z, 1), (1, max_sl, 1)),
              [batch_size, -1, params.latent_size])
            dec_inps = tf.concat([dec_inps, z_out], 2)
            initial_state = rnn_placeholders(
                cell.zero_state(tf.shape(dec_inps)[0], tf.float64))
        elif params.decode == 'mlp':
            # z->decoder initial state
            w1 = tf.get_variable('whl', [params.latent_size, params.highway_ls],
                                 tf.float64,
                                 initializer=tf.truncated_normal_initializer())
            b1 = tf.get_variable('bhl', [params.highway_ls], tf.float64,
                                 initializer=tf.ones_initializer())
            z_dec = tf.matmul(z, w1) + b1
            inp_h, inp_c = tf.split(tf.layers.dense(z_dec,params.decoder_hidden * 2),2, axis=1)
            initial_state = rnn_placeholders((tf.contrib.rnn.LSTMStateTuple(inp_c, inp_h), ))
        
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs=dec_inps,
                                                 sequence_length=d_seq_l,
                                                 initial_state=initial_state,
                                                 swap_memory=True,
                                                 dtype=tf.float64)
        # define decoder network
        if gen_mode:
                # only interested in the last output
                outputs = outputs[:, -1, :]
        # print(outputs.shape)
        outputs_r = tf.reshape(outputs, [-1, params.decoder_hidden])
        # print(outputs_r.shape,     "===============")
        x_logits = tf.layers.dense(outputs_r, units=vocab_size, activation=None)
        print(x_logits)
        if params.beam_search:
            sample = tf.nn.softmax(x_logits)
        else:
            sample = tf.multinomial(x_logits / params.temperature, 10)[0]
        print(sample)
        return x_logits, (initial_state, final_state), sample

def lstm_decoder_words(z_in, d_inputs,label_logits, d_seq_l, batch_size, embed, vocab_size,gen_mode=False,zsent=None,scope=None):

    with tf.variable_scope(scope, "decoder") as sc:
        with tf.device("/cpu:0"):
            dec_inps = tf.nn.embedding_lookup(embed, d_inputs)
        # turn off dropout for generation:
        if params.dec_keep_rate < 1 and not gen_mode:
            dec_inps = tf.nn.dropout(dec_inps, params.dec_keep_rate)
        
        label_logits=tf.nn.softmax(label_logits)
        dep=int(label_logits.shape[1])
        bs, T = tf.shape(dec_inps)[0], tf.shape(dec_inps)[1]
        print(bs, T)
        label_logits=tf.reshape(label_logits,[bs,T,dep])        
        print(label_logits)
        print(dec_inps)
        dec_inps=tf.concat([dec_inps,label_logits],axis=-1)
        print(dec_inps)
        # exit()
        max_sl = tf.shape(dec_inps)[1]
        # define cell
        if params.base_cell == 'lstm':
          base_cell = tf.contrib.rnn.LSTMCell
        elif params.base_cell == 'rnn':
            base_cell = tf.contrib.rnn.RNNCell
        else:
          # not working for now
          base_cell = tf.contrib.rnn.GRUCell
        
        cell = model.make_rnn_cell([params.decoder_hidden for _ in range(params.decoder_rnn_layers)], base_cell=base_cell)
        
        if gen_mode:
            z=zsent
        else:
            z=z_in
        if params.decode == 'hw':
            # Higway network [S.Sementiuta et.al]
            for i in range(params.highway_lc):
                with tf.variable_scope("hw_layer_dec{0}".format(i)) as scope:
                    z_dec = fully_connected(z, params.decoder_hidden * 2,activation_fn=tf.nn.sigmoid,
                            weights_initializer=xavier_initializer(),
                            biases_initializer=tf.zeros_initializer(),
                            scope="decoder_inp_state")

            inp_h, inp_c = tf.split(z_dec, 2, axis=1)
            initial_state = rnn_placeholders(
                (tf.contrib.rnn.LSTMStateTuple(inp_c, inp_h), ))
        elif params.decode == 'concat':
            z_out = tf.reshape(
              tf.tile(tf.expand_dims(z, 1), (1, max_sl, 1)),
              [batch_size, -1, params.latent_size])
            dec_inps = tf.concat([dec_inps, z_out], 2)
            initial_state = rnn_placeholders(
                cell.zero_state(tf.shape(dec_inps)[0], tf.float64))
        elif params.decode == 'mlp':
            # z->decoder initial state
            w1 = tf.get_variable('whl', [params.latent_size, params.highway_ls],
                                 tf.float64,
                                 initializer=tf.truncated_normal_initializer())
            b1 = tf.get_variable('bhl', [params.highway_ls], tf.float64,
                                 initializer=tf.ones_initializer())
            z_dec = tf.matmul(z, w1) + b1
            inp_h, inp_c = tf.split(tf.layers.dense(z_dec,params.decoder_hidden * 2),2, axis=1)
            initial_state = rnn_placeholders((tf.contrib.rnn.LSTMStateTuple(inp_c, inp_h), ))
        
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs=dec_inps,
                                                 sequence_length=d_seq_l,
                                                 initial_state=initial_state,
                                                 swap_memory=True,
                                                 dtype=tf.float64)
        # define decoder network
        if gen_mode:
                # only interested in the last output
                outputs = outputs[:, -1, :]
        # print(outputs.shape)
        outputs_r = tf.reshape(outputs, [-1, params.decoder_hidden])
        # print(outputs_r.shape,     "===============")
        x_logits = tf.layers.dense(outputs_r, units=vocab_size, activation=None)
        print(x_logits)
        if params.beam_search:
            sample = tf.nn.softmax(x_logits)
        else:
            sample = tf.multinomial(x_logits / params.temperature, 10)[0]
        print(sample)
        return x_logits, (initial_state, final_state), sample

def decoder(zglobal_sample, d_word_input, d_labels,seq_length,batch_size,label_embed,word_embed, word_vocab_size, label_vocab_size,gen_mode=False,zsent=None,inp_logits=None):

    Zglobal_dec_distribution = [tf.cast(0,dtype=tf.float64),tf.cast(1.0,dtype=tf.float64)]
    label_logits,( initial_state , final_state ),l_sample =lstm_decoder_labels(zglobal_sample, d_labels, seq_length, batch_size, label_embed,label_vocab_size,gen_mode,scope="zglobal_decoder_rnn")
    final_state = tf.concat(final_state[0], 1)
    # print(zglobal_sample.shape, zglobal_sample.dtype)
    # print(final_state.shape, final_state.dtype)
    gaussian_input=tf.concat([zglobal_sample,final_state],axis=-1) #########concatinate these as inputs to gaussian layer
    # print(gaussian_input.shape, gaussian_input.dtype)
    zsent_dec_mu,zsent_dec_sigma,zsent_dec_sample=gauss_layer(gaussian_input,params.latent_size, scope="zsent_dec_gauss")
    zsent_dec_distribution=[zsent_dec_mu,zsent_dec_sigma]

    # d_word_input=tf.cast(d_word_input, tf.float64)
    # decoder_input=tf.concat([d_word_input,tf.nn.softmax(label_logits)],axis=-1)
    # print(tf.shape(decoder_input))
    # print(d_word_input,label_logits)
    # print("################")
    if gen_mode:
        logits=inp_logits
    else:
        logits=label_logits

    word_logits,(initial_state_1 , final_state_1),w_sample =lstm_decoder_words(zsent_dec_sample, d_word_input,logits, seq_length, batch_size, word_embed,word_vocab_size,gen_mode,zsent,scope="zsent_decoder_rnn")

    return word_logits,label_logits,zsent_dec_distribution,Zglobal_dec_distribution,l_sample,w_sample,zsent_dec_sample