from __future__ import print_function, division, absolute_import
import tensorflow as tf
import numpy as np

# import zhusuan as zs
# from zhusuan import reuse

import utils.data as data_
import utils.label_data as label_data_
import utils.model as model
from utils.ptb import reader
from utils import parameters
from utils.beam_search import beam_search

from tensorflow.python import debug as tf_debug
from tensorflow.python.util.nest import flatten
import os
from tensorflow.python.tools import inspect_checkpoint as chkp
from hvae_model import encoder,decoder

# PTB input from tf tutorial
class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)

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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def main(params):
    if params.input == 'PTB':
        # data in form [data, labels]
        train_data_raw, train_label_raw = data_.ptb_read('./DATA/train_untrans_6k/')
        word_data, encoder_word_data,word_labels_arr, word_embed_arr, word_data_dict = data_.prepare_data(train_data_raw,train_label_raw, params,'./DATA/train_untrans_6k/')

        train_label_raw, valid_label_raw, test_label_raw = label_data_.ptb_read('./DATA/train_untrans_6k/')
        label_data, label_labels_arr, label_embed_arr, label_data_dict = label_data_.prepare_data(train_label_raw, params)
        
    with tf.Graph().as_default() as graph:
        
        label_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None],name="lable_inputs")
        word_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None],name="word_inputs")

        d_word_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None],name="d_word_inputs")
        d_label_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None],name="d_label_inputs")
        
        
        
        d_word_labels = tf.placeholder(shape=[None, None], dtype=tf.int32,name="d_word_labels")
        d_label_labels = tf.placeholder(shape=[None, None], dtype=tf.int32,name="d_label_labels")
        
        with tf.device("/cpu:0"):
            if not params.pre_trained_embed:
                word_embedding = tf.get_variable(
                    "word_embedding", [data_dict.vocab_size,
                                  params.embed_size], dtype=tf.float64)
                vect_inputs = tf.nn.embedding_lookup(word_embedding, word_inputs)
            else:
                # [data_dict.vocab_size, params.embed_size]
                word_embedding = tf.Variable(
                    word_embed_arr,
                    trainable=params.fine_tune_embed,
                    name="word_embedding", dtype=tf.float64) #creates a variable that can be used as a tensor
                vect_inputs = tf.nn.embedding_lookup(word_embedding, word_inputs,name="word_lookup")
                
                label_embedding = tf.Variable(
                    label_embed_arr,
                    trainable=params.fine_tune_embed,
                    name="label_embedding", dtype=tf.float64) #creates a variable that can be used as a tensor

                label_inputs_1=tf.nn.embedding_lookup(label_embedding, label_inputs,name="label_lookup")
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        sizes=word_data_dict.sizes
        word_vocab_size = max(sizes[1],sizes[2],sizes[0])
        label_vocab_size=label_data_dict.vocab_size
        seq_length = tf.placeholder_with_default([0.0], shape=[None])
        d_seq_length = tf.placeholder(shape=[None], dtype=tf.float64)
        # qz = q_net(word_inputs, seq_length, params.batch_size)
        zglobal_sample = tf.placeholder(dtype=tf.float64,shape=[None, params.latent_size])
        zsent_sample=tf.placeholder(dtype=tf.float64,shape=[None,params.latent_size])
        inp_logits=tf.placeholder(dtype=tf.float64,shape=[None,params.label_embed_size])
        word_logits,label_logits,_,_,l_smpl,w_smpl,zs=decoder(zglobal_sample, d_word_inputs, d_label_inputs,seq_length,params.batch_size,label_embedding,word_embedding, word_vocab_size, label_vocab_size,gen_mode=True,zsent=zsent_sample,inp_logits=inp_logits)

        # word_logits,_,_, _,_,w_smpl,_=decoder(zglobal_sample, d_word_inputs, d_label_inputs,seq_length,params.batch_size,label_embedding,word_embedding, word_vocab_size, label_vocab_size,gen_mode_1=False,gen_mode_2=True,zsent=zsent_sample)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(),
                      tf.local_variables_initializer()])
            print("here")
            try:

                path="./models_ckpts_"+params.name+"/vae_lstm_model-11900"
                # print(path)
                # chkp.print_tensors_in_checkpoint_file(path, tensor_name='', all_tensors=True)
                saver.restore(sess,path )
            # saver.restore(sess, "./models_ckpts_1/vae_lstm_model-258600")
            except:
                print("-----exception occurred--------")
                exit()
                # traceback.print_exc()

            print("Model Restored")

           
            total_parameters = 0
            #print_vars("trainable variables")
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                print(shape, variable.name)
                #print(len(shape))
                variable_parameters = 1
                for dim in shape:
                    print(dim)
                    variable_parameters *= dim.value
                print(variable_parameters)
                total_parameters += variable_parameters
            print(total_parameters)

            batch_size=1
            # number_of_samples=params.number_of_samples
            number_of_samples=10000
            same_context_sentences=1
            sentence_file="./r_VACS_kl_10k.txt"
            labels_file="./r_VACS_kl_10k_labels.txt"

            f1=open(sentence_file,'w+')
            f2=open(labels_file,'w+')

            print("----------------------SENTENCES------------------------\n\n")
            for num in range(number_of_samples):
                params.is_training = False
                
                sentence=['<BOS>']
                label_seq = ['3']
                state = None
                input_sent_vect = [word_data_dict.word2idx[word] for word in sentence]
                # z = tf.random_normal(tf.shape([1, params.latent_size]), name='z',dtype=tf.float64).eval()
                z=np.random.normal(0,1,(1,params.latent_size))

                ###initialising random variables for label decoder
                z_1=np.random.normal(0,1,(1,params.latent_size))
                l_1=np.random.rand(1,params.label_embed_size)
                # print(input_sent_vect)
                # print(z)
                # print(z)

                for i in range(params.gen_length):
                    # generate until <EOS> tag
                    if "4" in label_seq:
                        break
                    input_label_vect = [label_data_dict.word2idx[word] for word in label_seq]
                    feed = {d_label_inputs: np.array(input_label_vect).reshape([1, len(input_label_vect)]),zsent_sample: z_1,inp_logits:l_1,
                            seq_length: [len(input_label_vect)],zglobal_sample:z, d_word_inputs: np.array(input_sent_vect).reshape([1, len(input_sent_vect)]) }
                    # for the first decoder step, the state is None
                    # if state is not None:
                    #      feed.update({in_state: state})
                    a,index = sess.run([label_logits,l_smpl], feed)
                    # print(a,a.shape)
                    if(i==0):
                        logit_arr=np.array(a)
                    else:
                        logit_arr=np.concatenate((logit_arr,a))

                    # print(index)
                    # exit()
                    index=index[0]
                    label_seq += [label_data_dict.idx2word[int(index)]]
                label_seq=[word for word in label_seq if word not in ['3','4']]
                label_out = ' '.join([w for w in label_seq])
                print(label_out,len(label_seq))
                # sizes=word_data_dict.sizes
                # print(logit_arr)
                print(logit_arr.shape)
                # exit()
                

                for num in range(same_context_sentences):
                    print(num)
                    i=0
                    z_sent_sample=sess.run(zs,feed)
                    
                    b1=sizes[0]
                    b2=sizes[0]+sizes[1]
                    b3=sizes[0]+sizes[1]+sizes[2]
                    sentence=['<BOS>']
                    input_sent_vect = [word_data_dict.word2idx[word] for word in sentence]
                    while(i<len(label_seq)):
                        # for i in range(len(label_seq)):
                        # generate until <EOS> tag
                        
                        input_sent_vect = [word_data_dict.word2idx[word] for word in sentence]
                        feed = {d_label_inputs: np.array(input_label_vect).reshape([1, len(input_label_vect)]),zsent_sample: z_sent_sample,inp_logits:logit_arr[:i+1],
                                seq_length: [len(input_sent_vect)],zglobal_sample:z, d_word_inputs: np.array(input_sent_vect).reshape([1, len(input_sent_vect)]) }
                        tmp=np.array(input_sent_vect).reshape([1, len(input_sent_vect)])
                        # print(tmp, tmp.shape)
                        # print(a,a.shape)

                        w_logits= sess.run(word_logits, feed)
                        # print(w_logits)
                        if(label_seq[i]=='0'):
                            w_logits=w_logits[0][:sizes[1]]
                            w_probs=softmax(w_logits)
                            # index_arr=np.argsort(np.array(w_probs))    
                            index_arr=np.random.choice(len(w_probs),5,p=w_probs)
                            index_arr=index_arr+b1

                        elif(label_seq[i]=='1'):
                            w_logits=w_logits[0][:sizes[2]]
                            w_probs=softmax(w_logits)
                            # index_arr=np.argsort(np.array(w_probs))    
                            index_arr=np.random.choice(len(w_probs),5,p=w_probs)
                            index_arr=index_arr+b2

                        elif(label_seq[i]=='2'):
                            w_logits=w_logits[0][:sizes[0]]
                            w_probs=softmax(w_logits)
                            # index_arr=np.argsort(np.array(w_probs))    
                            index_arr=np.random.choice(len(w_probs),5,p=w_probs)

                        for j in index_arr:
                            index=j
                            word=word_data_dict.idx2word[int(index)]
                            if(word!="<EOS>" and word!="<BOS>"):
                                i+=1
                                # print(i,index)
                                # print(word)
                                sentence += [word]
                                
                                break
                        # print(w_logits)
                        # print(w_logits.shape)
                        # print(min(w_logits[0]),max(w_logits[0]))
                        # exit()
                        # print(label_seq[i])
                        # if(label_seq[i]=='0'):
                        #     # print(label_seq[i])
                        #     req_logits=w_logits[0][sizes[0]:sizes[0]+sizes[1]]
                        #     req_probs=softmax(req_logits)
                        #     req_index=np.argmax(np.array(req_probs))
                        #     index=sizes[0]+req_index
                        # elif (label_seq[i]=='1'):
                        #     # print(label_seq[i])
                        #     req_logits=w_logits[0][(sizes[0]+sizes[1]):(sizes[0]+sizes[1]+sizes[2])]
                        #     req_probs=softmax(req_logits)
                        #     req_index=np.argmax(np.array(req_probs))
                        #     index=sizes[0]+sizes[1]+req_index
                        # elif (label_seq[i]=='2'):
                        #     # print(label_seq[i])
                        #     req_logits=w_logits[0][:sizes[0]]
                        #     req_probs=softmax(req_logits)
                        #     req_index=np.argmax(np.array(req_probs))
                        #     index=req_index
                        # # print(label_seq[i],i,index)
                        # print(b,b.shape)
                        # print(index)
                        
                    sentence=[word for word in sentence if word not in ['<BOS>','<EOS>']]
                    sentence_cm = ' '.join([w for w in sentence])
                    print(sentence_cm,len(sentence))
                    print("\n")
                    f1.write(sentence_cm)
                    f1.write("\n")
                    f2.write(label_out)
                    f2.write("\n")
                print("-----------------------------------------\n")

            f1.close()
            f2.close()
                
if __name__=="__main__":
    params = parameters.Parameters()
    params.parse_args()
    main(params)
