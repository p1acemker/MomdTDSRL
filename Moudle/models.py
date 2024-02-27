from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers import LSTMCell, LSTM, TimeDistributed, Dense, Input, Lambda, Dropout
from keras.optimizers import Adam, SGD
from sklearn.metrics import roc_auc_score
import numpy as np
from keras import Model
import rdkit.Chem as Chem
from keras.layers import LeakyReLU, Bidirectional, Multiply
from keras.regularizers import l2
from keras.layers import Concatenate, Flatten, Softmax
from Modules.global_parameters import MAX_FRAGMENTS, MAX_SWAP, N_DENSE, \
                              N_DENSE2, N_LSTM

from keras.layers import *

from keras_bert import load_trained_model_from_checkpoint
from Transformer import TransformerEncoder


from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Add
import math
from Transformer.Attention import MultiHeadedAttention
from Transformer.LayerNormalization import LayerNormalization
from Transformer.PositionWiseFeedForward import PositionWiseFeedForward

class TransformerEncoder():
    '''
    Main Transformer Encoder block : Encapsulates different layers with a Transformer Encoder block and calls them in order.
    Inputs
    d_model : dimensions of the output and internal layers
    heads   : number of heads
    dim_q   : query and key dimension
    dim_v   : value dimension
    hidden_units : hidden units for the positionwise feed forward network
    dropout_rate : dropout_rate

    Outputs
    A tuple:Transformer Encoder Representation, attention weights for each head and token
    '''
    def __init__(self, d_model,heads,dim_q,dim_v,hidden_units,dropout_rate,name,activation='relu', **kwargs):
        self.dim_v        = dim_v
        self.dim_q        = dim_q
        self.hidden_units = hidden_units
        self.heads        = heads

        self.attention_layer      = MultiHeadedAttention(d_model = d_model,heads = self.heads,dim_q = self.dim_q,dim_v = self.dim_v,dropout_rate=dropout_rate,name=name)
        self.normalization_layer  = LayerNormalization()
        self.feedforward          = PositionWiseFeedForward(d_model = d_model,inner_dim = self.hidden_units,dropout_rate=dropout_rate,name=name)


    def __call__(self, x):

        attention_vec,attention_weights   = self.attention_layer(x)
        normalized_inp                    = self.normalization_layer(Add()([attention_vec,x]))
        feedforward_out                   = self.feedforward(normalized_inp)

        transformer_out = self.normalization_layer(Add()([feedforward_out,normalized_inp]))

        return [transformer_out,attention_weights]



# Objective to optimize
def maximization(y_true, y_pred):
    return K.mean(-K.log(y_pred) * y_true)



n_actions = MAX_FRAGMENTS * MAX_SWAP + 1


# Create models
def build_models(inp_shape):
    #bert_model = load_trained_model_from_checkpoint('/home/leeh/下载/deep/chinese_L-12_H-768_A-12/bert_config.json', '/home/leeh/下载/deep/chinese_L-12_H-768_A-12/bert_model.ckpt', seq_len=None)

    # for l in bert_model.layers:
    #     l.trainable = True
    # Build the actor

    #x = bert_model([hidden,hidden2])
    #x = Lambda(lambda x: x[:, 0])(x)


    inp = Input(inp_shape)
    hidden_inp = LeakyReLU(0.1)(TimeDistributed(Dense(N_DENSE, activation="linear"))(inp))
    hidden = LSTM(N_LSTM, return_sequences=True)(hidden_inp)
    hidden = Flatten()(hidden)

    hidden2 = LSTM(N_LSTM, return_sequences=True, go_backwards=True)(hidden_inp)
    hidden2 = Flatten()(hidden2)
    inp2 = Input((1,))
    hidden = Concatenate()([hidden, hidden2, inp2])

    hidden = LeakyReLU(0.1)(Dense(N_DENSE2, activation="linear")(hidden))
    out = Dense(n_actions, activation="softmax", activity_regularizer=l2(0.001))(hidden)
    hidden = Concatenate()([out, inp2])

    hidden = LeakyReLU(0.1)(Dense(N_DENSE2, activation="linear")(hidden))
    out = Dense(n_actions, activation="softmax", activity_regularizer=l2(0.001))(hidden)
    actor = Model([inp,inp2], out)
    actor.compile(loss=maximization, optimizer=Adam(0.0005))


    # Build the critic
    inp = Input(inp_shape)
    hidden = LeakyReLU(0.1)(TimeDistributed(Dense(N_DENSE, activation="linear"))(inp))
    hidden = Bidirectional(LSTM(2*N_LSTM))(hidden)

    inp2 = Input((1,))
    hidden = Concatenate()([hidden, inp2])
    hidden = LeakyReLU(0.1)(Dense(N_DENSE2, activation="linear")(hidden))
    out = Dense(1, activation="linear")(hidden)

    critic = Model([inp,inp2], out)
    critic.compile(loss="MSE", optimizer=Adam(0.0001))


    return actor, critic
