# coding=utf-8

# @Author  : zhzhx2008
# @Time    : 19-7-11


import math
import os
import warnings

import jieba
import numpy as np
from keras import Model
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.engine.topology import Layer, Input, InputSpec
from keras.layers import Embedding, SpatialDropout1D, GlobalAveragePooling1D, Dropout, Dense, initializers, regularizers, constraints, Lambda, Concatenate, Add
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# from:https://github.com/CyberZHG/keras-pos-embd/blob/master/keras_pos_embd/pos_embd.py
class PositionEmbedding(Layer):
    """Turn integers (positions) into dense vectors of fixed size.
    eg. [[-4], [10]] -> [[0.25, 0.1], [0.6, -0.2]]
    Expand mode: negative integers (relative position) could be used in this mode.
        # Input shape
            2D tensor with shape: `(batch_size, sequence_length)`.
        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
    Add mode:
        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.
        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.
    Concat mode:
        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.
        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim + output_dim)`.
    """
    MODE_EXPAND = 'expand'
    MODE_ADD = 'add'
    MODE_CONCAT = 'concat'

    def __init__(self,
                 input_dim,
                 output_dim,
                 mode=MODE_EXPAND,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 **kwargs):
        """
        :param input_dim: The maximum absolute value of positions.
        :param output_dim: The embedding dimension.
        :param embeddings_initializer:
        :param embeddings_regularizer:
        :param activity_regularizer:
        :param embeddings_constraint:
        :param mask_zero: The index that represents padding. Only works in `append` mode.
        :param kwargs:
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mode = mode
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero is not False

        self.embeddings = None
        super(PositionEmbedding, self).__init__(**kwargs)

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'output_dim': self.output_dim,
                  'mode': self.mode,
                  'embeddings_initializer': initializers.serialize(self.embeddings_initializer),
                  'embeddings_regularizer': regularizers.serialize(self.embeddings_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'embeddings_constraint': constraints.serialize(self.embeddings_constraint),
                  'mask_zero': self.mask_zero}
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.mode == self.MODE_EXPAND:
            self.embeddings = self.add_weight(
                shape=(self.input_dim * 2 + 1, self.output_dim),
                initializer=self.embeddings_initializer,
                name='embeddings',
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
            )
        else:
            self.embeddings = self.add_weight(
                shape=(self.input_dim, self.output_dim),
                initializer=self.embeddings_initializer,
                name='embeddings',
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
            )
        super(PositionEmbedding, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        if self.mode == self.MODE_EXPAND:
            if self.mask_zero:
                output_mask = K.not_equal(inputs, self.mask_zero)
            else:
                output_mask = None
        else:
            output_mask = mask
        return output_mask

    def compute_output_shape(self, input_shape):
        if self.mode == self.MODE_EXPAND:
            return input_shape + (self.output_dim,)
        if self.mode == self.MODE_CONCAT:
            return input_shape[:-1] + (input_shape[-1] + self.output_dim,)
        return input_shape

    def call(self, inputs, **kwargs):
        if self.mode == self.MODE_EXPAND:
            if K.dtype(inputs) != 'int32':
                inputs = K.cast(inputs, 'int32')
            return K.gather(
                self.embeddings,
                K.minimum(K.maximum(inputs, -self.input_dim), self.input_dim) + self.input_dim,
            )
        input_shape = K.shape(inputs)
        if self.mode == self.MODE_ADD:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], input_shape[2]
        else:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], self.output_dim
        pos_embeddings = K.tile(
            K.expand_dims(self.embeddings[:seq_len, :self.output_dim], axis=0),
            [batch_size, 1, 1],
        )
        if self.mode == self.MODE_ADD:
            return inputs + pos_embeddings
        return K.concatenate([inputs, pos_embeddings], axis=-1)


# from:https://github.com/CyberZHG/keras-pos-embd/blob/master/keras_pos_embd/trig_pos_embd.py
class TrigPosEmbedding(Layer):
    """Position embedding use sine and cosine functions.
    See: https://arxiv.org/pdf/1706.03762
    Expand mode:
        # Input shape
            2D tensor with shape: `(batch_size, sequence_length)`.
        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
    Add mode:
        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.
        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.
    Concat mode:
        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.
        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim + output_dim)`.
    """
    MODE_EXPAND = 'expand'
    MODE_ADD = 'add'
    MODE_CONCAT = 'concat'

    def __init__(self,
                 mode=MODE_ADD,
                 output_dim=None,
                 **kwargs):
        """
        :param output_dim: The embedding dimension.
        :param kwargs:
        """
        if mode in [self.MODE_EXPAND, self.MODE_CONCAT]:
            if output_dim is None:
                raise NotImplementedError('`output_dim` is required in `%s` mode' % mode)
            if output_dim % 2 != 0:
                raise NotImplementedError('It does not make sense to use an odd output dimension: %d' % output_dim)
        self.mode = mode
        self.output_dim = output_dim
        self.supports_masking = True
        super(TrigPosEmbedding, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'mode': self.mode,
            'output_dim': self.output_dim,
        }
        base_config = super(TrigPosEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        if self.mode == self.MODE_EXPAND:
            return input_shape + (self.output_dim,)
        if self.mode == self.MODE_CONCAT:
            return input_shape[:-1] + (input_shape[-1] + self.output_dim,)
        return input_shape

    def call(self, inputs, mask=None):
        input_shape = K.shape(inputs)
        if self.mode == self.MODE_ADD:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], input_shape[2]
            pos_input = K.tile(K.expand_dims(K.arange(0, seq_len), axis=0), [batch_size, 1])
        elif self.mode == self.MODE_CONCAT:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], self.output_dim
            pos_input = K.tile(K.expand_dims(K.arange(0, seq_len), axis=0), [batch_size, 1])
        else:
            output_dim = self.output_dim
            pos_input = inputs
        if K.dtype(pos_input) != K.floatx():
            pos_input = K.cast(pos_input, K.floatx())
        evens = K.arange(0, output_dim // 2) * 2
        odds = K.arange(0, output_dim // 2) * 2 + 1
        even_embd = K.sin(
            K.dot(
                K.expand_dims(pos_input, -1),
                K.expand_dims(1.0 / K.pow(
                    10000.0,
                    K.cast(evens, K.floatx()) / K.cast(output_dim, K.floatx())
                ), 0)
            )
        )
        odd_embd = K.cos(
            K.dot(
                K.expand_dims(pos_input, -1),
                K.expand_dims(1.0 / K.pow(
                    10000.0, K.cast((odds - 1), K.floatx()) / K.cast(output_dim, K.floatx())
                ), 0)
            )
        )
        embd = K.stack([even_embd, odd_embd], axis=-1)
        output = K.reshape(embd, [-1, K.shape(inputs)[1], output_dim])
        if self.mode == self.MODE_CONCAT:
            output = K.concatenate([inputs, output], axis=-1)
        if self.mode == self.MODE_ADD:
            output += inputs
        return output


# reference:https://github.com/google-research/bert/blob/master/modeling.py
class Attention(Layer):
    def __init__(self,
                 attention_mask=None,
                 num_attention_heads=1,
                 size_per_head=512,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 attention_probs_dropout_prob=0.0,
                 initializer_range=0.02,
                 do_return_2d_tensor=False,
                 from_seq_length=None,
                 to_seq_length=None,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 **kwargs):

        self.init = initializers.truncated_normal(stddev=initializer_range)

        self.attention_mask = attention_mask
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.query_act = query_act
        self.key_act = key_act
        self.value_act = value_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.do_return_2d_tensor = do_return_2d_tensor
        self.from_seq_length = from_seq_length
        self.to_seq_length = to_seq_length

        self.output_dim = num_attention_heads * size_per_head

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):

        assert len(input_shape) == 2
        from_shape, to_shape = input_shape

        self.WQ = self.add_weight(name='{}_WQ'.format(self.name),
                                  shape=(from_shape[-1], self.output_dim),
                                  initializer=self.init,
                                  regularizer=self.W_regularizer,
                                  trainable=True,
                                  constraint=self.b_constraint)
        self.WK = self.add_weight(name='{}_WK'.format(self.name),
                                  shape=(to_shape[-1], self.output_dim),
                                  initializer=self.init,
                                  regularizer=self.W_regularizer,
                                  trainable=True,
                                  constraint=self.b_constraint)
        self.WV = self.add_weight(name='{}_WV'.format(self.name),
                                  shape=(to_shape[-1], self.output_dim),
                                  initializer=self.init,
                                  regularizer=self.W_regularizer,
                                  trainable=True,
                                  constraint=self.b_constraint)
        if self.bias:
            self.bq = self.add_weight(name='{}_bq'.format(self.name),
                                      shape=(self.output_dim,),
                                      initializer='zero',
                                      regularizer=self.b_regularizer,
                                      trainable=True,
                                      constraint=self.b_constraint)
            self.bk = self.add_weight(name='{}_bk'.format(self.name),
                                      shape=(self.output_dim,),
                                      initializer='zero',
                                      regularizer=self.b_regularizer,
                                      trainable=True,
                                      constraint=self.b_constraint)
            self.bv = self.add_weight(name='{}_bv'.format(self.name),
                                      shape=(self.output_dim,),
                                      initializer='zero',
                                      regularizer=self.b_regularizer,
                                      trainable=True,
                                      constraint=self.b_constraint)

        super(Attention, self).build(input_shape)

    def call(self, inputs, **kwargs):

        def reshape_to_matrix(input_tensor):
            ndims = K.ndim(input_tensor)
            if ndims < 2:
                raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                                 (input_tensor.shape))
            if ndims == 2:
                return input_tensor

            width = input_tensor.shape[-1]
            output_tensor = K.reshape(input_tensor, [-1, width])
            return output_tensor

        def dense(x, w, b, act):
            x = K.dot(x, w)
            if b:
                x = K.bias_add(x, b)
            if act.lower().strip() == 'softmax':
                x = K.softmax(x)
            elif act.lower().strip() == 'elu':
                x = K.elu(x)
            elif act.lower().strip() == 'gelu':
                x = 0.5 * x * (1 + K.tanh(math.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3))))
            elif act.lower().strip() == 'selu':
                alpha = 1.6732632423543772848170429916717
                scale = 1.0507009873554804934193349852946
                x = scale * K.elu(x, alpha)
            elif act.lower().strip() == 'softplus':
                x = K.softplus(x)
            elif act.lower().strip() == 'softsign':
                x = K.softsign(x)
            elif act.lower().strip() == 'relu':
                x = K.relu(x)
            elif act.lower().strip() == 'leaky_relu':
                x = K.relu(x, alpha=0.01)
            elif act.lower().strip() == 'tanh':
                x = K.tanh(x)
            elif act.lower().strip() == 'sigmoid':
                x = K.sigmoid(x)
            elif act.lower().strip() == 'hard_sigmoid':
                x = K.hard_sigmoid(x)
            return x

        from_tensor, to_tensor = inputs
        from_shape = K.int_shape(from_tensor)
        to_shape = K.int_shape(to_tensor)

        if len(from_shape) != len(to_shape):
            raise ValueError(
                "The rank of `from_tensor` must match the rank of `to_tensor`.")

        if len(from_shape) == 3:
            self.from_seq_length = from_shape[1]
            self.to_seq_length = to_shape[1]
        elif len(from_shape) == 2:
            if (self.from_seq_length is None or self.to_seq_length is None):
                raise ValueError(
                    "When passing in rank 2 tensors to attention_layer, the values "
                    "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                    "must all be specified.")

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`

        from_tensor_2d = reshape_to_matrix(from_tensor)
        to_tensor_2d = reshape_to_matrix(to_tensor)

        # `query_layer` = [B*F, N*H]
        query_layer = dense(from_tensor_2d, self.WQ, self.bq, self.query_act)
        # `key_layer` = [B*T, N*H]
        key_layer = dense(to_tensor_2d, self.WK, self.bk, self.key_act)
        # `value_layer` = [B*T, N*H]
        value_layer = dense(to_tensor_2d, self.WV, self.bv, self.value_act)

        # `query_layer` = [B, F, N, H]
        query_layer = K.reshape(query_layer, [-1, self.from_seq_length, self.num_attention_heads, self.size_per_head])
        # `query_layer` = [B, N, F, H]
        query_layer = K.permute_dimensions(query_layer, [0, 2, 1, 3])
        # `key_layer` = [B, T, N, H]
        key_layer = K.reshape(key_layer, [-1, self.to_seq_length, self.num_attention_heads, self.size_per_head])
        # `key_layer` = [B, N, T, H]
        key_layer = K.permute_dimensions(key_layer, [0, 2, 1, 3])

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]
        attention_scores = K.batch_dot(query_layer, key_layer, axes=[3, 3])
        attention_scores = attention_scores * 1.0 / K.sqrt(K.cast(self.size_per_head, dtype=K.floatx()))

        if self.attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = K.expand_dims(self.attention_mask, axis=1)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - K.cast(attention_mask, dtype=K.floatx())) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = K.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = K.dropout(attention_probs, self.attention_probs_dropout_prob)

        # `value_layer` = [B, T, N, H]
        value_layer = K.reshape(value_layer, [-1, self.to_seq_length, self.num_attention_heads, self.size_per_head])

        # `value_layer` = [B, N, T, H]
        value_layer = K.permute_dimensions(value_layer, [0, 2, 1, 3])

        # `context_layer` = [B, N, F, H]
        context_layer = K.batch_dot(attention_probs, value_layer, axes=[3, 2])

        # `context_layer` = [B, F, N, H]
        context_layer = K.permute_dimensions(context_layer, [0, 2, 1, 3])

        if self.do_return_2d_tensor:
            # `context_layer` = [B*F, N*V]
            context_layer = K.reshape(context_layer, [-1, self.num_attention_heads * self.size_per_head])
        else:
            # `context_layer` = [B, F, N*V]
            context_layer = K.reshape(context_layer, [-1, self.from_seq_length, self.num_attention_heads * self.size_per_head])

        return context_layer

    def compute_output_shape(self, input_shape):
        from_shape, to_shape = input_shape
        if self.do_return_2d_tensor:
            return from_shape[0], self.num_attention_heads * self.size_per_head
        else:
            return from_shape[0], self.from_seq_length, self.num_attention_heads * self.size_per_head


# from:https://github.com/CyberZHG/keras-layer-normalization/blob/master/keras_layer_normalization/layer_normalization.py
class LayerNormalization(Layer):
    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'beta_constraint': constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs


def reshape_to_matrix(input_tensor):
    ndims = K.ndim(input_tensor)
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = Lambda(lambda x: K.reshape(x, [-1, width]))(input_tensor)
    # output_tensor = K.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_tuple):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_tuple) == 2:
        return output_tensor

    output_shape = K.int_shape(output_tensor)

    orig_dims = orig_shape_tuple[0:-1]
    orig_dims = list(orig_dims)
    orig_dims[0] = -1
    width = output_shape[-1]

    # return K.reshape(output_tensor, orig_dims + [width])
    return Lambda(lambda x: K.reshape(x, orig_dims + [width]))(output_tensor)


# reference:https://github.com/google-research/bert/blob/master/modeling.py
def Transformer(input_tensor,
                attention_mask=None,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                intermediate_act_fn=None,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                initializer_range=0.02,
                do_return_all_layers=False):
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = K.int_shape(input_tensor)
    # batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                         (input_width, hidden_size))

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.
    prev_output = reshape_to_matrix(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        layer_input = prev_output

        attention_heads = []
        attention_head = Attention(
            attention_mask=attention_mask,
            num_attention_heads=num_attention_heads,
            size_per_head=attention_head_size,
            query_act='gelu',
            key_act='gelu',
            value_act='gelu',
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            do_return_2d_tensor=True,
            from_seq_length=seq_length,
            to_seq_length=seq_length)([layer_input, layer_input])
        attention_heads.append(attention_head)
        attention_output = None
        if len(attention_heads) == 1:
            attention_output = attention_heads[0]
        else:
            # In the case where we have other sequences, we just concatenate
            # them to the self-attention head before the projection.
            attention_output = Concatenate(axis=-1)(attention_heads)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        attention_output = Dense(hidden_size,
                                 kernel_initializer=initializers.truncated_normal(stddev=initializer_range))(attention_output)
        attention_output = Dropout(hidden_dropout_prob)(attention_output)
        attention_output = Add()([attention_output, layer_input])
        attention_output = LayerNormalization()(attention_output)

        # The activation is only applied to the "intermediate" hidden layer.
        intermediate_output = Dense(intermediate_size, activation=intermediate_act_fn,
                                    kernel_initializer=initializers.truncated_normal(stddev=initializer_range))(attention_output)

        # Down-project back to `hidden_size` then add the residual.
        layer_output = Dense(hidden_size,
                             kernel_initializer=initializers.truncated_normal(stddev=initializer_range))(intermediate_output)
        layer_output = Dropout(hidden_dropout_prob)(layer_output)
        layer_output = Add()([layer_output, attention_output])
        layer_output = LayerNormalization()(layer_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output


warnings.filterwarnings("ignore")

seed = 2019
np.random.seed(seed)


def get_labels_datas(input_dir):
    datas_word = []
    datas_char = []
    labels = []
    label_dirs = os.listdir(input_dir)
    for label_dir in label_dirs:
        txt_names = os.listdir(os.path.join(input_dir, label_dir))
        for txt_name in txt_names:
            with open(os.path.join(input_dir, label_dir, txt_name), 'r') as fin:
                content = fin.readline()  # 只取第一行
                content = content.strip().replace(' ', '')
                datas_word.append(' '.join(jieba.cut(content)))
                datas_char.append(' '.join(list(content)))
                labels.append(label_dir)
    return labels, datas_word, datas_char


def get_label_id_map(labels):
    labels = set(labels)
    id_label_map = {}
    label_id_map = {}
    for index, label in enumerate(labels):
        id_label_map[index] = label
        label_id_map[label] = index
    return id_label_map, label_id_map


input_dir = './data/THUCNews'
labels, datas_word, datas_char = get_labels_datas(input_dir)
id_label_map, label_id_map = get_label_id_map(labels)

labels, labels_test, datas_word, datas_word_test, datas_char, datas_char_test = train_test_split(labels, datas_word, datas_char, test_size=0.3, shuffle=True, stratify=labels)
labels_train, labels_dev, datas_word_train, datas_word_dev, datas_char_train, datas_char_dev = train_test_split(labels, datas_word, datas_char, test_size=0.1, shuffle=True, stratify=labels)

y_train = [label_id_map.get(x) for x in labels_train]
y_dev = [label_id_map.get(x) for x in labels_dev]
y_test = [label_id_map.get(x) for x in labels_test]

num_classes = len(set(y_train))
y_train_index = to_categorical(y_train, num_classes)
y_dev_index = to_categorical(y_dev, num_classes)
y_test_index = to_categorical(y_test, num_classes)

# keras extract feature
tokenizer = Tokenizer()
tokenizer.fit_on_texts(datas_word_train)
# feature5: word index for deep learning
x_train_word_index = tokenizer.texts_to_sequences(datas_word_train)
x_dev_word_index = tokenizer.texts_to_sequences(datas_word_dev)
x_test_word_index = tokenizer.texts_to_sequences(datas_word_test)

max_word_length = max([len(x) for x in x_train_word_index])
x_train_word_index = pad_sequences(x_train_word_index, maxlen=max_word_length)
x_dev_word_index = pad_sequences(x_dev_word_index, maxlen=max_word_length)
x_test_word_index = pad_sequences(x_test_word_index, maxlen=max_word_length)

input = Input(shape=(max_word_length,))
embedding = Embedding(len(tokenizer.word_index) + 1, 128)(input)

# embedding = TrigPosEmbedding(mode='add', output_dim=128)(embedding)

embedding = PositionEmbedding(input_dim=128, output_dim=128, mode='add')(embedding)

embedding = SpatialDropout1D(0.2)(embedding)
transformer = Transformer(embedding,
                          attention_mask=None,
                          hidden_size=128,
                          num_hidden_layers=8,
                          num_attention_heads=8,
                          intermediate_size=128,
                          intermediate_act_fn='relu',
                          hidden_dropout_prob=0.1,
                          attention_probs_dropout_prob=0.1,
                          initializer_range=0.02,
                          do_return_all_layers=False)
att = GlobalAveragePooling1D()(transformer)
att = Dropout(0.2)(att)
output = Dense(num_classes, activation='softmax')(att)
model = Model(inputs=input, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
print(model.summary())

model_weight_file = './model_multi_head_multi_layer_transformer.h5'
model_file = './model_multi_head_multi_layer_transformer.model'
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint(model_weight_file, save_best_only=True, save_weights_only=True)
model.fit(x_train_word_index,
          y_train_index,
          batch_size=8,
          epochs=1000,
          verbose=2,
          callbacks=[early_stopping, model_checkpoint],
          validation_data=(x_dev_word_index, y_dev_index),
          shuffle=True)

model.load_weights(model_weight_file)
# model save error, if you want save model, see https://github.com/keras-team/keras/issues/9342
# model.save(model_file)
evaluate = model.evaluate(x_test_word_index, y_test_index, batch_size=8, verbose=2)
print('loss value=' + str(evaluate[0]))
print('metrics value=' + str(evaluate[1]))

# no position embedding
# loss value=1.119690725727687
# metrics value=0.6507936517397562

# TrigPosEmbedding
# loss value=1.1005233223476107
# metrics value=0.5634920634920635

# PositionEmbedding
# loss value=1.3514895968967013
# metrics value=0.6111111111111112
