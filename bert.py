import tensorflow as tf



def gelu(x):
  return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.)))



class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()

    self.num_heads = num_heads
    self.d_model = d_model

    self.depth = self.d_model // num_heads

    self.weights_q = tf.keras.layers.Dense(d_model)
    self.weights_k = tf.keras.layers.Dense(d_model)
    self.weights_v = tf.keras.layers.Dense(d_model)

    self.weights_o = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def scaled_dot_product_attention(self, q, k, v, mask, adjoin_matrix):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    if adjoin_matrix is not None:
        scaled_attention_logits += adjoin_matrix

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

  def call(self, v, k, q, mask, adjoin_matrix):
    batch_size = tf.shape(q)[0]

    q = self.weights_q(q)  # (batch_size, seq_len, d_model)
    k = self.weights_k(k)  # (batch_size, seq_len, d_model)
    v = self.weights_v(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask, adjoin_matrix)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.weights_o(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights



class EncodingLayer(tf.keras.layers.Layer):
  """
    d_model: dimensionality of the embedding space
    num_heads: number of parallel attention mechanism, as described in multi-head attention
    dff: dimensionality of the "deep"-layer of the Feed Forward Network
    rate: dropout-rate
  """

  def __init__(self, d_model, num_heads, dff, rate=0.1):
      super(EncodingLayer, self).__init__()

      self.mha = MultiHeadAttention(d_model, num_heads)

      self.ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation=gelu),
        tf.keras.layers.Dense(d_model)
      ])

      self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
      self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

      self.dropout1 = tf.keras.layers.Dropout(rate)
      self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask, adjoin_matrix):
    attn_output, attention_weights = self.mha(x, x, x, mask,adjoin_matrix)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2,attention_weights



class Encoder(tf.keras.layers.Layer):

  def __init__(self, num_layers, d_model, dff, num_heads, input_vocab_size, rate):
    super(Encoder, self).__init__()

    self.num_layers = num_layers
    self.d_model = d_model

    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

    self.encoding_layers = [EncodingLayer(d_model, num_heads, dff) for _ in range(self.num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask, adjoin_matrix):
    seq_len = tf.shape(x)[1]
    adjoin_matrix = adjoin_matrix[:,tf.newaxis,:,:]
    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
        x, attention_weights = self.enc_layers[i](x, training, mask,adjoin_matrix)
    return x  # (batch_size, input_seq_len, d_model)



class MGBert(tf.keras.Model):

  def __init__(self, num_layers, d_model, dff, num_heads, vocab_size, dropout_rate):
    super(MGBert, self).__init__()
    
    self.encoder = Encoder(num_layers, d_model, dff, num_heads, vocab_size, dropout_rate)

    self.fc1 = tf.keras.layers.Dense(d_model, activation=gelu)
    self.layernorm = tf.keras.layers.LayerNormalization(-1)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

  def call(self, x, adjoin_matrix, mask, training=False):
    x = self.encoder(x, training=training, mask=mask, adjoin_matrix=adjoin_matrix)
    x = self.fc1(x)
    x = self.layernorm(x)
    x = self.fc2(x)
    return x