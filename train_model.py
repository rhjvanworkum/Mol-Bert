import tensorflow as tf

from bert import MGBert
from dataloader import graph_bert_dataset

n_epochs = 10
vocab_size = 17
dropout_rate = 0.1

model = MGBert(num_layers=6, d_model=256, dff=256*2, num_heads=8, vocab_size=vocab_size, dropout_rate=dropout_rate)

train_dataset, test_dataset = graph_bert_dataset(path='./chembl_29_chemreps.txt').get_data()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(1e-4)

def train_step(x, adjoin_matrix, y, char_weight):
  seq = tf.cast(tf.math.equal(x, 0), tf.float32)
  mask = seq[:, tf.newaxis, tf.newaxis, :]
  with tf.GradientTape() as tape:
    predictions = model(x,adjoin_matrix=adjoin_matrix,mask=mask,training=True)
    loss = loss_function(y,predictions,sample_weight=char_weight)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss.update_state(loss)
  train_accuracy.update_state(y,predictions,sample_weight=char_weight)

def test_step(x, adjoin_matrix,y, char_weight):
  seq = tf.cast(tf.math.equal(x, 0), tf.float32)
  mask = seq[:, tf.newaxis, tf.newaxis, :]
  predictions = model(x,adjoin_matrix=adjoin_matrix,mask=mask,training=False)
  test_accuracy.update_state(y,predictions,sample_weight=char_weight)

for epoch in range(n_epochs):
  train_loss.reset_states()

  for (batch, (x, adjoin_matrix, y, char_weight)) in enumerate(train_dataset):
    train_step(x, adjoin_matrix, y , char_weight)

    if batch % 500 == 0:
      print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, train_loss.result()))
      print('Accuracy: {:.4f}'.format(train_accuracy.result()))
      train_accuracy.reset_states()

  # print(arch['path'] + '/bert_weights{}_{}.h5'.format(arch['name'], epoch+1))
  # print('Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result()))
  # print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
  # print('Accuracy: {:.4f}'.format(train_accuracy.result()))
  # model.save_weights(arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'],epoch+1))
  # print('Saving checkpoint')