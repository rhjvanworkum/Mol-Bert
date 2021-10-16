import tensorflow as tf
import pandas as pd
import numpy as np

from utils import smiles2adjoin

str2num = {'<pad>':0 ,'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'Cl': 7, 'P': 8, 'Br':  9,
         'B': 10,'I': 11,'Si':12,'Se':13,'<unk>':14,'<mask>':15,'<global>':16}

num2str =  {i:j for j,i in str2num.items()}


class graph_bert_dataset:

  def __init__(self, path, smiles_field='canonical_smiles', addH=True):
    if path.endswith('.txt') or path.endswith('.tsv'):
      self.df = pd.read_csv(path,sep='\t')
    else:
      self.df = pd.read_csv(path)
    self.smiles_field = smiles_field
    self.vocab = str2num
    self.devocab = num2str
    self.addH = addH

  def get_data(self):
    data = self.df
    train_idx = []
    idx = data.sample(frac=0.9).index
    train_idx.extend(idx)

    data1 = data[data.index.isin(train_idx)]
    data2 = data[~data.index.isin(train_idx)]

    self.dataset1 = tf.data.Dataset.from_tensor_slices(data1[self.smiles_field].tolist())
    self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).padded_batch(256, padded_shapes=(
        tf.TensorShape([None]),tf.TensorShape([None,None]), tf.TensorShape([None]) ,tf.TensorShape([None]))).prefetch(50)

    self.dataset2 = tf.data.Dataset.from_tensor_slices(data2[self.smiles_field].tolist())
    self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
        tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None]),
        tf.TensorShape([None]))).prefetch(50)
    
    return self.dataset1, self.dataset2

  def numerical_smiles(self, smiles):
    smiles = smiles.numpy().decode()
    atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
    atoms_list = ['<global>'] + atoms_list
    nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
    temp = np.ones((len(nums_list),len(nums_list)))
    temp[1:,1:] = adjoin_matrix
    adjoin_matrix = (1 - temp) * (-1e9)

    choices = np.random.permutation(len(nums_list)-1)[:max(int(len(nums_list)*0.15),1)] + 1
    y = np.array(nums_list).astype('int64')
    weight = np.zeros(len(nums_list))
    for i in choices:
        rand = np.random.rand()
        weight[i] = 1
        if rand < 0.8:
            nums_list[i] = str2num['<mask>']
        elif rand < 0.9:
            nums_list[i] = int(np.random.rand() * 14 + 1)

    x = np.array(nums_list).astype('int64')
    weight = weight.astype('float32')
    return x, adjoin_matrix, y, weight

  def tf_numerical_smiles(self, data):
    x, adjoin_matrix, y, weight = tf.py_function(self.numerical_smiles, [data],
                                                  [tf.int64, tf.float32, tf.int64, tf.float32])
    x.set_shape([None])
    adjoin_matrix.set_shape([None,None])
    y.set_shape([None])
    weight.set_shape([None])
    return x, adjoin_matrix, y, weight
