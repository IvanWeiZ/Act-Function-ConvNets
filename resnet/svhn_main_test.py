import os
import sys
import scipy.io as sio
import tensorflow as tf
import numpy as np

_DATA_PATH = os.path.join(os.path.dirname(__file__), '../datasets')

def get_data(is_training, data_dir):
  if is_training:
    filename = 'train_32x32.mat'
  else:
    filename = 'test_32x32.mat'

  filepath = os.path.join(data_dir, filename)
  assert (os.path.exists(filepath))
  
  data = sio.loadmat(filepath)
  X = np.rollaxis(data['X'], 3)
  y = data['y'].reshape((X.shape[0], 1))
  dataset = tf.data.Dataset.from_tensor_slices((X, y))
  
  return dataset

if __name__ == "__main__":
  input_fn(True, _DATA_PATH, 128) 
