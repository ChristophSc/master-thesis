import os
import torch
from read_data import index_ent_rel, graph_size, read_data
from data_utils import inplace_shuffle, filter_heads_tails

task_dirs = ['umls', 'kinship', 'fb15k237', 'wn18', 'wn18rr']
for task_dir in task_dirs:  
  print('---------- ', task_dir, ' ----------')
  kb_index = index_ent_rel(os.path.join('data', task_dir, 'train.txt'),
                          os.path.join('data',task_dir, 'valid.txt'),
                          os.path.join('data',task_dir, 'test.txt'))
  n_ent, n_rel = graph_size(kb_index)
  print('#ent:', n_ent)
  print('#rel:', n_rel)
  train_data = read_data(os.path.join('data',task_dir, 'train.txt'), kb_index)
  print('train:', len(train_data[0]))
  valid_data = read_data(os.path.join('data',task_dir, 'valid.txt'), kb_index)
  print('valid:', len(valid_data[0]))
  test_data = read_data(os.path.join('data',task_dir, 'test.txt'), kb_index)
  print('test:', len(test_data[0]))
  
  heads_filt, tails_filt = filter_heads_tails(n_ent, train_data, valid_data, test_data)
  print('#heads:', len(heads_filt))
  print('#tails:', len(tails_filt))
  print('')
  

