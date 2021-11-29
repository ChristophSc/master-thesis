from sklearn.model_selection import train_test_split
import numpy
import random

def save_file(filename, data):
  with open(filename, 'w') as f:
    for d in data:
        f.write(d)
        f.write('\n')
        
with open("./data/kinship/all.txt", "rb") as f:
  lines = [x.decode('utf8').strip().replace(' ', '\t') for x in f.readlines()]
  data = numpy.array(lines)
  #convert array to numpy type array

  random.shuffle(data)
  train_data = data[:int((len(data)+1)*.80)] #Remaining 80% to training set
  test_data = data[int(len(data)*.80+1):] #Splits 20% data to test set
  
  valid_data = train_data[:int((len(train_data)+1)*.20)] #Remaining 20% to validation set
  train_data = train_data[int(len(train_data)*.20+1):]
  
  save_file('./data/kinship/train.txt', train_data)
  save_file('./data/kinship/valid.txt', valid_data)
  save_file('./data/kinship/test.txt', test_data)