from tqdm import tqdm
import numpy as np
import os

nVal = 100
root_dir = '/data/gb318/datasets/'
dataset = 'DFAUST'
name = ''

data = os.path.join(root_dir, dataset, 'preprocessed',name)
train = np.load(data+'/train.npy')

for i in tqdm(range(len(train)-nVal)):
    np.save(os.path.join(data,'points_train','{}'.format(i)+'.npy'),train[i])
for i in range(len(train)-nVal,len(train)):
    np.save(os.path.join(data,'points_val','{}'.format(i)+'.npy'),train[i])
    
test = np.load(data+'/test.npy')
for i in range(len(test)):
    np.save(os.path.join(data,'points_test','{}'.format(i)+'.npy'),test[i])
    
files = []
for r, d, f in os.walk(os.path.join(data,'points_train')):
    for file in f:
        if '.npy' in file:
            files.append(os.path.splitext(file)[0])
np.save(os.path.join(data,'paths_train.npy'),files)

files = []
for r, d, f in os.walk(os.path.join(data,'points_val')):
    for file in f:
        if '.npy' in file:
            files.append(os.path.splitext(file)[0])
np.save(os.path.join(data,'paths_val.npy'),files)

files = []
for r, d, f in os.walk(os.path.join(data,'points_test')):
    for file in f:
        if '.npy' in file:
            files.append(os.path.splitext(file)[0])
np.save(os.path.join(data,'paths_test.npy'),files)