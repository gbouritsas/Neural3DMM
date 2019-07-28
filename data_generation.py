from tqdm import tqdm
import numpy as np
import os, argparse



parser = argparse.ArgumentParser(description='Arguments for dataset split')
parser.add_argument('data_directory', type=str, required=True,
            help='Root data directory location, should be same as in neural3dmm.ipynb')
parser.add_argument('dataset_name', type=str, default='DFAUST', required=False,
            help='Dataset name, Default is DFAUST')
parser.add_argument('num_valid', type=int, default=100, required=False,
            help='Number of meshes in validation set, default 100')

args = parser.parse_args()


nVal = args.num_valid
root_dir = args.data_directory
dataset = args.dataset_name
name = ''

data = os.path.join(root_dir, dataset, 'preprocessed',name)
train = np.load(data+'/train.npy')

if os.path.exists(os.path.join(data,'points_train')):
    os.makedirs(os.path.join(data,'points_train'))

if os.path.exists(os.path.join(data,'points_val')):
    os.makedirs(os.path.join(data,'points_val'))

if os.path.exists(os.path.join(data,'points_test')):
    os.makedirs(os.path.join(data,'points_test'))


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

