import numpy as np
import lmdb
import caffe
import os
import random
import h5py
import pdb

def loadtoXY(option, classdict):
    if option == 'train':
        path = train_path
    elif option == 'test':
        path = test_path

    # shuffle the data
    content = os.listdir(path)
    filelist = random.sample(content, len(content))
    N = len(filelist)
    X = np.zeros((N, 1, 32, 32), dtype=np.int64) # from uint8 turn into int64
    Y = np.zeros(N, dtype=np.int64)
    count = 0
    for fname in filelist:
        X[count] = np.load(path+'/'+fname)['data']
        Y[count] = classdict[str(np.load(path+'/'+fname)['classname'])]
        count += 1
    return X, Y, N

def compute_mean(X):
    data_mean = np.mean(X, axis=0)
    return data_mean

def shift_data(data, mean):
    new_data = data - mean
    return new_data


def creat_hdf5(option, X, Y, N, batsz):
    for i in range(N):
        if (i % batsz) == 0:
            filestr = option+'_dir/'+option+'_%07d'%i+'.h5'
            with h5py.File(filestr,'w') as f:
                try:
                    f['data'] = X[i:i+batsz]
                    f['label'] = Y[i:i+batsz]
                    print('batch %d finished'%i)
                except:
                    f['data'] = X[i:]
                    f['label'] = Y[i:]
                    print('batch %d finished'%i)

train_path = '/home/closerbibi/3D/understanding/rankpooling/python/train_dir'
test_path = '/home/closerbibi/3D/understanding/rankpooling/python/test_dir'

classdict = {
        'bathtub': 1,
        'bed': 2,
        'chair': 3,
        'desk': 4,
        'dresser': 5,
        'monitor': 6,
        'night_stand': 7,
        'sofa': 8,
        'table': 9,
        'toilet': 10
        }
if not os.path.exists('train_dir'):
        os.makedirs('train_dir')
if not os.path.exists('test_dir'):
        os.makedirs('test_dir')

#training
pdb.set_trace()
X,Y,N = loadtoXY('train', classdict)
pdb.set_trace()
data_mean = compute_mean(X)
new_X = shift_data(X, data_mean)
creat_hdf5('train', new_X, Y, N, 200)

#testing
X,Y,N = loadtoXY('test', classdict)
data_mean = compute_mean(X)
new_X = shift_data(X, data_mean)
creat_hdf5('test', X, Y, N, 200)
