import numpy as np
import lmdb
import caffe
import os
import random
import h5py
import pdb
import scipy.io as sio

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

def loadtoXY_mat(option, classdict):
    if option == 'train':
        path = train_path
    elif option == 'test':
        path = test_path

    # shuffle the data
    content = os.listdir(path)
    filelist = random.sample(content, len(content))
    N = len(filelist)
    X = np.zeros((N, 1, 112, 112), dtype=np.int64) # from uint8 turn into int64
    Y = np.zeros(N, dtype=np.int64)
    count = 0
    for fname in filelist:
        tmpX = sio.loadmat(path+'/'+fname)['target_grid']
        tmpX = cv2.resize(tmpX,(112,112))
        X[count] = tmpX
        Y[count] = classdict[str(fname.split('.')[0].split('_')[-1])]
        count += 1
        print 'file %s done' % fname
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
                    f['data'] = X[i:i+batsz-1,:,:,:]
                    f['label'] = Y[i:i+batsz-1]
                    print('batch %d finished'%i)
                except:
                    f['data'] = X[i:,:,:,:]
                    f['label'] = Y[i:]
                    print('batch %d finished'%i)

train_path = '../picture_roi'
test_path = '../picture_roi'

# bed=157, chair=5, table=19, sofa=83, toilet=124
classdict = {
        'chair': 1,
        'table': 2,
        'sofa': 3,
        'toilet': 4,
        'bed': 5,
        }
if not os.path.exists('train_dir'):
        os.makedirs('train_dir')
if not os.path.exists('test_dir'):
        os.makedirs('test_dir')

#training
X,Y,N = loadtoXY_mat('train', classdict)
data_mean = compute_mean(X)
new_X = shift_data(X, data_mean)
creat_hdf5('train', new_X, Y, N, 10)

#testing
X,Y,N = loadtoXY_mat('test', classdict)
data_mean = compute_mean(X)
new_X = shift_data(X, data_mean)
creat_hdf5('test', X, Y, N, 10)
