import numpy as np
import lmdb
import caffe
import os
import pdb
import cv2
import shutil
import scipy.io as sio
import scipy.misc as smi

def get_N(option):
    if option == 'train':
        return 50000
    elif option == 'test':
        return 20000

def loadtoXY4voxeldata(option, classdict):
    # Let's pretend this is interesting data
    N = get_N(option)
    X = np.zeros((N, 1, 32, 32), dtype=np.int64) # from uint8 turn into int64
    Y = np.zeros(N, dtype=np.int64)
    if option == 'train':
        path = train_path
    elif option == 'test':
        path = test_path
    count = 0
    for fname in os.listdir(path):
        X[count] = np.load(path+'/'+fname)['data']
        Y[count] = classdict[str(np.load(path+'/'+fname)['classname'])]
        count += 1
    return X, Y, N
def loadtoXY(option, classdict):
    # Let's pretend this is interesting data
    if option == 'train':
        path = train_path
    elif option == 'test':
        path = test_path
    filelist = np.sort(os.listdir(path))
    N = 100#len(filelist)#get_N(option)
    X = np.zeros((N, 1, 256, 256), dtype=np.uint8) # from uint8 turn into int64
    Y = np.zeros(N, dtype=np.int64)

    # loading data and label
    count = 0
    for fname in filelist:
        tmpX = sio.loadmat(path+'/'+fname)['target_grid']
        tmpX = smi.imresize(tmpX,(256,256))
        X[count] = tmpX

        # label
        Y[count] = classdict[fname.split('.')[0].split('_')[-1]]
        print 'file %s done' % fname
        count += 1
        if count == N:
            break
    return X, Y, N

def creat_lmdb(option, X, Y, N):
    # We need to prepare the database for the size. We'll set it 10 times
    # greater than what we theoretically need. There is little drawback to
    # setting this too big. If you still run into problem after raising
    # this, you might want to try saving fewer entries in a single
    # transaction.
    map_size = X.nbytes * 10

    env = lmdb.open(option+'_lmdb', map_size=map_size)

    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in range(N):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = X.shape[1]
            datum.height = X.shape[2]
            datum.width = X.shape[3]
            datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
            datum.label = int(Y[i])
            str_id = '{:08}'.format(i)
            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())


#train_path = '/home/closerbibi/3D/understanding/rankpooling/python/train_dir'
#test_path = '/home/closerbibi/3D/understanding/rankpooling/python/test_dir'
test_path = '../picture_roi'


if os.path.exists('train_lmdb'):
    shutil.rmtree('train_lmdb')
if os.path.exists('test_lmdb'):
    shutil.rmtree('test_lmdb')
os.makedirs('train_lmdb')
os.makedirs('test_lmdb')


classdict = {
        'chair': 1,
        'bed': 2,
        'table': 3,
        'toilet': 4,
        'sofa': 5,
        }
#X,Y,N = loadtoXY('train', classdict)
#creat_lmdb('train', X, Y, N)
X,Y,N = loadtoXY('test', classdict)
creat_lmdb('test', X, Y, N)
