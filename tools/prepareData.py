import scipy.io as sio
import os

data = h5py.File('/media/closerbibi/internal3/3D/understanding/rankpooling/rgbd_data/nyu_v2_labeled/nyu_depth_v2_labeled.mat','r')

depths=data.get('depths')
depths=np.array(depths)

K = [[5.7616540758591043e+02,0,3.2442516903961865e+02],[0,5.7375619782082447e+02,2.3584766381177013e+02],[0,0,1]];

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


count=1;
for imagenum in xrange(len(data['depths'],3)):
    target = indices(data['labels']) 


