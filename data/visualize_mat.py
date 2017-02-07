import os
import scipy.io as sio
import matplotlib.pyplot as plt
import pdb

#picture_000969_01_bed.mat
data_path = 'picture_roi/'
scene_path = 'picture_readytotrain/picture_'

'''
plt.figure(1)
img1 = sio.loadmat(data_path + 'picture_000001_01_chair.mat')['target_grid']
plt.imshow(img1)

plt.figure(2)
img2 = sio.loadmat(data_path + 'picture_000001_02_chair.mat')['target_grid']
plt.imshow(img2)

plt.figure(3)
sceneimg = sio.loadmat(scene_path + '000001.mat')['grid']
plt.imshow(sceneimg)
plt.title('000001')
plt.show()
'''
for fname in os.listdir(data_path):
    plt.figure(1)
    img = sio.loadmat(data_path + fname)['target_grid']
    plt.imshow(img)

    plt.figure(2)
    sceneimg = sio.loadmat(scene_path + fname.split('_')[1] + '.mat')['grid']
    plt.imshow(sceneimg)
    plt.title(fname)
    plt.show()
