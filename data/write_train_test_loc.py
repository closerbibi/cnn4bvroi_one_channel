import os

def write_file(option, path):
    with open(option+'_loc.txt', 'w') as f:
        for fname in os.listdir(path):
            f.write('data/' + path+'/'+fname+'\n')

        f.close()


write_file('train', 'hdf5/train_dir')
write_file('test', 'hdf5/test_dir')

