import os

import scipy.misc
import scipy.io as scio
import mat73


root = '/mnt/data/publicdata/NYUv2Seg'

allsplit_dir = os.path.join(root, 'nyu_depth_v2_labeled.mat')
data_dict = mat73.loadmat(allsplit_dir)

#split = scipy.io.loadmat(allsplit_dir)

print(data_dict.keys())


'''
split_train = split['alltrain']


split_train = [f.replace('/n/fs/sun3d/data/SUNRGBD', self.root + '/train') for f in split_train]
split_train = [f[:-1] if f[-1] == '/' else f for f in split_train]
print(len(split_train), split_train)
test_split, train_split = [], []
'''
