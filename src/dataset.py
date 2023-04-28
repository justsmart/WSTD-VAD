import torch.utils.data as data
import numpy as np
from utils import process_feat
import torch
from torch.utils.data import DataLoader
# torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):

        self.is_normal = is_normal
        self.dataset = args.dataset
        # self.featmode = args.featmode
        if self.dataset == 'sh':
            if test_mode:
                self.rgb_list_file = '../list/sh-{}-test.list'.format(args.featmode)
            else:
                self.rgb_list_file = '../list/sh-{}-train.list'.format(args.featmode)
        elif self.dataset == 'ucf':
            if test_mode:
                self.rgb_list_file = '../list/ucf-{}-test.list'.format(args.featmode)
            else:
                self.rgb_list_file = '../list/ucf-{}-train.list'.format(args.featmode)

        elif self.dataset == 'xd':
            if test_mode:
                self.rgb_list_file = '../list/xd-{}-test.list'.format(args.featmode)
            else:
                self.rgb_list_file = '../list/xd-{}-train.list'.format(args.featmode)
        else:
            raise
        self.tranform = transform
        self.test_mode = test_mode
        self.num_ab = args.abvideonum
        self._parse_list()
        self.num_frame = 0
        self.labels = None



    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.dataset == 'sh':
                if self.is_normal:
                    self.list = self.list[63:]
                    print('normal list for ',self.dataset)
                    # print(self.list)
                else:
                    if self.num_ab!=-1:
                        self.list = self.list[:self.num_ab]
                    else:
                        self.list = self.list[:63]
                    print('abnormal list for ',self.dataset)
                    print(len(self.list))

            elif self.dataset == 'ucf' or self.dataset == 'ucf-o':
                if self.is_normal:
                    self.list = self.list[810:]
                    print('normal list for ',self.dataset)
                    # print(self.list)
                else:
                    self.list = self.list[:810]
                    print('abnormal list for ',self.dataset)
                    # print(self.list)
            elif self.dataset == 'xd':
                if self.is_normal:
                    nor_list = []
                    for npy in self.list:
                        if npy.split('_')[-1]=='A.npy\n':
                            nor_list.append(npy)
                    self.list=nor_list
                    print('normal list for ',self.dataset,len(self.list))
                    # print(self.list)
                else:
                    abn_list = []
                    for npy in self.list:
                        if npy.split('_')[-1]!='A.npy\n':
                            abn_list.append(npy)
                    self.list=abn_list
                    print('abnormal list for ',self.dataset,len(self.list))
                    # print(self.list)
            else:
                raise


    def __getitem__(self, index):
        try:
            label = self.get_label()  # get video level label 0/1
            features = np.load(self.list[index].strip('\n'), allow_pickle=True)
            features = np.array(features, dtype=np.float32)
        except:
            print('error file:',self.list[index].strip('\n'))
        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features
        else:
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F]
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)

            return divided_features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
