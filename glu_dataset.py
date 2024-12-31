import os
import torch
import logging
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class gluDataset(Dataset):
    def __init__(self, data_dir, mode='train', block_size=56, start_from=7):

        self.block_size = block_size
        self.data_dir = data_dir
        self.start_from = start_from
    
        if mode not in ['train', 'val', 'test']:
            raise ValueError('Invalid mode: %s' % mode)
        self.mode = mode

        self._load_data()

    def __len__(self):
        return len(self.basic_datas)
    

    def _load_data(self):

        if self.mode == 'train':
            basic_info = pd.read_csv(os.path.join(self.data_dir, 'basic_train.csv'))
        elif self.mode == 'val':
            basic_info = pd.read_csv(os.path.join(self.data_dir, 'basic_val.csv'))
        else:
            basic_info = pd.read_csv(os.path.join(self.data_dir, 'basic_test.csv'))
        basic_info.fillna(0, inplace=True)
        
        basic_datas = []
        ts_datas = []

        for idx in range(len(basic_info)):
            basic_data = basic_info.iloc[idx]
            id = basic_data['id']
            basic_datas.append(basic_data)

            ts_data = pd.read_csv(os.path.join(self.data_dir, str(int(id)) + '.csv'))

            ts_data['time_group'] = ts_data['time_group'] + 1
            ts_data['y'] = ts_data['glucose'].shift(-1)
            ts_data['mask'] = ts_data['glucose'].apply(lambda x: 0 if np.isnan(x) else 1).shift(-1)

            # mask the first $start_from points
            ts_data.loc[ts_data.index < self.start_from, 'mask'] = 0

            # padding to max length
            pad = self.block_size - ts_data.shape[0]
            ts_data.fillna(0, inplace=True)
            if pad > 0:
                ts_data = pd.concat([ts_data, pd.DataFrame(np.zeros((pad, ts_data.shape[1]), dtype=np.float32), columns=ts_data.columns)], axis=0)

            ts_datas.append(ts_data)
        
        self.basic_datas = basic_datas
        self.ts_datas = ts_datas
        assert len(basic_datas) == len(ts_datas) 
        print('Load data done...')
        print(f'The size of {self.mode} dataset is {len(basic_datas)}')

    def __getitem__(self, idx):

        basic_data = self.basic_datas[idx]
        ts_data = self.ts_datas[idx]
        basic_data.drop(columns=['id'], inplace=True)
        basic_data = basic_data.values

        posd = ts_data['date'].values
        post = ts_data['time_group'].values
        label = ts_data['y'].values
        mask = ts_data['mask'].values   
        ts_data = ts_data.drop(columns=['date', 'time_group', 'y', 'mask'])
        ts_data = ts_data.values
        
        return{
            'basic_info': torch.from_numpy(basic_data).float(),
            'ts_data': torch.from_numpy(ts_data).float(),
            'posd': torch.from_numpy(posd).int(),
            'post': torch.from_numpy(post).int(),
            'label': torch.from_numpy(label).float(),
            'mask': torch.from_numpy(mask).float()
        }



        
### DEBUG

# data_dir = '/remote-home/hongquanliu/Datasets/ZS_DATA/collect_by_id'
# dataset = gluDataset(data_dir, mode='train')
# print(len(dataset))

# from torch.utils.data import DataLoader
# dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
# for i, data in enumerate(dataloader):
#     print(data['basic_info'].shape)
#     print(data['ts_data'].shape)
#     print(data['posd'].shape)
#     print(data['post'].shape)
#     print(data['label'].shape)
#     print(data['mask'].shape)
#     # label = [data['label'][0][i] * 1.7 + 4.4 if data['post'][0][i] in [1, 3, 5] else data['label'][0][i] * 1.7 + 6.1 for i in range(data['label'].shape[1])]
#     # print(label[:5])
#     print(data['label'][0][:5])
#     print(data['posd'][0][:5])
#     print(data['post'][0][:5])
#     break
