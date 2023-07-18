import os
import torch
from torch.utils.data import Dataset
from pathlib import Path

from torch.utils.data import DataLoader
import cv2

from torchvision.transforms import Resize, RandomHorizontalFlip, RandomVerticalFlip
import matplotlib.pyplot as plt
import h5py
import numpy as np


class DIV2K(Dataset):
    def __init__(self, dir, transforms: list=None) -> None:
        self.dir = dir
        self.transforms = transforms

        #TODO add mean and stf for all other datasets

    def __len__(self):
        return len([name for name in os.listdir(self.dir/'LR')])

    def __getitem__(self, index):
        lr = torch.tensor(
            cv2.imread(str(self.dir/'LR'/f'{index + 1:04d}x3.png')), dtype=torch.float32).permute(2,0,1)
        hr = torch.tensor(
            cv2.imread(str(self.dir/'HR'/f'{index + 1:04d}.png')), dtype=torch.float32).permute(2,0,1)

        if self.transforms:
            mslr = self.transforms[0](mslr)
            hr = self.transforms[1](hr)

        return (mslr, hr)


'''if __name__ == "__main__":
    batch_size = 8
    shuffle = True

    dir_tr = Path(f'F:/Data/DIV2K/train')
    dir_val = Path(f'F:/Data/DIV2K/val')

    tr_dataset = DIV2K(
        dir_tr, transforms=(Resize((512, 512)), RandomHorizontalFlip(p=1.)))
    train_loader = DataLoader(
        dataset=tr_dataset, shuffle=shuffle)
    
    val_dataset = DIV2K(
        dir_tr, transforms=(Resize((512, 512)), RandomHorizontalFlip(p=1.)))
    validation_loader = DataLoader(
        dataset=val_dataset, shuffle=shuffle)

    # train shapes
    mslr, hr = next(iter(train_loader))
    print(mslr.shape, hr.shape)

    # validation shapes
    mslr, hr = next(iter(validation_loader))
    print(mslr.shape, hr.shape)'''


class GaoFen2(Dataset):
    def __init__(self, dir, transforms=None) -> None:
        f = h5py.File(str(dir),'r+')
        self.hr = torch.tensor(f['gt'][()], dtype=torch.float32)
        self.mslr = torch.tensor(f['ms'][()], dtype=torch.float32)
        self.pan = torch.tensor(f['pan'][()], dtype=torch.float32)
        self.transforms = transforms

        #precomputed
        self.pan_mean = torch.tensor([250.0172]).view(1,1,1,1)
        self.pan_std = torch.tensor([80.2501]).view(1,1,1,1)

        self.mslr_mean = torch.tensor([449.9449, 308.7544, 238.3702, 220.3061]).view(1,4,1,1)
        self.mslr_std = torch.tensor([70.8778, 63.7980, 71.3171, 66.8198]).view(1,4,1,1)

    def __len__(self):
        return self.mslr.shape[0]

    def __getitem__(self, index):

        pan = self.pan[index]
        mslr = self.mslr[index]
        hr = self.hr[index]

        if self.transforms:
            for transform, prob in self.transforms:
                if torch.randn(1) < prob:
                    pan = transform(pan)
                    mslr = transform(mslr)
                    hr = transform(hr)

        return (pan, mslr, hr)

"""if __name__ == "__main__":
    batch_size = 1
    shuffle = True

    dir_tr = Path(f'F:/Data/GaoFen-2/train/train_gf2-001.h5')
    dir_val = Path(f'F:/Data/GaoFen-2/val/valid_gf2.h5')
    #dir_test = Path(f'F:/Data/GaoFen-2/train/train_gf2-001.h5')

    tr_dataset = GaoFen2(
        dir_tr, transforms=[(RandomHorizontalFlip(1), 0.3), (RandomVerticalFlip(1), 0.3)])
    train_loader = DataLoader(
        dataset=tr_dataset, batch_size=batch_size, shuffle=shuffle)
    
    val_dataset = GaoFen2(
        dir_val)
    validation_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=shuffle)

    '''# train shapes
    pan, mslr, hr = next(iter(train_loader))
    print(pan.shape, mslr.shape, hr.shape)

    # validation shapes
    pan, mslr, hr = next(iter(validation_loader))
    print(pan.shape, mslr.shape, hr.shape)'''


    channel_sum = 0
    channel_sum_of_squares = 0
    total_samples = 0
    # Iterate over the DataLoader
    print('Length of Dataloader: ', len(tr_dataset))
    for pan, mslr, mshr in tr_dataset:
        # Assuming your data is a tensor
        # Compute the channel-wise mean and mean of squares
        channel_sum += torch.mean(pan, dim=(1, 2))
        channel_sum_of_squares += torch.mean(pan ** 2, dim=(1, 2))

        total_samples += 1

    # Compute the mean and standard deviation for each channel
    mean = channel_sum / total_samples
    std = torch.sqrt((channel_sum_of_squares / total_samples) - mean ** 2)

    print('mean: ', mean, ' std: ', std)
"""

class GaoFen2panformer(Dataset):
    def __init__(self, dir, transforms=None) -> None:
        self.dir = dir
        self.transforms = transforms

        #precomputed
        self.pan_mean = torch.tensor([255.2780]).view(1,1,1,1)
        self.pan_std = torch.tensor([119.8152]).view(1,1,1,1)

        self.mslr_mean = torch.tensor([385.9424, 268.0104, 218.5947, 259.1452]).view(1,4,1,1)
        self.mslr_std = torch.tensor([134.2627, 110.1456, 117.1064, 113.4461]).view(1,4,1,1)

    def __len__(self):
        dt_len = len([name for name in os.listdir(self.dir/'LR')])
        print('dataset len: ',  dt_len)
        return dt_len
        
    def __getitem__(self, index):

        pan = torch.tensor(
            np.load(self.dir/'PAN'/f'{index:04d}.npy', allow_pickle=True).astype('float32'))
        mslr = torch.tensor(
            np.load(self.dir/'LR'/f'{index:04d}.npy', allow_pickle=True).astype('float32'))
        hr = torch.tensor(
            np.load(self.dir/'HR'/f'{index:04d}.npy', allow_pickle=True).astype('float32'))
        
        

        if self.transforms:
            for transform, prob in self.transforms:
                if torch.randn(1) < prob:
                    pan = transform(pan)
                    mslr = transform(mslr)
                    hr = transform(hr)

        return (pan, mslr, hr)#(None, None, None) #

if __name__ == "__main__":
    batch_size = 1

    dir_tr = Path(f'F:/Data/GaoFen-2_panformer/train/')
    dir_test = Path(f'F:/Data/GaoFen-2_panformer/test/')

    # Load training dataset
    tr_dataset = GaoFen2panformer(dir_tr)
    train_loader = DataLoader(
        dataset=tr_dataset, batch_size=batch_size, shuffle=True)

    # Load test dataset
    test_dataset = GaoFen2panformer(dir_test)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False)

    channel_sum = 0
    channel_sum_of_squares = 0

    lr_channel_sum = 0
    lr_channel_sum_of_squares = 0 

    total_samples = 0
    # Iterate over the DataLoader
    print('Length of Dataloader: ', len(train_loader))
    for pan, mslr, mshr in train_loader:
        # Assuming your data is a tensor
        # Compute the channel-wise mean and mean of squares
        channel_sum += torch.mean(pan)
        channel_sum_of_squares += torch.mean(pan ** 2)

        lr_channel_sum += torch.mean(mslr, dim=(0,2,3))
        lr_channel_sum_of_squares += torch.mean(mslr ** 2, dim=(0,2,3))

        total_samples += 1

    # Compute the mean and standard deviation for each channel
    mean = channel_sum / total_samples
    std = torch.sqrt((channel_sum_of_squares / total_samples) - mean ** 2)

    # Compute the mean and standard deviation for each channel
    lr_mean = lr_channel_sum / total_samples
    lr_std = torch.sqrt((lr_channel_sum_of_squares / total_samples) - lr_mean ** 2)

    print('mean: ', mean, ' std: ', std)
    print('mean: ', lr_mean, ' std: ', lr_std)
    


class Sev2Mod(Dataset):
    def __init__(self, dir, task, transform=None) -> None:
        self.dir = dir
        self.task = task
        self.transform = transform

    def __len__(self):
        return len([name for name in os.listdir(self.dir/'LR')])

    def __getitem__(self, index):

        pan = torch.tensor(
            np.load(self.dir/'PAN'/f'{index:04d}_{self.task}.npy', allow_pickle=True))
        mslr = torch.tensor(
            np.load(self.dir/'LR'/f'{self.task}'/f'{index:04d}_{self.task}.npy', allow_pickle=True))[:3,...]
        hr = torch.tensor(
            np.load(self.dir/'HR'/f'{self.task}'/f'{index:04d}_{self.task}.npy', allow_pickle=True))[:3,...]

        if self.transform:
            pan = self.transform(pan)
            mslr = self.transform(mslr)
            hr = self.transform(hr)

        return (pan, mslr, hr)

'''if __name__ == "__main__":
    batch_size = 8
    task = 'x3'

    dir_tr = Path(f'F:/Data/SEV2MOD/train/')
    dir_val = Path(f'F:/Data/SEV2MOD/train/')
    dir_test = Path(f'F:/Data/SEV2MOD/train/')

    # Load training dataset
    tr_dataset = Sev2Mod(dir_tr, task)
    train_loader = DataLoader(
        dataset=tr_dataset, batch_size=batch_size, shuffle=True)

    # Load validation dataset
    val_dataset = Sev2Mod(dir_val, task)
    validation_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    # Load test dataset
    test_dataset = Sev2Mod(dir_test, task)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # train shapes
    pan, mslr, hr = next(iter(train_loader))
    print(pan.shape, mslr.shape, hr.shape)

    # validation shapes
    pan, mslr, hr = next(iter(validation_loader))
    print(pan.shape, mslr.shape, hr.shape)

    #test shapes
    pan, mslr, hr = next(iter(test_dataset))
    print(pan.shape, mslr.shape, hr.shape)'''


class WV3(Dataset):
    def __init__(self, dir, transform=None) -> None:
        f = h5py.File(str(dir),'r+')
        self.hr = torch.tensor(f['gt'][()], dtype=torch.float32)
        self.mslr = torch.tensor(f['ms'][()], dtype=torch.float32)
        self.pan = torch.tensor(f['pan'][()], dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return self.mslr.shape[0]

    def __getitem__(self, index):

        pan = self.pan[index]
        mslr = self.mslr[index ]
        hr = self.hr[index]

        if self.transform:
            pan = self.transform(pan)
            mslr = self.transform(mslr)
            hr = self.transform(hr)

        return (pan, mslr, hr)

'''if __name__ == "__main__":
    batch_size = 8
    shuffle = True

    dir_tr = Path(f'F:/Data/WorldView3/train/train_wv3-001.h5')
    dir_val = Path(f'F:/Data/WorldView3/val/valid_wv3.h5')

    dataset = WV3(
        dir_tr)
    train_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    

    val_dataset = WV3(
        dir_val)
    validation_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=shuffle)

    # train shapes
    pan, mslr, hr = next(iter(train_loader))
    print(pan.shape, mslr.shape, hr.shape)

    # validation shapes
    pan, mslr, hr = next(iter(validation_loader))
    print(pan.shape, mslr.shape, hr.shape)'''