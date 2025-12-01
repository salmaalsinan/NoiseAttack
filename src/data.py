import os
from torch.utils.data import Dataset
from utils import *
import yaml
import torch

# Read YAML file
with open(os.path.join("../config/", "global_config.yml"), 'r') as stream:
    data_loaded = yaml.safe_load(stream)
SEISMICROOT = data_loaded['SEISMICROOT']
#SEISMICDIR = os.path.join(SEISMICROOT, 'normalized_resized_slices/') #full dataset presplit
SEISMICDIR = os.path.join(SEISMICROOT, 'Splits/Train_Val/')

FIELDDIR = os.path.join(SEISMICROOT, 'fielddata/')

#SEISMICDIR = os.path.join(SEISMICROOT, 'normalized_resized_slices/') # full dataset pre-split (You can select it by choosing Dataclip=False in train.py)
#SEISMICDIRTEST = os.path.join(SEISMICROOT, 'Splits/Test/') # test data you can specify it through root dorectory in get_data set
#denoise_dataset = get_dataset(problem, noise_transforms=noise,rootdir='../Data/Splits/Test/')

class BaseLoader(Dataset):
    def read_input(self, idx):
        pass

    def read_target(self, idx):
        pass

class FirstBreakLoader(BaseLoader):
    def __init__(self, rootdir, transform=None):
        self.rootdir = rootdir
        self.inputdir = os.path.join(self.rootdir, 'input/')
        self.targetdir = os.path.join(self.rootdir, 'target/')
        self.inputs = sorted(os.listdir(self.inputdir))
        self.targets = sorted(os.listdir(self.targetdir))
        self.transform = transform
        self.class_names = ['empty', 'wave']

    def __len__(self):
        assert len(self.targets) == len(self.inputs)
        return len(self.inputs)

    def read_input(self, idx):
        image = np.load(os.path.join(self.inputdir,self.inputs[idx]))
        return image[...,None]

    def read_target(self, idx):
        return np.load(os.path.join(self.targetdir,self.targets[idx]))

    def __getitem__(self, idx):
        image = self.read_input(idx)
        target = self.read_target(idx)
        sample = {'input': image, 'target': target}
        return self.transform(sample) if self.transform else sample
    
class DenoiseLoader(FirstBreakLoader):
    def __init__(self, *pargs, **kwargs):
        super(DenoiseLoader, self).__init__(*pargs, **kwargs)
        self.targetdir = self.inputdir
        self.targets = self.inputs
        self.transform = self.transform

    def read_target(self, idx):
        return super(DenoiseLoader, self).read_target(idx)[...,None]

class NoiseLoader(FirstBreakLoader):
    def __init__(self, *pargs, **kwargs):
        super(NoiseLoader, self).__init__(*pargs, **kwargs)
        self.targetdir = self.inputdir
        self.targets = self.inputs
        self.transform = self.transform
        
    def read_input(self, idx):
        image = np.load(os.path.join(self.inputdir,self.inputs[idx]))
        return np.zeros_like(image)[...,None]
        
    def read_target(self, idx):
        return super(NoiseLoader, self).read_target(idx)[...,None]
    def __getitem__(self, idx):
        image = self.read_input(idx)
        target = self.read_target(idx)
        sample = {'input': image, 'target': target}
        return self.transform(sample) if self.transform else sample
    
def get_first_break_dataset(rootdir=None,
                            target_size=(224, 224),
                            noise_transforms=[]):
    if rootdir is None:
        rootdir = SEISMICDIR
    transforms_ = []
    transforms_ += noise_transforms
    transforms_ += [ChangeType(problem='segment')]
    transforms_ += [ScaleNormalize('input')]
    transforms_ += [FlipChannels(only_input=True), ToTensor()]
    return FirstBreakLoader(rootdir, transform=transforms.Compose(transforms_))


class RealDataLoader(BaseLoader):
    def __init__(self, rootdir, transform=None):
        self.rootdir = rootdir
        self.inputdir = os.path.join(self.rootdir, 'input/')
        self.targetdir = os.path.join(self.rootdir, 'input/')
        self.inputs = os.listdir(self.inputdir)
        self.targets = os.listdir(self.targetdir)
        self.transform = transform
        self.class_names = ['empty', 'wave']

    def __len__(self):
        assert len(self.targets) == len(self.inputs)
        return len(self.inputs)

    def read_input(self, idx):
        image = np.load(os.path.join(self.inputdir, self.inputs[idx]))
        return image[..., None]

    def read_target(self, idx):
        image = np.load(os.path.join(self.targetdir, self.targets[idx]))
        return image[..., None]

    def __getitem__(self, idx):
        image = self.read_input(idx)
        target = self.read_target(idx)
        sample = {'input': image, 'target': target}
        return self.transform(sample) if self.transform else sample


def get_denoise_dataset(rootdir=None,
                       noise_transforms=[]):
    if rootdir is None:
        rootdir = SEISMICDIR
    transforms_ = []
    transforms_ += noise_transforms
    transforms_ += [ChangeType()]
    transforms_ += [ScaleNormalize('input'), ScaleNormalize('target')]
    transforms_ += [FlipChannels(), ToTensor()]
    return DenoiseLoader(rootdir, transform=transforms.Compose(transforms_))

def get_noise_dataset(rootdir=None,
                       noise_transforms=[]):
    if rootdir is None:
        rootdir = SEISMICDIR
    transforms_ = []
    transforms_ += noise_transforms
    transforms_ += [ChangeType()] #comment if you dont want to scale
    transforms_ += [ScaleNormalize('input'), ScaleNormalize('target')] #comment if you dont want to scale
    transforms_ += [FlipChannels(), ToTensor()]
    return NoiseLoader(rootdir, transform=transforms.Compose(transforms_))


def get_real_dataset(rootdir=FIELDDIR,
                            target_size=(224, 224),
                            noise_transforms=[]):
    transforms_ = []
    transforms_ += noise_transforms
    transforms_ += [FlipChannels(), ToTensor()]
    return RealDataLoader(rootdir, transform=transforms.Compose(transforms_))


def get_dataset(dtype, *pargs, **kwargs):
    if dtype == 'firstbreak':
        dataset = get_first_break_dataset(*pargs, **kwargs)
    elif dtype == 'denoise':
        dataset = get_denoise_dataset(*pargs, **kwargs)
    elif dtype == 'real':
        dataset = get_real_dataset(*pargs, **kwargs)
    elif dtype == 'noise':
        dataset = get_noise_dataset(*pargs, **kwargs)
    else:
        raise ValueError("Unknown Dataset Type")
    return dataset


def get_train_val_dataset(dataset, valid_split=0.1, seed=911, **kwargs):
    """
    Split a PyTorch dataset into train and validation subsets with a reproducible seed.
    
    :param dataset: PyTorch Dataset
    :param valid_split: fraction of dataset for validation
    :param seed: random seed for reproducibility
    :param kwargs: additional arguments passed to random_split (e.g., generator)
    :return: (train_dataset, val_dataset)
    """
    train_size = int((1 - valid_split) * len(dataset))
    val_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(seed)
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
        **kwargs
    )
    
    return train_dataset, val_dataset
