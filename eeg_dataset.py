from scipy import io
import random
import numpy as np
import torch
from torch.utils.data import Dataset

# Dataset class for EEG data
class eegDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.labels = label  # Store labels
        self.data = data  # Store data

    def __getitem__(self, index):
        # Retrieve a single data-label pair
        data = self.data[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        # Return the total number of data samples
        return len(self.data)

# Function to set random seed for reproducibility
def seed_torch(seed):
    random.seed(seed)  # Set Python's built-in random seed
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)  # Disable hash randomization
    np.random.seed(seed)  # Set NumPy random seed
    torch.manual_seed(seed)  # Set PyTorch CPU random seed
    torch.cuda.manual_seed(seed)  # Set PyTorch GPU random seed
    torch.backends.cudnn.benchmark = False  # Disable CUDNN benchmark for reproducibility
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior in CUDNN

# Function to load EEG data from a .mat file
def load_data(data_file, stage='test'):
    print(f'loading data from: {data_file}')
    # Load data from the .mat file
    data = io.loadmat(data_file)

    # label numbers must start at 0
    # EEG_data original shape is [22, 1000, 288] and label shape is [288, 1]
    # EEG = data['EEG_data'].astype(np.float64) # 2a data tag
    EEG = data['data_set'].astype(np.float64)  # 2b data tag
    # EEG = data['data'].astype(np.float64) # HGD data tag
    # shape change to [288, 22, 1000]
    EEG = EEG.transpose((2, 0, 1))  # When you use HGD, please not using this row
    # shape change to [288,]
    labels = data['labels_set'].reshape(-1).astype(np.int32)  # 2b labels tag
    # labels = data['labels'].reshape(-1).astype(np.int32)  # N 2a labels tag
    # labels = data['labels'].reshape(-1).astype(np.int32) # HGD labels tag
    # index start from 0
    labels = labels - np.min(labels)

    # Shuffle EEG data and labels
    EEG, labels = shuffle_data(EEG, labels)

    print(f'preprocessing data: {EEG.shape} {labels.shape}')
    return EEG, labels

# Function to shuffle data and labels in unison
def shuffle_data(data, label):
    index = [i for i in range(len(data))]  # Create an index list
    random.shuffle(index)  # Shuffle the index list
    shuffle_data = data[index]  # Rearrange data using shuffled indices
    shuffle_label = label[index]  # Rearrange labels using shuffled indices
    return shuffle_data, shuffle_label
