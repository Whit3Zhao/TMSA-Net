import os
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from eeg_dataset import seed_torch, load_data, eegDataset
from train import train_evaluation


# Build learning rate scheduler using Cosine Annealing
def build_lr_scheduler(optimizer, n_iter_per_epoch):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter_per_epoch * config.epochs)
    return scheduler

# Build dataset file paths for training or testing
def build_datasets_files(stage='train'):
    datasets = []  # List to store file paths for each subject
    target_file = config.train_files if stage == 'train' else config.test_files
    for dir in sorted(os.listdir(config.data_path)):
        if '.' in dir:
            continue
        data_files = []
        for file in sorted(os.listdir(config.data_path + dir)):
            if file in target_file:
                data_files.append(config.data_path + dir + '/' + file)
        if data_files:
            datasets.append(data_files)
    return datasets

def main():
    # Set the CUDA device to be used
    torch.cuda.set_device(0)
    randomSeed = random.randint(1, 10000)
    print(f'seed is {randomSeed}')
    seed_torch(randomSeed)  # Set the random seed for reproducibility
    print(f'device {0} is used for training')

    accuracy = []  # List to store accuracy for each subject
    kappa = []  # List to store kappa for each subject

    # Build dataset file lists for training and testing
    train_datasets = build_datasets_files(stage='train')
    test_datasets = build_datasets_files(stage='test')

    for i in range(len(train_datasets)):
        subject = train_datasets[i][0].split('/')[-2]
        print(f'------start {subject} training------')

        # Create directory for saving model
        save_path = os.path.join(config.output, 'models', config.model_name, subject)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        train_file = train_datasets[i]
        test_file = test_datasets[i]

        # Load training and testing data
        train_data, train_labels = load_data(train_file[0])
        test_data, test_labels = load_data(test_file[0])

        # Create datasets and data loaders
        train_dataset = eegDataset(torch.from_numpy(train_data).cuda(), torch.from_numpy(train_labels).long().cuda())
        test_dataset = eegDataset(torch.from_numpy(test_data).cuda(), torch.from_numpy(test_labels).long().cuda())

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        # Initialize the model
        from network.TMSANet import TMSANet
        # ---- Model Initialization Tips ---- #
        # - Always set `radix = 1`.
        # - If using the BCIC-IV-2b dataset:
        #   Initialize the model as: TMSANet(3, 1, 1000, 2)
        #   where:
        #     - 3: Number of input channels
        #     - 1: Radix value
        #     - 1000: Number of time points
        #     - 2: Number of classes
        # - If using the HGD dataset:
        #   Initialize the model as: TMSANet(44, 1, 1125, 4)
        #   where:
        #     - 44: Number of input channels
        #     - 1: Radix value
        #     - 1125: Number of time points
        #     - 4: Number of classes
        model = TMSANet(22, 1, 1000, 4)
        print('\n', model)

        # Move model to GPU
        model.cuda()

        # Calculate and print the total number of trainable parameters
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        # Set up loss function, optimizer, and learning rate scheduler
        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
        scheduler = build_lr_scheduler(optimizer, len(train_loader))

        # Train and evaluate the model
        best_acc, best_kappa = train_evaluation(model, train_loader, test_loader, criterion, optimizer, scheduler, save_path)
        accuracy.append(best_acc)
        kappa.append(best_kappa)

    # Print accuracy and kappa for each subject
    for i in range(len(accuracy)):
        print(f'subject:A0{i+1},accuracy:{accuracy[i]:.6f},kappa:{kappa[i]:.6f}')
    
    # Print average accuracy and kappa
    print('average accuracy:', sum(accuracy) / len(accuracy), 'average kappa:', sum(kappa) / len(kappa), 'seed:', randomSeed)

if __name__ == '__main__':
    main()
