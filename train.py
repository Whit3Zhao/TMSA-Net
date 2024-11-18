import copy
import os
import time
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch import nn

import config
from util import data_augmentation

def train_evaluation(model: nn.Module, train_loader, test_loader, criterion, optimizer, scheduler, save_path,
                     epochs=config.epochs):
    # Initialize log writer if a save path is provided
    log_writer = None
    if save_path is not None:
        log_writer = open(os.path.join(save_path, 'log.txt'), 'w')

    best_acc = 0  # Variable to store the best accuracy
    avg_acc = 0  # Variable to calculate the average accuracy
    best_model = None  # Variable to store the best model's state dictionary

    # Lists to record loss and accuracy for visualization
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        start_time = time.time()
        model.train()  # Set the model to training mode
        train_loss = 0  # Initialize training loss for the current epoch
        train_predicted = []  # List to store predicted labels during training
        train_actual = []  # List to store actual labels during training

        # Training loop
        with torch.enable_grad():
            for train_data, train_labels in train_loader:
                # Perform data augmentation and concatenate augmented data
                aug_data, aug_labels = data_augmentation(train_data, train_labels)
                train_data = torch.cat((train_data, aug_data), dim=0).float().cuda()
                train_labels = torch.cat((train_labels, aug_labels), dim=0).long().cuda()

                # Forward pass
                output = model(train_data)
                loss = criterion(output, train_labels)

                # Clip gradients and update model weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                torch.cuda.synchronize()
                train_loss += loss.item()
                train_predicted.extend(torch.max(output, 1)[1].cpu().tolist())
                train_actual.extend(train_labels.cpu().tolist())

        train_loss /= len(train_loader)  # Compute average training loss
        train_losses.append(train_loss)

        train_acc = accuracy_score(train_actual, train_predicted)  # Compute training accuracy
        train_accuracies.append(train_acc)

        if scheduler:
            scheduler.step()  # Update the learning rate scheduler

        model.eval()  # Set the model to evaluation mode
        test_loss = 0  # Initialize test loss for the current epoch
        test_predicted = []  # List to store predicted labels during testing
        test_actual = []  # List to store actual labels during testing

        # Testing loop
        with torch.no_grad():
            for test_data, test_labels in test_loader:
                test_data = test_data.float().cuda()
                test_labels = test_labels.long().cuda()

                # Forward pass
                test_output = model(test_data)
                loss = criterion(test_output, test_labels)

                test_predicted.extend(torch.max(test_output, 1)[1].cpu().tolist())
                test_actual.extend(test_labels.cpu().tolist())
                test_loss += loss.item()

        test_loss /= len(test_loader)  # Compute average test loss
        test_losses.append(test_loss)

        test_acc = accuracy_score(test_actual, test_predicted)  # Compute test accuracy
        test_accuracies.append(test_acc)
        test_kappa = cohen_kappa_score(test_actual, test_predicted)  # Compute Cohen's kappa score

        avg_acc += test_acc

        # Save the best model if current accuracy is the highest
        if test_acc > best_acc:
            best_acc = test_acc
            best_kappa = test_kappa
            best_model = copy.deepcopy(model.state_dict())

        print('Epoch [%d] | Train Loss: %.6f  Train Accuracy: %.6f | Test Loss: %.6f  Test Accuracy: %.6f | lr: %.6f | Time: %.2f s'
              % (epoch + 1, train_loss, train_acc, test_loss, test_acc, optimizer.param_groups[0]['lr'], (time.time() - start_time)))
        
        if log_writer and epoch % 50 == 0:
            log_writer.write(
                f'Epoch [{epoch + 1}] | Train Loss: {train_loss:.6f}  Train Accuracy: {train_acc:.6f} | Test Loss: {test_loss:.6f} Test Accuracy: {test_acc:.6f} Test Kappa: {test_kappa:.6f} \n')

        # Stop training early if test accuracy reaches 100%
        if test_acc == 1.0:
            print("Test accuracy reached 100%, stopping early.")
            avg_acc /= (epoch + 1)
            break

    avg_acc /= epochs
    print('The average accuracy is: ', avg_acc)
    print('The best accuracy is: ', best_acc)
    if log_writer:
        log_writer.write(f'The average accuracy is: {avg_acc:.6f}\n')
        log_writer.write(f'The best accuracy is: {best_acc:.6f}\n')
        log_writer.write(f'The best kappa is: {best_kappa:.6f}\n')
        log_writer.close()

    # Save the best model's state dictionary
    torch.save(best_model, os.path.join(save_path, 'model.pth'))

    # Plot and save training and testing metrics
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_plots.png'))
    plt.show()

    return best_acc, best_kappa
