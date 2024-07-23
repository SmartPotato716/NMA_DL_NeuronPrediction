""" Created on Fri Jul 19 17:10:02 2024
    @author: dcupolillo """

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import numpy as np
import copy


def augment_trace(trace, noise_level=0.1):
    noise = torch.randn_like(trace) * noise_level
    augmented_trace = trace + noise
    return augmented_trace


def compute_l1_regularization(model):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return l1_norm


class decoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(50 * 250 * 6, 2)
        self.dropout2 = nn.Dropout(0.4)

    def forward(self, x):
        """
        x: input basal activity (250, 50) as (batch_size, time, neurons)
        """

        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = x.view(-1, 250 * 50 * 6)
        x = self.fc1(x)
        x = self.dropout2(x)

        return x


def train(
        train_loader,
        valid_loader,
        model,
        device,
        epochs,
        lr,
        lambda1=0.0,
        lambda2=0.01,  # between 0-1
        patience=20,
        augment_prob=0.5,
        noise_level=0.1
):
    """
    Training loop with early stopping based on validation accuracy

    Args:
    model: nn.Module
      Neural network instance
    device: str
      GPU/CUDA if available, CPU otherwise
    epochs: int
      Number of epochs
    train_loader: torch.utils.data.DataLoader
      Training Set
    valid_loader: torch.utils.data.DataLoader
      Validation set
    lr: float
      Learning rate
    patience: int
      Number of epochs to wait for improvement before stopping
    augment_prob: float
      Probability to apply Gaussian noise augmentation
    noise_level: float
      Standard deviation of the Gaussian noise to be added

    Returns:
    best_model: nn.Module
      The best model based on validation accuracy
    train_loss, validation_loss, train_acc, validation_acc: lists
      Lists of training and validation loss and accuracy
    """

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=lambda2)
    train_loss, validation_loss = [], []
    train_acc, validation_acc = [], []

    best_acc = 0.0
    best_epoch = 0
    best_model = copy.deepcopy(model)
    wait = 0

    start_time = time.time()

    for epoch in range(epochs):

        model.train()
        correct, total = 0, 0
        running_loss = 0.0

        # Looping through all the trials
        for data, target in tqdm(
                train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):

            data = data.float()
            data = data.unsqueeze(1)  # add a batch dimension
            data, target = data.to(device), target.to(device)

            if np.random.rand() < augment_prob:
                data = augment_trace(data, noise_level)

            # Output is the normalized probability
            output = model(data)

            optimizer.zero_grad()
            target = (target + 1) // 2  # Transferring -1 labels to 0
            target = target.long()
            loss = criterion(output, target)

            l1_norm = compute_l1_regularization(model)
            loss += lambda1 * l1_norm
            
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Get accuracy
            _, predicted = torch.max(output, dim=1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

        train_loss.append(running_loss)
        train_acc.append((correct / total) * 100)

        model.eval()
        correct, total = 0, 0
        running_loss = 0.0

        for data, target in valid_loader:

            data = data.float()
            data = data.unsqueeze(1)
            data, target = data.to(device), target.to(device)
            output = model(data)
            target = (target + 1) // 2
            target = target.long()
            loss = criterion(output, target)
            running_loss += loss.item()

            # We will not be calculating gradient here!!!
            _, predicted = torch.max(output, dim=1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

        validation_loss.append(running_loss)
        val_acc = (correct / total) * 100
        validation_acc.append(val_acc)

        # Early Stopping Check based on accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            wait = 0
        else:
            wait += 1

        if wait > patience:
            print(f"Early stopping at epoch: {epoch+1}")
            break

        print(f"\tTrain loss: {train_loss[-1]:.3f},"
              f" Validation loss: {validation_loss[-1]:.3f},"
              f" Validation accuracy: {val_acc:.3f}")

    end_time = time.time()
    print(f"Time for training: {(end_time - start_time) / 60.0} mins.")
    print(f"Best validation accuracy: {best_acc} at epoch {best_epoch}")

    return best_model, train_loss, validation_loss, train_acc, validation_acc
