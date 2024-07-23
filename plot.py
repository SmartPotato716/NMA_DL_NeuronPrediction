""" Created on Mon Jul 22 17:22:50 2024
    @author: dcupolillo """

import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_loss_accuracy(
        train_loss,
        train_acc,
        validation_loss,
        validation_acc
):
    """
    Code to plot loss and accuracy

    Args:
      train_loss: list
        Log of training loss
      validation_loss: list
        Log of validation loss
      train_acc: list
        Log of training accuracy
      validation_acc: list
        Log of validation accuracy

    Returns:
      Nothing
    """

    # Ensure inputs are lists
    if isinstance(train_loss, torch.Tensor):
        train_loss = train_loss.cpu().tolist()
    if isinstance(validation_loss, torch.Tensor):
        validation_loss = validation_loss.cpu().tolist()
    if isinstance(train_acc, torch.Tensor):
        train_acc = train_acc.cpu().tolist()
    if isinstance(validation_acc, torch.Tensor):
        validation_acc = validation_acc.cpu().tolist()

    epochs = len(train_loss)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.5))

    ax1.plot(list(range(epochs)), train_loss, label='Training Loss')
    ax1.plot(list(range(epochs)), validation_loss, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Epoch vs Loss')
    ax1.legend()

    ax2.plot(list(range(epochs)), train_acc, label='Training Accuracy')
    ax2.plot(list(range(epochs)), validation_acc, label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Epoch vs Accuracy')
    ax2.legend()

    plt.show()
