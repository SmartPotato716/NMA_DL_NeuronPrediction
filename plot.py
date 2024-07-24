import matplotlib.pyplot as plt
import torch


def plot_loss_accuracy(
        train_loss,
        train_acc,
        validation_loss,
        validation_acc,
        hyperparameters
):
    """
    Code to plot loss and accuracy with best validation accuracy
    and hyperparameters.

    Args:
      train_loss: list
        Log of training loss
      validation_loss: list
        Log of validation loss
      train_acc: list
        Log of training accuracy
      validation_acc: list
        Log of validation accuracy
      hyperparameters: dict
        Model hyperparameters to display

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
    best_val_acc_epoch = validation_acc.index(max(validation_acc))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5.5))

    ax1.plot(list(range(epochs)), train_loss, label='Training Loss')
    ax1.plot(list(range(epochs)), validation_loss, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Epoch vs Loss')
    ax1.legend()

    ax2.plot(list(range(epochs)), train_acc, label='Training Accuracy')
    ax2.plot(list(range(epochs)), validation_acc, label='Validation Accuracy')
    ax2.scatter(
        best_val_acc_epoch,
        validation_acc[best_val_acc_epoch],
        color='red',
        label=f'Best Val Acc ({validation_acc[best_val_acc_epoch]:.2f})')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Epoch vs Accuracy')
    ax2.legend()

    # Hyperparameters plot
    ax3.axis('off')
    hyperparams_text = '\n'.join([f'{key}: {value}'
                                  for key, value in hyperparameters.items()])
    ax3.text(
        0.5, 0.5,
        hyperparams_text,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=12)

