import matplotlib.pyplot as plt
import torch
from motiondecoder import augment_trace


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
                                  for key, value in hyperparameters.items()
                                  if key not in [
                                          'train_loader',
                                          'valid_loader',
                                          'model',
                                          'device',
                                          'epochs']])
    ax3.text(
        0.5, 0.5,
        hyperparams_text,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=12)


def augmentation_example_trace(
        trace,
        save: bool = False
):

    augmented_trace = augment_trace(trace, noise_level=0.1)

    # Plot the original and augmented trace
    fig, ax = plt.subplots(2, 1, figsize=(4, 3), sharex=True, sharey=True)

    ax[0].plot(trace.numpy())
    ax[1].plot(augmented_trace.numpy())

    ax[1].set_xlabel('Time Bins')
    ax[0].set_ylabel('Firing Rate (Hz)')

    plt.tight_layout()

    if save:
        fig.savefig("figures/augmentation_example_trace.png")
        print("Figure saved!")


def plot_multiple_runs(
        results,
        to_plot='acc',
        varying_param='lr',
        save=False
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), sharex=True)

    for result in results:
        params = result['params']
        label = f"{params[varying_param]}"

        if to_plot == 'acc':
            ax1.plot(result['train_acc'], label=f"{label}")
            ax2.plot(result['validation_acc'], label=f"{label}")
            ax1.set_ylabel('Accuracy')
            ax1.set_title(f'Train. across {varying_param}')
            ax2.set_title(f'Valid. across {varying_param}')
            ax2.set_ylim(0, 100)
        elif to_plot == 'loss':
            ax1.plot(result['train_loss'], label=f"{label}")
            ax2.plot(result['validation_loss'], label=f"{label}")
            ax1.set_ylabel('Loss')
            ax1.set_title(f'Train. across {varying_param}')
            ax2.set_title(f'Valid. across {varying_param}')
        else:
            raise ValueError("Invalid value for to_plot. Use 'acc' or 'loss'.")

    ax1.set_xlabel('Epochs')
    ax2.set_xlabel('Epochs')

    # Creating a combined legend for both train and validation plots
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels)

    plt.tight_layout()

    if save:
        # Generate filename based on to_plot and varying_param
        filename = f"figures/{to_plot}_across_{varying_param}.png"
        plt.savefig(filename)
        print(f"Figure saved as {filename}")

