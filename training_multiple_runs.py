""" Created on Wed Jul 24 20:50:15 2024
    @author: dcupolillo """

from download_load_dataset import alldat
from dataset import DatasetMotion
import torch
from utils import set_seed, split
from motiondecoder import decoder, train
from plot import plot_multiple_runs
from torchsummary import summary

# set seed for reproducibility
set_seed(seed=2024)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize dataset
dataset = DatasetMotion(alldat)

# initialize model and model's parameters
motion_decoder = decoder()

# merge trials across sessions
merged_trials = (
    torch.cat([session[0] for session in dataset], dim=2),
    torch.cat([session[1] for session in dataset], dim=-1))

# Split dataset into training and validation
train_loader, valid_loader = split(merged_trials, batch_size=15)
summary(motion_decoder, input_size=(1, 250, 50))

base_hyperparams = {
    'epochs': 100,
    'lr': 0.00001,
    'patience': 20,
    'augment_prob': 0.4,
    'noise_level': 0.1,
    'lambda1': 0.0001,
    'lambda2': 0.001,
}

# Define the ranges for the hyperparameters you want to vary
varying_hyperparam = 'augment_prob'
varying_hyperparam_values = [0.4, 0.5, 0.6]

# Generate the list of hyperparameter configurations
hyperparams_list = []

for parameter in varying_hyperparam_values:
    hyperparams = base_hyperparams.copy()
    hyperparams[varying_hyperparam] = parameter
    hyperparams_list.append(hyperparams)

# To store results
results = []

for params in hyperparams_list:
    args = {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'model': motion_decoder,
        'device': device,
        **params,
    }

    _, train_loss, validation_loss, train_acc, validation_acc = train(**args)

    results.append({
        'params': params,
        'train_loss': train_loss,
        'validation_loss': validation_loss,
        'train_acc': train_acc,
        'validation_acc': validation_acc,
    })

for plotting in ['loss', 'acc']:
    plot_multiple_runs(
        results,
        to_plot=plotting,
        varying_param=varying_hyperparam,
        save=True)
