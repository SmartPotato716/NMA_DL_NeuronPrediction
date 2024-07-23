from download_load_dataset import alldat
from dataset import DatasetMotion
import torch
from utils import set_seed, split
from motiondecoder import decoder, train
from plot import plot_loss_accuracy

# set seed for reproducibility
set_seed(seed=2024)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize dataset
dataset = DatasetMotion(alldat)

# initialize model and model's parameters
Motion_decoder = decoder()

# merge trials across sessions
merged_trials = (
    torch.cat([session[0] for session in dataset], dim=2),
    torch.cat([session[1] for session in dataset], dim=-1))

# Split dataset into training and validation
train_loader, valid_loader = split(merged_trials)

# Actual training of the motion decoder
args = {
    'train_loader': train_loader,
    'valid_loader': valid_loader,
    'model': Motion_decoder,
    'device': device,
    'epochs': 50,
    'lr':  0.0005,
    'patience': 20,
    'augment_prob': 0.5,
    'noise_level': 0.1,
    'lambda1': 0.0,
    'lambda2': 0.01,
}

_, train_loss, validation_loss, train_acc, validation_acc = train(**args)

plot_loss_accuracy(
    train_loss,
    train_acc,
    validation_loss,
    validation_acc)
