""" Created on Fri Jul 19 17:29:18 2024
    @author: dcupolillo """

import numpy as np
import random
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from dataset import Session


def split(
        session,
        validation_split=0.2,
        shuffle_trials=True,
        batch_size=1
):
    num_trials = session[0].shape[2]
    indices = list(range(num_trials))
    split_index = int(np.floor(validation_split * num_trials))
    if shuffle_trials:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split_index:], indices[:split_index]

    # Creating PT data samplers and loaders
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    session_dataset = Session(session)

    train_loader = DataLoader(
        session_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(
        session_dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader


def set_seed(
        seed=None,
        seed_torch=True
):
    """
    Function that controls randomness.
    NumPy and random modules must be imported.

    Args:
        seed : Integer
        A non-negative integer that defines the random state. Default is `None`
    seed_torch : Boolean
        If `True` sets the random seed for pytorch tensors, so pytorch module
        must be imported. Default is `True`.

    Returns:
        Nothing.
    """
    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)

    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f'Random seed {seed} has been set.')
