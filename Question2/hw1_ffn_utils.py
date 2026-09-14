import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import utils
from hw1_ffn import FeedforwardNetwork, train_batch, evaluate

def setup_data(path="emnist-letters.npz"):
    #use utils to load the datasets of test, train, dev
    data = utils.load_dataset(path)
    dataset = utils.ClassificationDataset(data)
    n_classes = torch.unique(dataset.y).shape[0] #number of classes in train dataset
    n_feats = dataset.X.shape[1] #number of features on each input
    return dataset, n_classes, n_feats

#define the dataloader
def get_dataloader(dataset, batch_size=64, seed=42):
    #always reset the seed before training each model so the system can be reporduced
    #later the exact same way.
    utils.configure_seed(seed)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )

#setup model and optimizer
def setup_model_and_optimizer(n_classes, n_feats, hidden_size, layers, 
                              activation_type, dropout, optimizer_name, 
                              learning_rate, l2_val):
    """
    Initializes the model, optimizer, and loss function in one go.
    """
    # create the model with the defined structure and hyperparameters
    model = FeedforwardNetwork( n_classes=n_classes, n_features=n_feats,
        hidden_size=hidden_size, layers=layers, activation_type=activation_type,
        dropout=dropout
    )

    # define optimizer. if the argument string is not one of the available ones, give error
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    if optimizer_name not in optims: raise ValueError(f"Optimizer {optimizer_name} not supported")
    optimizer = optims[optimizer_name]( model.parameters(), lr=learning_rate, weight_decay=l2_val)

    # criterion is defined as cross entropy for the whole Q2 as is in the skeleton
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion