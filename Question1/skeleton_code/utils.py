import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch


def configure_seed(seed):
    """
    Set random seeds for reproducibility across all libraries.
    
    This ensures that:
    - Python's random module produces same sequence
    - NumPy's random functions produce same sequence  
    - PyTorch operations are deterministic
    
    Args:
        seed (int): random seed value
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dataset(data_path, bias=False):
    """
    Load EMNIST Letters data from compressed npz file.
    
    The dataset contains:
    - Training set: for learning model parameters
    - Validation set: for hyperparameter tuning and model selection
    - Test set: for final evaluation (touch only once!)
    
    Args:
        data_path (str): path to the .npz file
        bias (bool): if True, append a constant 1 feature to each example
        
    Returns:
        dict: dictionary with keys "train", "dev", "test"
              each containing (X, y) tuple
    """
    # Load compressed numpy archive
    data_dict = np.load(data_path)

    # Extract and normalize data
    # Original data is uint8 (0-255), we normalize to [0, 1]
    X_train = data_dict["X_train"].astype(float) / 256
    X_valid = data_dict["X_valid"].astype(float) / 256
    X_test = data_dict["X_test"].astype(float) / 256

    # Extract labels and ensure they are integers
    y_train = data_dict["y_train"].astype(int)
    y_valid = data_dict["y_valid"].astype(int)
    y_test = data_dict["y_test"].astype(int)

    # Some datasets use 1-based indexing (labels 1-26)
    # We need 0-based indexing (labels 0-25) for array indexing
    if np.min(y_train) == 1:
        y_train -= 1
        y_valid -= 1
        y_test -= 1

    # Add bias feature if requested
    # This is equivalent to adding a bias term in the linear model
    # Instead of y = Wx, we compute y = Wx + b
    # By appending 1 to x, we can write it as y = W[x, 1]
    if bias:
        X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
        X_valid = np.hstack((X_valid, np.ones((X_valid.shape[0], 1))))
        X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

    return {
        "train": (X_train, y_train),
        "dev": (X_valid, y_valid),
        "test": (X_test, y_test),
    }


def plot(x_label, y_label, curves, filename=None):
    """
    Create a line plot with multiple curves.
    
    Args:
        x_label (str): label for x-axis
        y_label (str): label for y-axis
        curves (dict): mapping from curve label to (x_values, y_values) tuple
        filename (str): if provided, save plot to this file
    """
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Plot each curve
    for curve_label, (x, y) in curves.items():
        plt.plot(x, y, label=curve_label)

    # Add legend to distinguish curves
    plt.legend()
    
    # Save to file if path provided
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    
    # Clear figure for next plot
    plt.clf()


class ClassificationDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for classification data.
    
    This is used in Question 2 with PyTorch DataLoader for batching.
    """

    def __init__(self, data):
        """
        Args:
            data: the dict returned by load_dataset
        """
        train_X, train_y = data["train"]
        dev_X, dev_y = data["dev"]
        test_X, test_y = data["test"]

        # Convert numpy arrays to PyTorch tensors
        # float32 for features (continuous values)
        # long (int64) for labels (class indices)
        self.X = torch.tensor(train_X, dtype=torch.float32)
        self.y = torch.tensor(train_y, dtype=torch.long)

        self.train_X = self.X
        self.train_y = self.y

        self.dev_X = torch.tensor(dev_X, dtype=torch.float32)
        self.dev_y = torch.tensor(dev_y, dtype=torch.long)

        self.test_X = torch.tensor(test_X, dtype=torch.float32)
        self.test_y = torch.tensor(test_y, dtype=torch.long)

    def __len__(self):
        """Return number of training examples."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get a single training example.
        
        Args:
            idx (int): index of example
            
        Returns:
            tuple: (features, label)
        """
        return self.X[idx], self.y[idx]