#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import time
import pickle
import json

import numpy as np

import utils


class Perceptron:
    def __init__(self, n_classes, n_features):
        # Initialize weight matrix with zeros
        # Shape: (n_classes, n_features)
        # Each row contains weights for one class
        self.W = np.zeros((n_classes, n_features))

    def save(self, path):
        """
        Save perceptron to the provided path
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """
        Load perceptron from the provided path
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def update_weight(self, x_i, y_i):
        """
        Perform perceptron weight update for a single example.
        
        Algorithm:
        1. Compute scores for all classes: score_k = W[k] · x_i
        2. Predict class with highest score: y_hat = argmax(scores)
        3. If prediction is wrong:
           - Add x_i to weights of correct class (to increase its score)
           - Subtract x_i from weights of predicted class (to decrease its score)
        
        Args:
            x_i (n_features,): a single training example
            y_i (scalar): the gold label for that example
        """
        # Compute scores for all classes
        # scores[k] = sum over j of W[k,j] * x_i[j]
        scores = self.W.dot(x_i)  # Shape: (n_classes,)
        
        # Predict the class with maximum score
        y_hat = np.argmax(scores)
        
        # Update only if prediction is wrong
        if y_hat != y_i:
            # Boost the weights for the correct class
            # This increases the score for correct class on similar inputs
            self.W[y_i, :] += x_i
            
            # Reduce the weights for the incorrectly predicted class
            # This decreases the score for wrong class on similar inputs
            self.W[y_hat, :] -= x_i

    def train_epoch(self, X, y):
        """
        Train the perceptron for one complete pass through the dataset.
        
        Goes through each training example sequentially and updates
        weights using the perceptron update rule.
        
        Args:
            X (n_examples, n_features): features for the whole dataset
            y (n_examples,): labels for the whole dataset
        """
        # Iterate through all training examples
        for i in range(X.shape[0]):
            # Get the i-th example and its label
            x_i = X[i, :]
            y_i = y[i]
            
            # Perform perceptron update for this example
            self.update_weight(x_i, y_i)

    def predict(self, X):
        """
        Make predictions for all examples in X.
        
        For each example, compute scores for all classes and return
        the class with the highest score.
        
        Args:
            X (n_examples, n_features): input features
            
        Returns:
            predictions (n_examples,): predicted labels
        """
        # Compute scores for all examples and all classes
        # scores[i,k] = W[k] · X[i] = score for example i, class k
        # Matrix multiplication: (n_examples, n_features) @ (n_features, n_classes)^T
        scores = X.dot(self.W.T)  # Shape: (n_examples, n_classes)
        
        # For each example (row), find the class (column) with maximum score
        # axis=1 means we take argmax across columns (classes) for each row (example)
        predictions = np.argmax(scores, axis=1)  # Shape: (n_examples,)
        
        return predictions

    def evaluate(self, X, y):
        """
        Compute classification accuracy on the given dataset.
        
        Accuracy = (number of correct predictions) / (total predictions)
        
        Args:
            X (n_examples x n_features): input features
            y (n_examples): gold labels

        Returns:
            accuracy (float): fraction of correct predictions
        """
        # Get predictions for all examples
        predictions = self.predict(X)
        
        # Calculate accuracy: fraction of examples where prediction matches true label
        # (predictions == y) creates boolean array, mean() gives fraction of True values
        accuracy = np.mean(predictions == y)
        
        return accuracy


def main(args):
    # Set random seed for reproducibility
    utils.configure_seed(seed=args.seed)

    # Load EMNIST Letters dataset
    # bias=True adds a bias term (constant 1) as an additional feature
    data = utils.load_dataset(data_path=args.data_path, bias=True)
    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]
    
    # Get dataset dimensions
    n_classes = np.unique(y_train).size  # 26 (letters A-Z)
    n_feats = X_train.shape[1]            # 785 (784 pixels + 1 bias)

    # Initialize the perceptron model
    model = Perceptron(n_classes, n_feats)

    # Create array of epoch numbers for plotting
    epochs = np.arange(1, args.epochs + 1)

    # Lists to store accuracies for plotting
    valid_accs = []
    train_accs = []

    # Track the best validation accuracy and corresponding epoch
    best_valid = 0.0
    best_epoch = -1
    
    start = time.time()

    # Training loop: train for specified number of epochs
    for i in epochs:
        print('Training epoch {}'.format(i))
        
        # Shuffle training data each epoch
        # This prevents the model from learning patterns based on example order
        train_order = np.random.permutation(X_train.shape[0])
        X_train = X_train[train_order]
        y_train = y_train[train_order]

        # Train for one epoch (one pass through all training data)
        model.train_epoch(X_train, y_train)

        # Evaluate performance on training and validation sets
        train_acc = model.evaluate(X_train, y_train)
        valid_acc = model.evaluate(X_valid, y_valid)

        # Store accuracies for plotting
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        print('train acc: {:.4f} | val acc: {:.4f}'.format(train_acc, valid_acc))

        # Save checkpoint if this is the best validation accuracy so far
        # We use validation accuracy (not training) to select the best model
        # because it better indicates generalization to unseen data
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_epoch = i
            print('  -> New best validation accuracy! Saving checkpoint...')
            model.save(args.save_path)

    # Calculate total training time
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    # Load the best model (based on validation accuracy)
    print("Reloading best checkpoint from epoch {}".format(best_epoch))
    best_model = Perceptron.load(args.save_path)
    
    # Evaluate best model on test set
    test_acc = best_model.evaluate(X_test, y_test)
    print('Best model test acc: {:.4f}'.format(test_acc))

    # Generate plot of training and validation accuracies
    utils.plot(
        "Epoch", "Accuracy",
        {"train": (epochs, train_accs), "valid": (epochs, valid_accs)},
        filename=args.accuracy_plot
    )

    # Save numerical results to JSON file
    with open(args.scores, "w") as f:
        json.dump(
            {"best_valid": float(best_valid),
             "selected_epoch": int(best_epoch),
             "test": float(test_acc),
             "time": elapsed_time},
            f,
            indent=4
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int,
                        help="""Number of epochs to train for.""")
    parser.add_argument('--data-path', type=str, default="emnist-letters.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--accuracy-plot", default="Q1-perceptron-accs.pdf")
    parser.add_argument("--scores", default="Q1-perceptron-scores.json")
    args = parser.parse_args()
    main(args)