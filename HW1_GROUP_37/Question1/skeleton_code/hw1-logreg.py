#!/usr/bin/env python

# Deep Learning Homework 1
# Logistic Regression with SGD and l2 regularization

import argparse
import time
import pickle
import json

import numpy as np

import utils


class LogisticRegression:
    def __init__(self, n_classes, n_features, learning_rate, l2_penalty):
        """
        Multiclass logistic regression with softmax output.

        Args:
            n_classes (int): number of classes
            n_features (int): number of input features
            learning_rate (float): SGD learning rate
            l2_penalty (float): l2 regularization strength (lambda)
        """
        # Initialize weights.
        # Shape: (n_classes, n_features)
        # You can use zeros or small random values.
        self.W = np.zeros((n_classes, n_features), dtype=np.float64)

        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty

    def save(self, path):
        """Save model to the provided path."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """Load model from the provided path."""
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _softmax(scores):
        """
        Compute softmax probabilities in a numerically stable way.

        Args:
            scores (np.ndarray): shape (n_classes,)

        Returns:
            probs (np.ndarray): shape (n_classes,)
        """
        # Shift for numerical stability
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores)
        return probs

    def sgd_step(self, x_i, y_i):
        """
        Perform one SGD update on a single example.

        Args:
            x_i (np.ndarray): shape (n_features,)
            y_i (int): gold label

        Returns:
            loss (float): loss value for monitoring
        """
        # Compute scores: shape (n_classes,)
        scores = self.W.dot(x_i)

        # Softmax probabilities
        probs = self._softmax(scores)

        # Negative log likelihood loss
        # Add l2 regularization
        nll = -np.log(probs[y_i] + 1e-15)  # small epsilon for safety
        l2_term = self.l2_penalty * np.sum(self.W ** 2)
        loss = nll + l2_term

        # Gradient of NLL w.r.t. scores is (p - one_hot(y))
        grad_scores = probs.copy()
        grad_scores[y_i] -= 1.0  # shape (n_classes,)

        # Gradient w.r.t. W from data term
        # outer product: (n_classes,) x (n_features,) -> (n_classes, n_features)
        grad_W = np.outer(grad_scores, x_i)

        # Add gradient from l2 regularization
        if self.l2_penalty > 0.0:
            grad_W += 2.0 * self.l2_penalty * self.W

        # SGD update
        self.W -= self.learning_rate * grad_W

        return float(loss)

    def train_epoch(self, X, y):
        """
        Train for one epoch over the dataset using SGD with batch size 1.

        Args:
            X (np.ndarray): shape (n_examples, n_features)
            y (np.ndarray): shape (n_examples,)

        Returns:
            avg_loss (float): average loss over the epoch
        """
        n_examples = X.shape[0]
        losses = []

        for i in range(n_examples):
            x_i = X[i, :]
            y_i = y[i]
            loss = self.sgd_step(x_i, y_i)
            losses.append(loss)

        return float(np.mean(losses))

    def predict(self, X):
        """
        Predict class labels for all examples in X.

        Args:
            X (np.ndarray): shape (n_examples, n_features)

        Returns:
            preds (np.ndarray): shape (n_examples,)
        """
        # scores: (n_examples, n_classes)
        scores = X.dot(self.W.T)
        preds = np.argmax(scores, axis=1)
        return preds

    def evaluate(self, X, y):
        """
        Compute accuracy on given dataset.

        Args:
            X (np.ndarray): shape (n_examples, n_features)
            y (np.ndarray): shape (n_examples,)

        Returns:
            accuracy (float)
        """
        preds = self.predict(X)
        accuracy = np.mean(preds == y)
        return float(accuracy)


def main(args):
    # Set random seed for reproducibility
    utils.configure_seed(seed=args.seed)

    # Load dataset
    # You can choose bias=True if you want an explicit bias term in the features
    data = utils.load_dataset(data_path=args.data_path, bias=True)

    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]

    n_classes = np.unique(y_train).size
    n_feats = X_train.shape[1]

    print(f"N classes: {n_classes}")
    print(f"N features: {n_feats}")

    # Initialize model
    model = LogisticRegression(
        n_classes=n_classes,
        n_features=n_feats,
        learning_rate=args.learning_rate,
        l2_penalty=args.l2_penalty,
    )

    epochs = np.arange(1, args.epochs + 1)
    train_accs = []
    valid_accs = []
    train_losses = []

    best_valid = 0.0
    best_epoch = -1

    start = time.time()

    for epoch in epochs:
        print(f"Training epoch {epoch}")

        # Shuffle training data each epoch
        order = np.random.permutation(X_train.shape[0])
        X_train = X_train[order]
        y_train = y_train[order]

        # One full pass with SGD
        avg_loss = model.train_epoch(X_train, y_train)

        # Evaluate on train and valid
        train_acc = model.evaluate(X_train, y_train)
        valid_acc = model.evaluate(X_valid, y_valid)

        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        print(
            f"epoch {epoch:02d} | "
            f"avg loss: {avg_loss:.4f} | "
            f"train acc: {train_acc:.4f} | "
            f"val acc: {valid_acc:.4f}"
        )

        # Save best checkpoint based on validation accuracy
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_epoch = epoch
            print("  -> New best validation accuracy. Saving checkpoint.")
            model.save(args.save_path)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Training took {minutes} minutes and {seconds} seconds.")

    # Load best model and evaluate on test set
    print(f"Reloading best checkpoint from epoch {best_epoch}")
    best_model = LogisticRegression.load(args.save_path)
    test_acc = best_model.evaluate(X_test, y_test)
    print(f"Best model test acc: {test_acc:.4f}")

    # Plot train and validation accuracies
    utils.plot(
        "Epoch",
        "Accuracy",
        {"train": (epochs, train_accs), "valid": (epochs, valid_accs)},
        filename=args.accuracy_plot,
    )

    # If you also want a loss plot:
    utils.plot(
        "Epoch",
        "Loss",
        {"train_loss": (epochs, train_losses)},
        filename=args.loss_plot,
    )

    # Save scores to JSON
    with open(args.scores, "w") as f:
        json.dump(
            {
                "best_valid": float(best_valid),
                "selected_epoch": int(best_epoch),
                "test": float(test_acc),
                "time": elapsed_time,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=20, type=int,
                        help="Number of epochs to train for.")
    parser.add_argument("--data-path", type=str, default="emnist-letters.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--l2-penalty", type=float, default=0.00001)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--accuracy-plot", default="Q1-logreg-accs.pdf")
    parser.add_argument("--loss-plot", default="Q1-logreg-loss.pdf")
    parser.add_argument("--scores", default="Q1-logreg-scores.json")
    args = parser.parse_args()
    main(args)
