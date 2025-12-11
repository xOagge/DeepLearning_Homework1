#!/usr/bin/env python

import argparse
import time
import pickle
import json

import numpy as np

import utils


# =============================================================================
# FEATURE EXTRACTION: DOWNSAMPLING VIA AVERAGE POOLING
# =============================================================================

def extract_downsampled_features(X, original_size=28, pool_size=2):
    n_samples = X.shape[0]
    new_size = original_size // pool_size  # 28 // 2 = 14
    n_features = new_size * new_size       # 14 * 14 = 196
    
    # Pre-allocate output array
    features = np.zeros((n_samples, n_features))
    
    for i in range(n_samples):
        # Reshape flattened image to 2D grid
        img = X[i].reshape(original_size, original_size)
        
        # Apply average pooling
        # We'll build the pooled image by iterating over output positions
        pooled = np.zeros((new_size, new_size))
        
        for py in range(new_size):
            for px in range(new_size):
                # Define the pooling region in the original image
                y_start = py * pool_size
                y_end = y_start + pool_size
                x_start = px * pool_size
                x_end = x_start + pool_size
                
                # Average all pixels in this region
                pooled[py, px] = np.mean(img[y_start:y_end, x_start:x_end])
        
        # Flatten the pooled image back to 1D
        features[i] = pooled.flatten()
    
    return features


def extract_downsampled_features_fast(X, original_size=28, pool_size=2):
    n_samples = X.shape[0]
    new_size = original_size // pool_size  # 14
    
    # Reshape to (n_samples, 28, 28)
    imgs = X.reshape(n_samples, original_size, original_size)
    
    # Reshape to (n_samples, 14, 2, 14, 2) 
    # This groups pixels into 2x2 blocks
    imgs = imgs.reshape(n_samples, new_size, pool_size, new_size, pool_size)
    
    # Average over the pool_size dimensions (axes 2 and 4)
    # Result shape: (n_samples, 14, 14)
    pooled = imgs.mean(axis=(2, 4))
    
    # Flatten back to (n_samples, 196)
    features = pooled.reshape(n_samples, -1)
    
    return features


# =============================================================================
# LOGISTIC REGRESSION MODEL
# =============================================================================

class LogisticRegression:
    
    def __init__(self, n_classes, n_features, learning_rate, l2_penalty):

        # Initialize weight matrix with zeros
        # Shape: (n_classes, n_features) - each row is weights for one class
        self.W = np.zeros((n_classes, n_features), dtype=np.float64)

        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty

    def save(self, path):
        """Save model to disk using pickle serialization."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """Load a saved model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _softmax(scores):
        # Subtract max for numerical stability (prevents exp overflow)
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores)
        return probs

    def sgd_step(self, x_i, y_i):
        # -----------------------------------------------------------------
        # Forward pass: compute predictions
        # -----------------------------------------------------------------
        scores = self.W.dot(x_i)      # Shape: (n_classes,)
        probs = self._softmax(scores)  # Shape: (n_classes,)

        # -----------------------------------------------------------------
        # Compute loss
        # -----------------------------------------------------------------
        nll = -np.log(probs[y_i] + 1e-15)  # Negative log-likelihood
        l2_term = self.l2_penalty * np.sum(self.W ** 2)  # Regularization
        loss = nll + l2_term

        # -----------------------------------------------------------------
        # Backward pass: compute gradients
        # -----------------------------------------------------------------
        # Gradient of softmax + cross-entropy is beautifully simple
        grad_scores = probs.copy()
        grad_scores[y_i] -= 1.0  # p - one_hot(y)

        # Gradient w.r.t weights: outer product
        grad_W = np.outer(grad_scores, x_i)

        # Add L2 regularization gradient
        if self.l2_penalty > 0.0:
            grad_W += 2.0 * self.l2_penalty * self.W

        # -----------------------------------------------------------------
        # Update weights
        # -----------------------------------------------------------------
        self.W -= self.learning_rate * grad_W

        return float(loss)

    def train_epoch(self, X, y):
        n_examples = X.shape[0]
        losses = []

        for i in range(n_examples):
            loss = self.sgd_step(X[i, :], y[i])
            losses.append(loss)

        return float(np.mean(losses))

    def predict(self, X):
        scores = X.dot(self.W.T)  # Shape: (n_examples, n_classes)
        preds = np.argmax(scores, axis=1)
        return preds

    def evaluate(self, X, y):
        preds = self.predict(X)
        accuracy = np.mean(preds == y)
        return float(accuracy)


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def main(args):
    # Set random seed for reproducibility
    utils.configure_seed(seed=args.seed)

    # =========================================================================
    # Load and preprocess data
    # =========================================================================
    print("Loading dataset...")
    data = utils.load_dataset(data_path=args.data_path, bias=False)

    X_train_raw, y_train = data["train"]
    X_valid_raw, y_valid = data["dev"]
    X_test_raw, y_test = data["test"]

    print(f"Original feature dimension: {X_train_raw.shape[1]} (28×28 pixels)")

    # =========================================================================
    # Extract downsampled features
    # =========================================================================
    print("\nExtracting downsampled features...")
    print(f"  - Pool size: {args.pool_size}×{args.pool_size} average pooling")
    new_size = 28 // args.pool_size
    print(f"  - Output size: {new_size}×{new_size} = {new_size**2} features")
    
    # Use the fast vectorized implementation
    X_train_down = extract_downsampled_features_fast(X_train_raw, pool_size=args.pool_size)
    X_valid_down = extract_downsampled_features_fast(X_valid_raw, pool_size=args.pool_size)
    X_test_down = extract_downsampled_features_fast(X_test_raw, pool_size=args.pool_size)

    reduction_pct = 100 * X_train_down.shape[1] / X_train_raw.shape[1]
    print(f"  - Dimensionality reduction: {X_train_raw.shape[1]} → {X_train_down.shape[1]} "
          f"({reduction_pct:.1f}% of original)")

    # =========================================================================
    # Add bias term
    # =========================================================================
    X_train = np.hstack((X_train_down, np.ones((X_train_down.shape[0], 1))))
    X_valid = np.hstack((X_valid_down, np.ones((X_valid_down.shape[0], 1))))
    X_test = np.hstack((X_test_down, np.ones((X_test_down.shape[0], 1))))

    n_classes = np.unique(y_train).size
    n_feats = X_train.shape[1]

    print(f"\nNumber of classes: {n_classes}")
    print(f"Number of features (downsampled + bias): {n_feats}")

    # =========================================================================
    # Initialize model
    # =========================================================================
    model = LogisticRegression(
        n_classes=n_classes,
        n_features=n_feats,
        learning_rate=args.learning_rate,
        l2_penalty=args.l2_penalty,
    )

    print(f"\nHyperparameters:")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - L2 penalty: {args.l2_penalty}")
    print(f"  - Epochs: {args.epochs}")

    # =========================================================================
    # Training loop
    # =========================================================================
    epochs = np.arange(1, args.epochs + 1)
    train_accs = []
    valid_accs = []
    train_losses = []

    best_valid = 0.0
    best_epoch = -1

    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    start = time.time()

    for epoch in epochs:
        # Shuffle training data each epoch
        order = np.random.permutation(X_train.shape[0])
        X_train = X_train[order]
        y_train = y_train[order]

        # Train for one epoch
        avg_loss = model.train_epoch(X_train, y_train)

        # Evaluate
        train_acc = model.evaluate(X_train, y_train)
        valid_acc = model.evaluate(X_valid, y_valid)

        # Store metrics
        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {valid_acc:.4f}"
        )

        # Save best checkpoint
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_epoch = epoch
            print("  ↳ New best validation accuracy! Saving checkpoint...")
            model.save(args.save_path)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print("="*60)
    print(f"Training completed in {minutes}m {seconds}s")
    print("="*60)

    # =========================================================================
    # Evaluate best model on test set
    # =========================================================================
    print(f"\nLoading best checkpoint from epoch {best_epoch}...")
    best_model = LogisticRegression.load(args.save_path)
    test_acc = best_model.evaluate(X_test, y_test)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS (Downsampled Features)")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_valid:.4f} (epoch {best_epoch})")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"{'='*60}")

    # =========================================================================
    # Generate plots
    # =========================================================================
    utils.plot(
        "Epoch",
        "Accuracy",
        {"train": (epochs, train_accs), "valid": (epochs, valid_accs)},
        filename=args.accuracy_plot,
    )
    print(f"\nAccuracy plot saved to: {args.accuracy_plot}")

    utils.plot(
        "Epoch",
        "Loss",
        {"train_loss": (epochs, train_losses)},
        filename=args.loss_plot,
    )
    print(f"Loss plot saved to: {args.loss_plot}")

    # =========================================================================
    # Save results to JSON
    # =========================================================================
    results = {
        "feature_type": "Downsampled",
        "feature_params": {
            "pool_size": args.pool_size,
            "original_size": 28,
            "new_size": new_size,
            "n_features": int(n_feats - 1),  # Exclude bias
        },
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "l2_penalty": args.l2_penalty,
            "epochs": args.epochs,
        },
        "best_valid": float(best_valid),
        "selected_epoch": int(best_epoch),
        "test": float(test_acc),
        "time": elapsed_time,
    }
    
    with open(args.scores, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {args.scores}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Logistic Regression with Downsampled features for EMNIST Letters"
    )
    parser.add_argument(
        "--epochs", default=20, type=int,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--data-path", type=str, default="emnist-letters.npz",
        help="Path to EMNIST dataset file"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.0001,
        help="SGD learning rate (default: 0.0001)"
    )
    parser.add_argument(
        "--l2-penalty", type=float, default=0.00001,
        help="L2 regularization strength (default: 0.00001)"
    )
    parser.add_argument(
        "--pool-size", type=int, default=2,
        help="Size of average pooling window (default: 2 for 2x2 pooling)"
    )
    parser.add_argument(
        "--save-path", required=True,
        help="Path to save model checkpoint"
    )
    parser.add_argument(
        "--accuracy-plot", default="Q1-logreg-downsample-accs.pdf",
        help="Filename for accuracy plot"
    )
    parser.add_argument(
        "--loss-plot", default="Q1-logreg-downsample-loss.pdf",
        help="Filename for loss plot"
    )
    parser.add_argument(
        "--scores", default="Q1-logreg-downsample-scores.json",
        help="Filename for JSON results"
    )
    args = parser.parse_args()
    main(args)