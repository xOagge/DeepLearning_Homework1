#!/usr/bin/env python

import argparse
import time
import pickle
import json
import os
from itertools import product  # For generating all combinations of hyperparameters

import numpy as np

# Import utility functions (assumed to be in same directory)
import utils


# =============================================================================
# FEATURE EXTRACTION FUNCTIONS
# =============================================================================

def extract_downsampled_features(X, original_size=28, pool_size=2):
    n_samples = X.shape[0]
    new_size = original_size // pool_size  # 28 // 2 = 14
    
    # Reshape all images at once: (n_samples, 784) → (n_samples, 28, 28)
    imgs = X.reshape(n_samples, original_size, original_size)
    
    # Reshape to separate pooling regions:
    # (n_samples, 28, 28) → (n_samples, 14, 2, 14, 2)
    # This groups pixels into 2×2 blocks that we'll average
    imgs = imgs.reshape(n_samples, new_size, pool_size, new_size, pool_size)
    
    # Average over the pool_size dimensions (axes 2 and 4)
    # This computes the mean of each 2×2 block
    # Result: (n_samples, 14, 14)
    pooled = imgs.mean(axis=(2, 4))
    
    # Flatten back to 2D: (n_samples, 196)
    features = pooled.reshape(n_samples, -1)
    
    return features


# =============================================================================
# LOGISTIC REGRESSION MODEL
# =============================================================================

class LogisticRegression:
   
    
    def __init__(self, n_classes, n_features, learning_rate, l2_penalty):
        # Initialize weights to zeros
        # Shape: (n_classes, n_features) - each row is weights for one class
        # Zero initialization is fine for convex problems like logistic regression
        self.W = np.zeros((n_classes, n_features), dtype=np.float64)
        
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty

    def save(self, path):
        """Save model weights to disk using pickle serialization."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """Load a saved model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _softmax(scores):
        # Shift scores so the maximum is 0 (prevents overflow)
        shifted = scores - np.max(scores)
        
        # Compute exponentials (all values now <= 1)
        exp_scores = np.exp(shifted)
        
        # Normalize to get probabilities
        probs = exp_scores / np.sum(exp_scores)
        
        return probs

    def sgd_step(self, x_i, y_i):
        # =====================================================================
        # FORWARD PASS: Compute predictions
        # =====================================================================
        
        # Compute raw scores for each class
        # scores[k] = W[k] · x_i (dot product of k-th weight row with input)
        scores = self.W.dot(x_i)  # Shape: (n_classes,)
        
        # Convert scores to probabilities
        probs = self._softmax(scores)  # Shape: (n_classes,)
        
        # =====================================================================
        # COMPUTE LOSS
        # =====================================================================
        
        # Negative log-likelihood: how "surprised" we are by the true label
        # Small epsilon (1e-15) prevents log(0) which would be -infinity
        nll = -np.log(probs[y_i] + 1e-15)
        
        # L2 regularization: penalty for large weights
        l2_term = self.l2_penalty * np.sum(self.W ** 2)
        
        # Total loss
        loss = nll + l2_term
        
        # =====================================================================
        # BACKWARD PASS: Compute gradients
        # =====================================================================
        
        # Gradient of cross-entropy with respect to scores
        # This is the famous "softmax gradient": p - one_hot(y)
        grad_scores = probs.copy()
        grad_scores[y_i] -= 1.0  # Subtract 1 at the true class position
        
        # Gradient with respect to weights (chain rule)
        # Since scores = W @ x, gradient is outer product of grad_scores and x
        # grad_W[k,j] = grad_scores[k] * x_i[j]
        grad_W = np.outer(grad_scores, x_i)  # Shape: (n_classes, n_features)
        
        # Add gradient from L2 regularization: d/dW(λ||W||²) = 2λW
        if self.l2_penalty > 0.0:
            grad_W += 2.0 * self.l2_penalty * self.W
        
        # =====================================================================
        # UPDATE WEIGHTS
        # =====================================================================
        
        # SGD update: move weights in negative gradient direction
        # W_new = W_old - learning_rate * gradient
        self.W -= self.learning_rate * grad_W
        
        return float(loss)

    def train_epoch(self, X, y):
        n_examples = X.shape[0]
        losses = []
        
        # Process each example one at a time (batch size = 1)
        for i in range(n_examples):
            loss = self.sgd_step(X[i, :], y[i])
            losses.append(loss)
        
        return float(np.mean(losses))

    def predict(self, X):
        # Compute scores for all examples at once (matrix multiplication)
        # X @ W.T gives (n_examples, n_classes) matrix of scores
        scores = X.dot(self.W.T)
        
        # For each example, find the class with highest score
        preds = np.argmax(scores, axis=1)
        
        return preds

    def evaluate(self, X, y):
        preds = self.predict(X)
        accuracy = np.mean(preds == y)
        return float(accuracy)


# =============================================================================
# TRAINING FUNCTION FOR A SINGLE CONFIGURATION
# =============================================================================

def train_configuration(X_train, y_train, X_valid, y_valid, 
                        learning_rate, l2_penalty, n_epochs, 
                        save_path, seed=42):
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Get dimensions
    n_classes = np.unique(y_train).size
    n_features = X_train.shape[1]
    
    # Initialize model
    model = LogisticRegression(
        n_classes=n_classes,
        n_features=n_features,
        learning_rate=learning_rate,
        l2_penalty=l2_penalty
    )
    
    # Track best validation accuracy
    best_valid_acc = 0.0
    best_epoch = -1
    
    # Make a copy of data for shuffling (don't modify original)
    X_train_copy = X_train.copy()
    y_train_copy = y_train.copy()
    
    # Training loop
    for epoch in range(1, n_epochs + 1):
        # Shuffle training data each epoch
        order = np.random.permutation(X_train_copy.shape[0])
        X_train_copy = X_train_copy[order]
        y_train_copy = y_train_copy[order]
        
        # Train for one epoch
        model.train_epoch(X_train_copy, y_train_copy)
        
        # Evaluate on validation set
        valid_acc = model.evaluate(X_valid, y_valid)
        
        # Save checkpoint if this is the best so far
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch
            model.save(save_path)
    
    # Load the best model
    best_model = LogisticRegression.load(save_path)
    
    return {
        'best_valid_acc': best_valid_acc,
        'best_epoch': best_epoch,
        'model': best_model
    }


# =============================================================================
# MAIN GRID SEARCH FUNCTION
# =============================================================================

def main(args):
    
    # Set random seed for reproducibility
    utils.configure_seed(seed=args.seed)
    
    # =========================================================================
    # STEP 1: Define the hyperparameter grid
    # =========================================================================
    
    # Learning rates to try
    # - We explore a range spanning an order of magnitude
    # - Too high: training diverges; too low: training is slow
    learning_rates = [0.0001, 0.0005, 0.001]
    
    # L2 penalties (regularization strengths) to try
    # - Higher values mean stronger regularization (simpler model)
    # - Lower values allow more complex models (risk of overfitting)
    l2_penalties = [0.00001, 0.0001]
    
    # Feature representations to try
    feature_types = ['pixels', 'downsampled']
    
    print("="*70)
    print("GRID SEARCH FOR LOGISTIC REGRESSION HYPERPARAMETERS")
    print("="*70)
    print(f"\nHyperparameter grid:")
    print(f"  Learning rates: {learning_rates}")
    print(f"  L2 penalties: {l2_penalties}")
    print(f"  Feature types: {feature_types}")
    print(f"  Total configurations: {len(learning_rates) * len(l2_penalties) * len(feature_types)}")
    print(f"  Epochs per configuration: {args.epochs}")
    
    # =========================================================================
    # STEP 2: Load and prepare data
    # =========================================================================
    
    print("\n" + "-"*70)
    print("Loading dataset...")
    print("-"*70)
    
    # Load raw data (without bias - we'll add it after feature extraction)
    data = utils.load_dataset(data_path=args.data_path, bias=False)
    
    X_train_raw, y_train = data["train"]
    X_valid_raw, y_valid = data["dev"]
    X_test_raw, y_test = data["test"]
    
    print(f"Training examples: {X_train_raw.shape[0]}")
    print(f"Validation examples: {X_valid_raw.shape[0]}")
    print(f"Test examples: {X_test_raw.shape[0]}")
    print(f"Original feature dimension: {X_train_raw.shape[1]}")
    
    # Prepare both feature representations
    print("\nPreparing feature representations...")
    
    # Feature type 1: Original pixels + bias
    X_train_pixels = np.hstack((X_train_raw, np.ones((X_train_raw.shape[0], 1))))
    X_valid_pixels = np.hstack((X_valid_raw, np.ones((X_valid_raw.shape[0], 1))))
    X_test_pixels = np.hstack((X_test_raw, np.ones((X_test_raw.shape[0], 1))))
    print(f"  Pixels: {X_train_pixels.shape[1]} features (784 pixels + 1 bias)")
    
    # Feature type 2: Downsampled + bias
    X_train_down = extract_downsampled_features(X_train_raw)
    X_valid_down = extract_downsampled_features(X_valid_raw)
    X_test_down = extract_downsampled_features(X_test_raw)
    
    X_train_down = np.hstack((X_train_down, np.ones((X_train_down.shape[0], 1))))
    X_valid_down = np.hstack((X_valid_down, np.ones((X_valid_down.shape[0], 1))))
    X_test_down = np.hstack((X_test_down, np.ones((X_test_down.shape[0], 1))))
    print(f"  Downsampled: {X_train_down.shape[1]} features (196 pooled + 1 bias)")
    
    # Store features in a dictionary for easy access
    features = {
        'pixels': {
            'train': X_train_pixels,
            'valid': X_valid_pixels,
            'test': X_test_pixels
        },
        'downsampled': {
            'train': X_train_down,
            'valid': X_valid_down,
            'test': X_test_down
        }
    }
    
    # =========================================================================
    # STEP 3: Run grid search
    # =========================================================================
    
    print("\n" + "="*70)
    print("RUNNING GRID SEARCH")
    print("="*70)
    
    # Store results for all configurations
    results = []
    
    # Track the best configuration overall
    best_overall_valid_acc = 0.0
    best_overall_config = None
    best_overall_model_path = None
    
    # Generate all combinations of hyperparameters
    # itertools.product gives us the Cartesian product
    configurations = list(product(learning_rates, l2_penalties, feature_types))
    
    total_configs = len(configurations)
    start_time = time.time()
    
    for config_idx, (lr, l2, feat_type) in enumerate(configurations, 1):
        print(f"\n--- Configuration {config_idx}/{total_configs} ---")
        print(f"  Learning rate: {lr}")
        print(f"  L2 penalty: {l2}")
        print(f"  Features: {feat_type}")
        
        # Get the appropriate feature matrices
        X_train = features[feat_type]['train']
        X_valid = features[feat_type]['valid']
        
        # Create a unique save path for this configuration's checkpoint
        checkpoint_path = f"checkpoint_lr{lr}_l2{l2}_{feat_type}.pkl"
        
        # Train the model
        config_start = time.time()
        result = train_configuration(
            X_train, y_train,
            X_valid, y_valid,
            learning_rate=lr,
            l2_penalty=l2,
            n_epochs=args.epochs,
            save_path=checkpoint_path,
            seed=args.seed
        )
        config_time = time.time() - config_start
        
        print(f"  Best validation accuracy: {result['best_valid_acc']:.4f} (epoch {result['best_epoch']})")
        print(f"  Training time: {config_time:.1f}s")
        
        # Store result
        result_entry = {
            'learning_rate': lr,
            'l2_penalty': l2,
            'feature_type': feat_type,
            'best_valid_acc': result['best_valid_acc'],
            'best_epoch': result['best_epoch'],
            'checkpoint_path': checkpoint_path
        }
        results.append(result_entry)
        
        # Check if this is the best configuration so far
        if result['best_valid_acc'] > best_overall_valid_acc:
            best_overall_valid_acc = result['best_valid_acc']
            best_overall_config = result_entry
            best_overall_model_path = checkpoint_path
    
    total_time = time.time() - start_time
    
    # =========================================================================
    # STEP 4: Report results
    # =========================================================================
    
    print("\n" + "="*70)
    print("GRID SEARCH RESULTS")
    print("="*70)
    
    # Print table header
    print(f"\n{'Config':<8} {'LR':<10} {'L2':<10} {'Features':<12} {'Valid Acc':<12}")
    print("-"*52)
    
    # Sort results by validation accuracy for easier reading
    results_sorted = sorted(results, key=lambda x: x['best_valid_acc'], reverse=True)
    
    for i, r in enumerate(results_sorted, 1):
        marker = "★" if r == best_overall_config else " "
        print(f"{marker}{i:<7} {r['learning_rate']:<10} {r['l2_penalty']:<10} "
              f"{r['feature_type']:<12} {r['best_valid_acc']:.4f}")
    
    # =========================================================================
    # STEP 5: Evaluate best configuration on test set
    # =========================================================================
    
    print("\n" + "="*70)
    print("BEST CONFIGURATION")
    print("="*70)
    print(f"  Learning rate: {best_overall_config['learning_rate']}")
    print(f"  L2 penalty: {best_overall_config['l2_penalty']}")
    print(f"  Feature type: {best_overall_config['feature_type']}")
    print(f"  Best validation accuracy: {best_overall_config['best_valid_acc']:.4f}")
    
    # Load the best model and evaluate on test set
    best_model = LogisticRegression.load(best_overall_model_path)
    
    # Get test features for the best feature type
    X_test_best = features[best_overall_config['feature_type']]['test']
    test_acc = best_model.evaluate(X_test_best, y_test)
    
    print(f"\n  *** TEST ACCURACY: {test_acc:.4f} ***")
    
    # =========================================================================
    # STEP 6: Save results to JSON
    # =========================================================================
    
    final_results = {
        'grid_search': {
            'learning_rates': learning_rates,
            'l2_penalties': l2_penalties,
            'feature_types': feature_types,
            'epochs': args.epochs
        },
        'all_configurations': results_sorted,
        'best_configuration': {
            'learning_rate': best_overall_config['learning_rate'],
            'l2_penalty': best_overall_config['l2_penalty'],
            'feature_type': best_overall_config['feature_type'],
            'best_valid_acc': best_overall_config['best_valid_acc'],
            'test_acc': test_acc
        },
        'total_time_seconds': total_time
    }
    
    with open(args.results_file, 'w') as f:
        json.dump(final_results, f, indent=4)
    
    print(f"\nResults saved to: {args.results_file}")
    print(f"Total grid search time: {total_time/60:.1f} minutes")
    
    # =========================================================================
    # STEP 7: Clean up checkpoint files (optional)
    # =========================================================================
    
    # Keep only the best checkpoint, remove others
    for r in results:
        if r['checkpoint_path'] != best_overall_model_path:
            if os.path.exists(r['checkpoint_path']):
                os.remove(r['checkpoint_path'])
    
    # Rename the best checkpoint to a more descriptive name
    if os.path.exists(best_overall_model_path):
        os.rename(best_overall_model_path, args.best_model_path)
        print(f"Best model saved to: {args.best_model_path}")
    
    print("\n" + "="*70)
    print("GRID SEARCH COMPLETE!")
    print("="*70)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid search for logistic regression hyperparameters on EMNIST Letters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python hw1-logreg-gridsearch.py --data-path emnist-letters.npz

This will run 12 configurations (3 learning rates × 2 L2 penalties × 2 feature types)
and report the best hyperparameter combination.
        """
    )
    
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="emnist-letters.npz",
        help="Path to the EMNIST Letters dataset file"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=20,
        help="Number of training epochs per configuration (default: 20)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--results-file", 
        type=str, 
        default="Q1-gridsearch-results.json",
        help="Path to save JSON results (default: Q1-gridsearch-results.json)"
    )
    parser.add_argument(
        "--best-model-path", 
        type=str, 
        default="Q1-gridsearch-best-model.pkl",
        help="Path to save the best model (default: Q1-gridsearch-best-model.pkl)"
    )
    
    args = parser.parse_args()
    main(args)