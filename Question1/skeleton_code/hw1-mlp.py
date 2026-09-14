#!/usr/bin/env python

import argparse
import time
import pickle
import json

import numpy as np

import utils


class MLP:
    """
    Multi-Layer Perceptron with one hidden layer.
    
    Architecture:
        Input (784) -> Hidden (100, ReLU) -> Output (26, Softmax)
    """
    
    def __init__(self, n_features, n_hidden, n_classes, learning_rate):
        # Store dimensions for later use
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        
        # ---------------------------------------------------------------------
        # Initialize weights using Normal distribution
        # As specified: μ = 0.1, σ² = 0.1² = 0.01, so σ = 0.1
        # ---------------------------------------------------------------------
        
        # W1: Input-to-hidden weights
        # Shape: (n_features, n_hidden) = (784, 100)
        # Each column represents weights connecting all inputs to one hidden unit
        self.W1 = np.random.normal(
            loc=0.1,      # Mean (μ)
            scale=0.1,    # Standard deviation (σ = sqrt(σ²) = sqrt(0.01) = 0.1)
            size=(n_features, n_hidden)
        )
        
        # b1: Hidden layer bias
        # Shape: (n_hidden,) = (100,)
        # One bias term per hidden unit
        self.b1 = np.zeros(n_hidden)
        
        # W2: Hidden-to-output weights
        # Shape: (n_hidden, n_classes) = (100, 26)
        # Each column represents weights connecting all hidden units to one output
        self.W2 = np.random.normal(
            loc=0.1,
            scale=0.1,
            size=(n_hidden, n_classes)
        )
        
        # b2: Output layer bias
        # Shape: (n_classes,) = (26,)
        # One bias term per output class
        self.b2 = np.zeros(n_classes)
        
        # ---------------------------------------------------------------------
        # Variables to store intermediate values for backpropagation
        # These are computed during forward pass and used during backward pass
        # ---------------------------------------------------------------------
        self.x = None          # Input
        self.z1 = None         # Pre-activation of hidden layer (before ReLU)
        self.h = None          # Hidden layer activations (after ReLU)
        self.z2 = None         # Pre-activation of output layer (before softmax)
        self.y_pred = None     # Output probabilities (after softmax)

    def save(self, path):
        """Save model to disk using pickle serialization."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """Load a saved model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)

    # =========================================================================
    # ACTIVATION FUNCTIONS
    # =========================================================================
    
    @staticmethod
    def relu(z):
        return np.maximum(0, z)
    
    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)
    
    @staticmethod
    def softmax(z):
        # Subtract max for numerical stability
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z)

    # =========================================================================
    # FORWARD PROPAGATION
    # =========================================================================
    
    def forward(self, x):
        # Store input for backpropagation
        self.x = x
        
        # ---------------------------------------------------------------------
        # Layer 1: Input → Hidden
        # ---------------------------------------------------------------------
        
        # Linear transformation: z1 = W1ᵀ @ x + b1
        # W1 shape: (n_features, n_hidden) = (784, 100)
        # x shape: (n_features,) = (784,)
        # z1 shape: (n_hidden,) = (100,)
        self.z1 = np.dot(x, self.W1) + self.b1
        
        # ReLU activation: h = max(0, z1)
        # h shape: (n_hidden,) = (100,)
        self.h = self.relu(self.z1)
        
        # ---------------------------------------------------------------------
        # Layer 2: Hidden → Output
        # ---------------------------------------------------------------------
        
        # Linear transformation: z2 = W2ᵀ @ h + b2
        # W2 shape: (n_hidden, n_classes) = (100, 26)
        # h shape: (n_hidden,) = (100,)
        # z2 shape: (n_classes,) = (26,)
        self.z2 = np.dot(self.h, self.W2) + self.b2
        
        # Softmax activation: convert to probabilities
        # y_pred shape: (n_classes,) = (26,)
        self.y_pred = self.softmax(self.z2)
        
        return self.y_pred

    # =========================================================================
    # LOSS COMPUTATION
    # =========================================================================
    
    def compute_loss(self, y_true):
        # Add small epsilon to prevent log(0) = -inf
        epsilon = 1e-15
        loss = -np.log(self.y_pred[y_true] + epsilon)
        return float(loss)

    # =========================================================================
    # BACKPROPAGATION
    # =========================================================================
    
    def backward(self, y_true):
        # =====================================================================
        # Step 1: Compute gradient at output layer
        # =====================================================================
        
        # The gradient of cross-entropy loss w.r.t. softmax input (z2) is:
        # ∂L/∂z2 = y_pred - one_hot(y_true)
        #
        # Proof sketch:
        # L = -log(softmax(z2)_y) = -z2_y + log(Σ exp(z2_j))
        # ∂L/∂z2_i = -δ_{i,y} + exp(z2_i)/Σexp(z2_j) = -δ_{i,y} + y_pred_i
        # So ∂L/∂z2 = y_pred - one_hot(y)
        
        # Create one-hot vector for true class
        one_hot = np.zeros(self.n_classes)
        one_hot[y_true] = 1.0
        
        # Gradient w.r.t. z2 (output pre-activation)
        # Shape: (n_classes,) = (26,)
        dL_dz2 = self.y_pred - one_hot
        
        # =====================================================================
        # Step 2: Compute gradients for output layer parameters (W2, b2)
        # =====================================================================
        
        # Gradient w.r.t. W2
        # z2 = h @ W2 + b2, so ∂z2/∂W2 = h (treating as outer product)
        # ∂L/∂W2 = ∂L/∂z2 ⊗ ∂z2/∂W2 = outer(h, dL_dz2)
        # Shape: (n_hidden, n_classes) = (100, 26)
        dL_dW2 = np.outer(self.h, dL_dz2)
        
        # Gradient w.r.t. b2
        # z2 = h @ W2 + b2, so ∂z2/∂b2 = 1 (identity)
        # ∂L/∂b2 = ∂L/∂z2
        # Shape: (n_classes,) = (26,)
        dL_db2 = dL_dz2
        
        # =====================================================================
        # Step 3: Backpropagate gradient to hidden layer
        # =====================================================================
        
        # Gradient w.r.t. hidden layer output (h)
        # z2 = h @ W2 + b2, so ∂z2/∂h = W2
        # Using chain rule: ∂L/∂h = W2 @ ∂L/∂z2
        # Shape: (n_hidden,) = (100,)
        dL_dh = np.dot(self.W2, dL_dz2)
        
        # =====================================================================
        # Step 4: Backpropagate through ReLU activation
        # =====================================================================
        
        # Gradient w.r.t. z1 (hidden pre-activation)
        # h = relu(z1), so ∂h/∂z1 = relu'(z1) = 1 if z1 > 0 else 0
        # Using chain rule: ∂L/∂z1 = ∂L/∂h ⊙ relu'(z1)
        # Shape: (n_hidden,) = (100,)
        dL_dz1 = dL_dh * self.relu_derivative(self.z1)
        
        # =====================================================================
        # Step 5: Compute gradients for hidden layer parameters (W1, b1)
        # =====================================================================
        
        # Gradient w.r.t. W1
        # z1 = x @ W1 + b1, so ∂z1/∂W1 = x (treating as outer product)
        # ∂L/∂W1 = outer(x, dL_dz1)
        # Shape: (n_features, n_hidden) = (784, 100)
        dL_dW1 = np.outer(self.x, dL_dz1)
        
        # Gradient w.r.t. b1
        # z1 = x @ W1 + b1, so ∂z1/∂b1 = 1 (identity)
        # ∂L/∂b1 = ∂L/∂z1
        # Shape: (n_hidden,) = (100,)
        dL_db1 = dL_dz1
        
        # Return all gradients as a dictionary
        return {
            'W1': dL_dW1,
            'b1': dL_db1,
            'W2': dL_dW2,
            'b2': dL_db2
        }

    # =========================================================================
    # PARAMETER UPDATE (SGD)
    # =========================================================================
    
    def update_parameters(self, gradients):
        # Update each parameter
        self.W1 -= self.learning_rate * gradients['W1']
        self.b1 -= self.learning_rate * gradients['b1']
        self.W2 -= self.learning_rate * gradients['W2']
        self.b2 -= self.learning_rate * gradients['b2']

    # =========================================================================
    # TRAINING STEP (combining forward, backward, update)
    # =========================================================================
    
    def sgd_step(self, x_i, y_i):
        # Forward pass: compute prediction
        self.forward(x_i)
        
        # Compute loss
        loss = self.compute_loss(y_i)
        
        # Backward pass: compute gradients
        gradients = self.backward(y_i)
        
        # Update parameters
        self.update_parameters(gradients)
        
        return loss

    # =========================================================================
    # TRAINING EPOCH
    # =========================================================================
    
    def train_epoch(self, X, y):
        n_examples = X.shape[0]
        losses = []
        
        for i in range(n_examples):
            loss = self.sgd_step(X[i], y[i])
            losses.append(loss)
        
        return float(np.mean(losses))

    # =========================================================================
    # PREDICTION AND EVALUATION
    # =========================================================================
    
    def predict(self, X):
        n_examples = X.shape[0]
        predictions = np.zeros(n_examples, dtype=int)
        
        for i in range(n_examples):
            # Forward pass for this example
            probs = self.forward(X[i])
            # Predict class with highest probability
            predictions[i] = np.argmax(probs)
        
        return predictions

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return float(accuracy)


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def main(args):
    # Set random seed for reproducibility
    utils.configure_seed(seed=args.seed)
    
    # =========================================================================
    # Load dataset
    # =========================================================================
    print("Loading EMNIST Letters dataset...")
    
    # Load without bias - the MLP has its own bias terms
    data = utils.load_dataset(data_path=args.data_path, bias=False)
    
    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]
    
    # Get dimensions
    n_features = X_train.shape[1]  # 784
    n_classes = np.unique(y_train).size  # 26
    
    print(f"Dataset loaded:")
    print(f"  Training examples: {X_train.shape[0]}")
    print(f"  Validation examples: {X_valid.shape[0]}")
    print(f"  Test examples: {X_test.shape[0]}")
    print(f"  Input features: {n_features}")
    print(f"  Output classes: {n_classes}")
    
    # =========================================================================
    # Initialize MLP
    # =========================================================================
    print("\nInitializing MLP...")
    print(f"  Architecture: {n_features} -> {args.hidden_units} (ReLU) -> {n_classes} (Softmax)")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight init: N(μ=0.1, σ²=0.01)")
    print(f"  Bias init: zeros")
    
    model = MLP(
        n_features=n_features,
        n_hidden=args.hidden_units,
        n_classes=n_classes,
        learning_rate=args.learning_rate
    )
    
    # =========================================================================
    # Training loop
    # =========================================================================
    epochs = np.arange(1, args.epochs + 1)
    train_losses = []
    train_accs = []
    valid_accs = []
    
    best_valid_acc = 0.0
    best_epoch = -1
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    start_time = time.time()
    
    for epoch in epochs:
        # Shuffle training data each epoch
        # This is important for SGD to work well
        order = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[order]
        y_train_shuffled = y_train[order]
        
        # Train for one epoch
        avg_loss = model.train_epoch(X_train_shuffled, y_train_shuffled)
        
        # Evaluate on training and validation sets
        train_acc = model.evaluate(X_train, y_train)
        valid_acc = model.evaluate(X_valid, y_valid)
        
        # Store metrics
        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Valid Acc: {valid_acc:.4f}")
        
        # Save checkpoint if best validation accuracy so far
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch
            print("  ↳ New best validation accuracy! Saving checkpoint...")
            model.save(args.save_path)
    
    # Calculate training time
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print("="*60)
    print(f"Training completed in {minutes}m {seconds}s")
    print("="*60)
    
    # =========================================================================
    # Evaluate best model on test set
    # =========================================================================
    print(f"\nLoading best checkpoint from epoch {best_epoch}...")
    best_model = MLP.load(args.save_path)
    test_acc = best_model.evaluate(X_test, y_test)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_valid_acc:.4f} (epoch {best_epoch})")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"{'='*60}")
    
    # =========================================================================
    # Generate plots
    # =========================================================================
    
    # Plot training and validation accuracies
    utils.plot(
        "Epoch",
        "Accuracy",
        {"train": (epochs, train_accs), "valid": (epochs, valid_accs)},
        filename=args.accuracy_plot
    )
    print(f"\nAccuracy plot saved to: {args.accuracy_plot}")
    
    # Plot training loss
    utils.plot(
        "Epoch",
        "Loss",
        {"train_loss": (epochs, train_losses)},
        filename=args.loss_plot
    )
    print(f"Loss plot saved to: {args.loss_plot}")
    
    # =========================================================================
    # Save results to JSON
    # =========================================================================
    results = {
        "architecture": {
            "input_features": n_features,
            "hidden_units": args.hidden_units,
            "output_classes": n_classes,
            "activation": "relu"
        },
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": 1
        },
        "results": {
            "best_valid_acc": float(best_valid_acc),
            "best_epoch": int(best_epoch),
            "test_acc": float(test_acc)
        },
        "training_time_seconds": elapsed_time
    }
    
    with open(args.scores, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {args.scores}")
    
    print("\n" + "="*60)
    print("All done!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Multi-Layer Perceptron on EMNIST Letters (Question 1.3a)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python hw1-mlp.py --save-path mlp_checkpoint.pkl

This implements a single hidden layer MLP with:
- 100 hidden units
- ReLU activation
- Cross-entropy loss
- SGD with learning rate 0.001
        """
    )
    
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--hidden-units", type=int, default=100,
        help="Number of hidden units (default: 100)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001,
        help="SGD learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--data-path", type=str, default="emnist-letters.npz",
        help="Path to EMNIST dataset"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save-path", required=True,
        help="Path to save the best model checkpoint"
    )
    parser.add_argument(
        "--accuracy-plot", default="Q1-mlp-accs.pdf",
        help="Filename for accuracy plot"
    )
    parser.add_argument(
        "--loss-plot", default="Q1-mlp-loss.pdf",
        help="Filename for loss plot"
    )
    parser.add_argument(
        "--scores", default="Q1-mlp-scores.json",
        help="Filename for JSON results"
    )
    
    args = parser.parse_args()
    main(args)