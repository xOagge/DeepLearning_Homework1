import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import utils
from hw1_ffn import FeedforwardNetwork, train_batch, evaluate

# ============================================================
# Setup
# ============================================================
OUTPUT_DIR = "Q3_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure seed
utils.configure_seed(43)

# ============================================================
# Load data
# ============================================================
data = utils.load_dataset("emnist-letters.npz")
dataset = utils.ClassificationDataset(data)
train_X, train_y = dataset.X, dataset.y
dev_X, dev_y     = dataset.dev_X, dataset.dev_y
test_X, test_y   = dataset.test_X, dataset.test_y

n_classes = torch.unique(train_y).shape[0]
n_feats   = train_X.shape[1]

# ============================================================
# Load depth results to get best depth & hyperparameters
# ============================================================
depth_df = pd.read_csv(f"{OUTPUT_DIR}/depth_results.csv")
best_row = depth_df.loc[depth_df["best_val_acc"].idxmax()]
best_depth = int(best_row["depth"])

print(f"Best depth: {best_depth} (val acc = {best_row['best_val_acc']:.4f})")

hyper = {
    "hidden_size": int(best_row["hidden_size"]),
    "activation_type": best_row["activation_type"],
    "dropout": float(best_row["dropout"]),
    "optimizer_name": best_row["optimizer_name"],
    "learning_rate": float(best_row["learning_rate"]),
    "l2_val": float(best_row["l2_val"]),
    "epochs": 30,
    "batch_size": 64
}

# ============================================================
# Print hyperparameters
# ============================================================
print("\n=== Considered Hyperparameters ===")
for k, v in hyper.items():
    print(f"{k}: {v}")
print("=================================\n")

# ============================================================
# Build model (do this BEFORE DataLoader to match first script)
# ============================================================
model = FeedforwardNetwork(
    n_classes,
    n_feats,
    hidden_size=hyper["hidden_size"],
    layers=best_depth,
    activation_type=hyper["activation_type"],
    dropout=hyper["dropout"]
)

optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
optimizer = optims[hyper["optimizer_name"]](
    model.parameters(),
    lr=hyper["learning_rate"],
    weight_decay=hyper["l2_val"]
)

criterion = nn.CrossEntropyLoss()

# ============================================================
# Create DataLoader AFTER model to avoid RNG drift
# ============================================================
train_loader = DataLoader(
    dataset,
    batch_size=hyper["batch_size"],
    shuffle=True,
    generator=torch.Generator().manual_seed(43)  # same generator as first script
)

# ============================================================
# Training loop (identical to first script)
# ============================================================
train_losses = []
val_accs = []
best_val = -1
test_acc_at_best_val = None
epoch_of_best_val = None

for epoch in range(hyper["epochs"]):
    model.train()
    epoch_loss = 0.0
    for Xb, yb in train_loader:
        loss = train_batch(Xb, yb, model, optimizer, criterion)
        epoch_loss += loss

    train_losses.append(epoch_loss)

    # Validation accuracy
    _, val_acc = evaluate(model, dev_X, dev_y, criterion)
    val_accs.append(val_acc)

    # Store training accuracy only at last epoch
    if epoch == hyper["epochs"] - 1:
        _, last_train_acc = evaluate(model, train_X, train_y, criterion)

    # Test accuracy at best validation epoch
    if val_acc > best_val:
        best_val = val_acc
        epoch_of_best_val = epoch + 1
        _, test_acc_at_best_val = evaluate(model, test_X, test_y, criterion)

print(f"\nBest validation accuracy = {best_val:.4f} at epoch {epoch_of_best_val}")
print(f"Test accuracy at that epoch = {test_acc_at_best_val:.4f}")
print(f"Last training accuracy (final epoch) = {last_train_acc:.4f}")

# ============================================================
# Save test accuracy information to CSV
# ============================================================
pd.DataFrame([{
    "best_depth": best_depth,
    "epoch_best_val": epoch_of_best_val,
    "best_val_accuracy": best_val,
    "test_accuracy_at_best_val": test_acc_at_best_val,
    "last_train_acc": last_train_acc
}]).to_csv(f"{OUTPUT_DIR}/best_depth_test_accuracy.csv", index=False)

print(f"\nSaved test accuracy CSV to {OUTPUT_DIR}/best_depth_test_accuracy.csv")

# ============================================================
# Plot training loss curve and validation accuracy curve
# ============================================================
utils.plot(
    "Epoch", "Training Loss",
    {"Training Loss": (list(range(1, hyper["epochs"] + 1)), train_losses)},
    filename=f"{OUTPUT_DIR}/best_depth_train_loss.png"
)

utils.plot(
    "Epoch", "Validation Accuracy",
    {"Validation Accuracy": (list(range(1, hyper["epochs"] + 1)), val_accs)},
    filename=f"{OUTPUT_DIR}/best_depth_val_acc.png"
)