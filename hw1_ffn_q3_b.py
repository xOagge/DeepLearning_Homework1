import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import utils
from hw1_ffn import FeedforwardNetwork, train_batch, evaluate

# ============================================================
OUTPUT_DIR = "Q3_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Load data
data = utils.load_dataset("emnist-letters.npz")
dataset = utils.ClassificationDataset(data)
train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(42))
train_X, train_y = dataset.X, dataset.y
dev_X, dev_y     = dataset.dev_X, dataset.dev_y
test_X, test_y   = dataset.test_X, dataset.test_y

n_classes = torch.unique(dataset.y).shape[0]
n_feats   = dataset.X.shape[1]

# ============================================================
# Load results from Q3a to find best depth
depth_df = pd.read_csv(f"{OUTPUT_DIR}/depth_results.csv")
best_row = depth_df.loc[depth_df["best_val_acc"].idxmax()]
best_depth = int(best_row["depth"])
print(f"Best depth: {best_depth} with val acc {best_row['best_val_acc']:.4f}")

# Best hyperparameters (from previous 32-unit grid)
best_hyperparams = {
    "hidden_size": 32,
    "activation_type": best_row["activation"] if "activation" in best_row else "relu",
    "dropout": best_row["dropout"],
    "optimizer_name": best_row["optimizer"] if "optimizer" in best_row else "sgd",
    "learning_rate": best_row["learning_rate"],
    "l2_val": best_row["l2_val"],
    "epochs": 30
}

# ============================================================
# Retrain model and record curves
model = FeedforwardNetwork(
    n_classes, n_feats,
    hidden_size=best_hyperparams["hidden_size"],
    layers=best_depth,
    activation_type=best_hyperparams["activation_type"],
    dropout=best_hyperparams["dropout"]
)

optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
optimizer = optims[best_hyperparams["optimizer_name"]](
    model.parameters(),
    lr=best_hyperparams["learning_rate"],
    weight_decay=best_hyperparams["l2_val"]
)

criterion = nn.CrossEntropyLoss()

train_losses = []
val_accs = []

for ep in range(best_hyperparams["epochs"]):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_dataloader:
        loss = train_batch(X_batch, y_batch, model, optimizer, criterion)
        epoch_loss += loss
    train_losses.append(epoch_loss)

    _, val_acc = evaluate(model, dev_X, dev_y, criterion)
    val_accs.append(val_acc)

# ============================================================
# Plot curves
plt.figure()
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title(f"Training Loss Curve (Depth {best_depth})")
plt.savefig(f"{OUTPUT_DIR}/best_depth_train_loss.png")
plt.close()

plt.figure()
plt.plot(val_accs)
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title(f"Validation Accuracy Curve (Depth {best_depth})")
plt.savefig(f"{OUTPUT_DIR}/best_depth_val_acc.png")
plt.close()

# Test accuracy
test_loss, test_acc = evaluate(model, test_X, test_y, criterion)
print(f"Test accuracy of best depth model: {test_acc:.4f}")