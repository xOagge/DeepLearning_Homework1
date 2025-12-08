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

# Load data once (heavy lifting)
data = utils.load_dataset("emnist-letters.npz")
dataset = utils.ClassificationDataset(data)
train_X, train_y = dataset.X, dataset.y
dev_X, dev_y     = dataset.dev_X, dataset.dev_y

n_classes = torch.unique(dataset.y).shape[0]
n_feats   = dataset.X.shape[1]

# ============================================================
# Depth settings & Hyperparameters
# ============================================================
depths = [1, 3, 5, 7, 9]

# Load best hyperparameters from Q2
df = pd.read_csv("Q2_outputs/best_per_width.csv")
best_32_unit = df[df['width'] == 32].iloc[0]

best_hyperparams = {
    "hidden_size": 32,
    "activation_type": best_32_unit["activation"] if "activation" in best_32_unit else "relu",
    "dropout": best_32_unit["dropout"],
    "optimizer_name": best_32_unit["optimizer"] if "optimizer" in best_32_unit else "sgd",
    "learning_rate": best_32_unit["learning_rate"],
    "l2_val": best_32_unit["l2"],
    "epochs": 30
}

results = []

for L in depths:
    print(f"\nTraining depth = {L}")
    
    # -------------------------------------------------------------
    # CRITICAL CHANGE: Reset Seed and DataLoader for every depth
    # This ensures each depth experiment is independent.
    # -------------------------------------------------------------
    utils.configure_seed(43)
    
    train_dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(43)
    )
    # -------------------------------------------------------------

    model = FeedforwardNetwork(
        n_classes, n_feats,
        hidden_size=best_hyperparams["hidden_size"],
        layers=L,
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
    best_val_acc = 0.0
    last_train_acc = 0.0

    for ep in range(best_hyperparams["epochs"]):
        model.train()
        for X_batch, y_batch in train_dataloader:
            train_batch(X_batch, y_batch, model, optimizer, criterion)

        # Validation accuracy
        _, val_acc = evaluate(model, dev_X, dev_y, criterion)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Store last epoch training accuracy
        if ep == best_hyperparams["epochs"] - 1:
            _, last_train_acc = evaluate(model, train_X, train_y, criterion)

    print(f"Depth {L} → Best val acc: {best_val_acc:.4f}, Last train acc: {last_train_acc:.4f}")
    results.append({
        "depth": L,
        "hidden_size": best_hyperparams["hidden_size"],
        "activation_type": best_hyperparams["activation_type"],
        "dropout": best_hyperparams["dropout"],
        "optimizer_name": best_hyperparams["optimizer_name"],
        "learning_rate": best_hyperparams["learning_rate"],
        "l2_val": best_hyperparams["l2_val"],
        "best_val_acc": best_val_acc,
        "last_train_acc": last_train_acc
    })

# Save results
pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/depth_results.csv", index=False)
print(f"Results saved in {OUTPUT_DIR}/depth_results.csv")