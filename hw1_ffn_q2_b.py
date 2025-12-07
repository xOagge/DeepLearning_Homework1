import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import utils

# Import your network and helpers
from hw1_ffn import FeedforwardNetwork, train_batch, evaluate

OUTPUT_DIR = "Q2_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # ==================== Config ====================
    data_path = "emnist-letters.npz"
    epochs = 30
    batch_size = 64

    # ==================== Load data ====================
    utils.configure_seed(42)
    data = utils.load_dataset(data_path)
    dataset = utils.ClassificationDataset(data)

    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(42)
    )

    train_X, train_y = dataset.X, dataset.y
    dev_X, dev_y     = dataset.dev_X, dataset.dev_y

    n_classes = torch.unique(dataset.y).shape[0]
    n_feats   = dataset.X.shape[1]

    # ==================== Load best config ====================
    df = pd.read_csv(f"{OUTPUT_DIR}/results.csv")
    best_idx = df["best_val_acc"].idxmax()
    best_cfg = df.loc[best_idx]

    print("\n=========== BEST OVERALL CONFIGURATION ===========\n")
    print(best_cfg)

    # ==================== Model ====================
    model = FeedforwardNetwork(
        n_classes=n_classes,
        n_features=n_feats,
        hidden_size=int(best_cfg.width),
        layers=1,
        activation_type="relu",  # fixed as per project instructions
        dropout=best_cfg.dropout
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=best_cfg.learning_rate,
        weight_decay=best_cfg.l2
    )

    criterion = nn.CrossEntropyLoss()

    # ==================== Training ====================
    train_losses, val_accs = [], []
    best_val_acc = 0.0
    best_model_state = None

    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(X_batch, y_batch, model, optimizer, criterion)
            epoch_loss += loss
        train_losses.append(epoch_loss)

        _, val_acc = evaluate(model, dev_X, dev_y, criterion)
        val_accs.append(val_acc)

        # Save model state if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    # ==================== Plotting ====================
    epochs_list = list(range(1, epochs + 1))
    utils.plot(
        x_label="Epoch",
        y_label="Training Loss",
        curves={"Training Loss": (epochs_list, train_losses)},
        filename=f"{OUTPUT_DIR}/best_model_train_loss.png"
    )

    utils.plot(
        x_label="Epoch",
        y_label="Validation Accuracy",
        curves={"Validation Accuracy": (epochs_list, val_accs)},
        filename=f"{OUTPUT_DIR}/best_model_val_acc.png"
    )

    # ==================== Test Accuracy ====================
    # Load best model (epoch with highest validation accuracy)
    model.load_state_dict(best_model_state)
    test_loss, test_acc = evaluate(model, dataset.test_X, dataset.test_y, criterion)
    print(f"\nTEST ACCURACY OF BEST MODEL (epoch with max val acc): {test_acc:.4f}")
    print(f"All outputs saved in '{OUTPUT_DIR}/'.")

if __name__ == "__main__":
    main()