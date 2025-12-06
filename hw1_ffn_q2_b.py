import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import utils

# Import your network and helpers
from hw1_ffn import FeedforwardNetwork, train_batch, evaluate

OUTPUT_DIR = "Q2_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str, default='emnist-letters.npz')
    parser.add_argument('-activation', choices=['tanh','relu'], default='relu')
    parser.add_argument('-optimizer', choices=['sgd','adam'], default='sgd')
    parser.add_argument('-epochs', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=64)
    opt = parser.parse_args()

    utils.configure_seed(42)
    data = utils.load_dataset(opt.data_path)
    dataset = utils.ClassificationDataset(data)

    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(42)
    )

    train_X, train_y = dataset.X, dataset.y
    dev_X, dev_y = dataset.dev_X, dataset.dev_y

    n_classes = torch.unique(dataset.y).shape[0]
    n_feats = dataset.X.shape[1]

    # ============================================================
    # Load results from Part A
    # ============================================================
    df = pd.read_csv(f"{OUTPUT_DIR}/results.csv")
    best_idx = df["best_val_acc"].idxmax()
    best_cfg = df.loc[best_idx]

    print("\n=========== BEST OVERALL CONFIGURATION ===========\n")
    print(best_cfg)

    # ============================================================
    # Retrain best model
    # ============================================================
    model = FeedforwardNetwork(
        n_classes,
        n_feats,
        int(best_cfg.width),
        layers=1,
        activation_type=opt.activation,
        dropout=best_cfg.dropout
    )

    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    optimizer = optims[opt.optimizer](
        model.parameters(),
        lr=best_cfg.learning_rate,
        weight_decay=best_cfg.l2
    )

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accs = []

    for ep in range(1, opt.epochs+1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(X_batch, y_batch, model, optimizer, criterion)
            epoch_loss += loss
        train_losses.append(epoch_loss)

        _, val_acc = evaluate(model, dev_X, dev_y, criterion)
        val_accs.append(val_acc)

    # ============================================================
    # Save plots
    # ============================================================
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.savefig(f"{OUTPUT_DIR}/best_model_train_loss.png")
    plt.close()

    plt.figure()
    plt.plot(val_accs)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.savefig(f"{OUTPUT_DIR}/best_model_val_acc.png")
    plt.close()

    # ============================================================
    # Test accuracy
    # ============================================================
    test_loss, test_acc = evaluate(model, dataset.test_X, dataset.test_y, criterion)
    print(f"\nTEST ACCURACY OF BEST MODEL: {test_acc:.4f}")
    print(f"All outputs saved in '{OUTPUT_DIR}/'.")

if __name__ == "__main__":
    main()