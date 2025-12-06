import argparse
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import utils

# import network and helper functions
from hw1_ffn import FeedforwardNetwork, train_batch, predict, evaluate

# ============================================================
# Ensure output directory exists
# ============================================================
OUTPUT_DIR = "Q2_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Train one configuration
# ============================================================
def run_one_config(train_dataloader, train_X, train_y, dev_X, dev_y,
                   n_classes, n_feats, hidden_size, learning_rate, dropout, l2_val,
                   activation_type, optimizer_name, epochs):     

    model = FeedforwardNetwork(
        n_classes,
        n_feats,
        hidden_size,
        layers=1,
        activation_type=activation_type,
        dropout=dropout
    )

    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    optimizer = optims[optimizer_name](
        model.parameters(), lr=learning_rate, weight_decay=l2_val
    )

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(X_batch, y_batch, model, optimizer, criterion)

        # evaluate
        model.eval()
        _, train_acc = evaluate(model, train_X, train_y, criterion)
        val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

        best_val_acc = max(best_val_acc, val_acc)

    return best_val_acc


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str, default='emnist-letters.npz')
    parser.add_argument('-activation', choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer', choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-epochs', type=int, default=30)
    opt = parser.parse_args()

    utils.configure_seed(42)
    data = utils.load_dataset(opt.data_path)
    dataset = utils.ClassificationDataset(data)

    train_dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(42)
    )

    train_X, train_y = dataset.X, dataset.y
    dev_X, dev_y     = dataset.dev_X, dataset.dev_y

    n_classes = torch.unique(dataset.y).shape[0]
    n_feats   = dataset.X.shape[1]

    # ALL TRIALS USE DEFAULT RELU AND SGD
    widths = [16, 32, 64, 128, 256]
    #learning_rates = [0.0005, 0.001, 0.002, 0.005] #used for first trial
    #learning_rates = [0.001, 0.005, 0.01, 0.02] #used for second trial
    learning_rates = [0.02, 0.05, 0.08, 0.15] #used for third trial
    #dropouts = [0.0, 0.3] #used for first trial
    dropouts = [0.0, 0.5] #used for second trial and third trial
    #l2_vals  = [0.0, 0.001] #used for first trial
    l2_vals  = [0.0, 0.01] #used for second trial and third trial

    results = []

    # ============================================================
    # Part (a): GRID SEARCH — 80 configurations
    # ============================================================
    for width in widths:
        print(f"\n=== WIDTH = {width} ===")

        for lr in learning_rates:
            for dr in dropouts:
                for l2v in l2_vals:

                    best_val_acc = run_one_config(
                        train_dataloader, train_X, train_y,
                        dev_X, dev_y,
                        n_classes, n_feats,
                        hidden_size=width,
                        learning_rate=lr,
                        dropout=dr,
                        l2_val=l2v,
                        activation_type=opt.activation,
                        optimizer_name=opt.optimizer,
                        epochs=opt.epochs
                    )

                    results.append({
                        "width": width,
                        "learning_rate": lr,
                        "dropout": dr,
                        "l2": l2v,
                        "best_val_acc": best_val_acc
                    })

                    print(f"  lr={lr} dr={dr} l2={l2v} → best_val_acc={best_val_acc:.4f}")

    # ============================================================
    # Save Part (a) results
    # ============================================================
    df = pd.DataFrame(results)
    df.to_csv(f"{OUTPUT_DIR}/results.csv", index=False)

    print("\n================ RESULTS TABLE ================\n")
    print(df)

    # Highlight best per width
    print("\n============= BEST PER WIDTH =============\n")
    best_per_width = []

    for w in widths:
        sub = df[df["width"] == w]
        best_row = sub.loc[sub["best_val_acc"].idxmax()]
        best_per_width.append(best_row)

        print(
            f"WIDTH {w}: best_val_acc={best_row.best_val_acc:.4f} | "
            f"lr={best_row.learning_rate} | dropout={best_row.dropout} | l2={best_row.l2}"
        )

    pd.DataFrame(best_per_width).to_csv(
        f"{OUTPUT_DIR}/best_per_width.csv", index=False
    )

    # ============================================================
    # Part (b): Identify BEST MODEL across all configs
    # ============================================================


    
    # best_idx = df["best_val_acc"].idxmax()
    # best_cfg = df.loc[best_idx]

    # print("\n=========== BEST OVERALL CONFIGURATION ===========\n")
    # print(best_cfg)

    # # Retrain best model and record curves
    # train_losses = []
    # val_accs = []

    # model = FeedforwardNetwork(
    #     n_classes,
    #     n_feats,
    #     int(best_cfg.width),
    #     layers=1,
    #     activation_type=opt.activation,
    #     dropout=best_cfg.dropout
    # )

    # optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    # optimizer = optims[opt.optimizer](
    #     model.parameters(),
    #     lr=best_cfg.learning_rate,
    #     weight_decay=best_cfg.l2
    # )

    # criterion = nn.CrossEntropyLoss()

    # for ep in range(1, opt.epochs + 1):
    #     model.train()
    #     epoch_loss = 0.0

    #     for X_batch, y_batch in train_dataloader:
    #         loss = train_batch(X_batch, y_batch, model, optimizer, criterion)
    #         epoch_loss += loss

    #     train_losses.append(epoch_loss)

    #     # validation accuracy
    #     _, val_acc = evaluate(model, dev_X, dev_y, criterion)
    #     val_accs.append(val_acc)

    # # ============================================================
    # # Save plots
    # # ============================================================
    # import matplotlib.pyplot as plt

    # plt.figure()
    # plt.plot(train_losses)
    # plt.xlabel("Epoch")
    # plt.ylabel("Training Loss")
    # plt.title("Training Loss Curve")
    # plt.savefig(f"{OUTPUT_DIR}/best_model_train_loss.png")
    # plt.close()

    # plt.figure()
    # plt.plot(val_accs)
    # plt.xlabel("Epoch")
    # plt.ylabel("Validation Accuracy")
    # plt.title("Validation Accuracy Curve")
    # plt.savefig(f"{OUTPUT_DIR}/best_model_val_acc.png")
    # plt.close()

    # # ============================================================
    # # Test accuracy
    # # ============================================================
    # test_loss, test_acc = evaluate(model, dataset.test_X, dataset.test_y, criterion)

    # print(f"\nTEST ACCURACY OF BEST MODEL: {test_acc:.4f}\n")
    # print(f"All outputs saved in '{OUTPUT_DIR}/'.")


# ============================================================
# ENTRY
# ============================================================
if __name__ == "__main__":
    main()