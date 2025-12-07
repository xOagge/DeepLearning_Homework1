import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import utils

from hw1_ffn import FeedforwardNetwork, train_batch, evaluate

OUTPUT_DIR = "Q2_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # --------------------------
    # Load dataset
    # --------------------------
    utils.configure_seed(42)
    data = utils.load_dataset("emnist-letters.npz")
    dataset = utils.ClassificationDataset(data)

    train_dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(42)
    )

    train_X, train_y = dataset.X, dataset.y
    n_classes = torch.unique(dataset.y).shape[0]
    n_feats = dataset.X.shape[1]

    # --------------------------
    # Load best per width from CSV
    # --------------------------
    df = pd.read_csv(f"{OUTPUT_DIR}/best_per_width.csv")

    widths = []
    final_train_accs = []

    # --------------------------
    # For each width, retrain the best config
    # --------------------------
    for _, row in df.iterrows():
        width = int(row.width)
        dropout = float(row.dropout)
        learning_rate = float(row.learning_rate)
        l2_val = float(row.l2)

        model = FeedforwardNetwork(
            n_classes,
            n_feats,
            hidden_size=width,
            layers=1,
            activation_type="relu",  # fixed for project
            dropout=dropout
        )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=l2_val
        )

        criterion = nn.CrossEntropyLoss()

        # Train for 30 epochs
        for _ in range(30):
            model.train()
            for X_batch, y_batch in train_dataloader:
                train_batch(X_batch, y_batch, model, optimizer, criterion)

        # Evaluate final training accuracy
        _, train_acc = evaluate(model, train_X, train_y, criterion)
        widths.append(width)
        final_train_accs.append(train_acc)

        print(f"Width {width}: final training accuracy = {train_acc:.4f}")

    # --------------------------
    # Plot training accuracy vs width
    # --------------------------
    curve_dict = {
        "Training Accuracy": (widths, final_train_accs)
    }
    utils.plot(
        x_label="Hidden Layer Width",
        y_label="Final Training Accuracy",
        curves=curve_dict,
        filename=f"{OUTPUT_DIR}/training_accuracy_vs_width.png"
    )
    print(f"\nPlot saved as '{OUTPUT_DIR}/training_accuracy_vs_width.png'")

if __name__ == "__main__":
    main()