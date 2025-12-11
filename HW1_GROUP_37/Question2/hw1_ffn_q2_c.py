import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import utils
import hw1_ffn_utils as ffn_utils

from hw1_ffn import FeedforwardNetwork, train_batch, evaluate

OUTPUT_DIR = "Q2_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    #load data, get datasets, n_classes, n_feats
    dataset, n_classes, n_feats = ffn_utils.setup_data()

    train_X, train_y = dataset.X, dataset.y

    # get the best results per width hyperparameters
    df = pd.read_csv(f"{OUTPUT_DIR}/best_per_width.csv")
    
    # for each model retrain with same seed and retrive the final training accuracy
    widths = []
    final_train_accs = []
    # index and row, dotn want index
    for _, row in df.iterrows():
        #get variable hyperparameters
        width = int(row.width)
        dropout = float(row.dropout)
        learning_rate = float(row.learning_rate)
        l2_val = float(row.l2)

        #print to sanity check
        print(f"Training Width {width} (LR={learning_rate}, Drop={dropout}, L2={l2_val})")

        #as stated before, important to reset the seed to the same value (42 choosen)
        # to make sure simulations are the same
        train_dataloader = ffn_utils.get_dataloader(dataset, batch_size=64, seed=42)

        #model and optimizer defined as before, relu, 1 layer, SGD optimizer
        #same criterion as before
        model, optimizer, criterion = ffn_utils.setup_model_and_optimizer(
            n_classes, n_feats, hidden_size=width,
            layers=1, activation_type="relu", dropout=dropout,
            optimizer_name="sgd", learning_rate=learning_rate, l2_val=l2_val
        )

        # Train for 30 epochs
        for _ in range(30):
            model.train()
            for X_batch, y_batch in train_dataloader:
                train_batch(X_batch, y_batch, model, optimizer, criterion)

        # Evaluate final training accuracy
        train_acc = evaluate(model, train_X, train_y, criterion)[1]
        widths.append(width)
        final_train_accs.append(train_acc)

        print(f"Final training accuracy: {train_acc:.4f}")

    #plot the final training accuracy as function of the width of the model
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