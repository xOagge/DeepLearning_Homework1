import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import utils
import hw1_ffn_utils as ffn_utils
from hw1_ffn import FeedforwardNetwork, train_batch, evaluate


OUTPUT_DIR = "Q3_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


#load data, get datasets, n_classes, n_feats
dataset, n_classes, n_feats = ffn_utils.setup_data()

train_X, train_y = dataset.X, dataset.y
dev_X, dev_y     = dataset.dev_X, dataset.dev_y
test_X, test_y   = dataset.test_X, dataset.test_y

#read csv file from previous exercise, find model with highest best_val_acc, store depth of best model
depth_df = pd.read_csv(f"{OUTPUT_DIR}/depth_results.csv")
best_row = depth_df.loc[depth_df["best_val_acc"].idxmax()]
best_depth = int(best_row["depth"])
print(f"Best depth: {best_depth} (val acc = {best_row['best_val_acc']:.4f})")

hyper = {
    "hidden_size": int(best_row["hidden_size"]),
    "activation_type": "relu",
    "dropout": float(best_row["dropout"]),
    "learning_rate": float(best_row["learning_rate"]),
    "l2_val": float(best_row["l2_val"]),
    "epochs": 30,
}

#print parameters to check
print("\n ======= Considered Hyperparameters ======")
for key, val in hyper.items():
    print(f"{key}: {val}")
print("=================================\n")

#seed, define model, optimizer, criterion, train loader, as in the other exercises
train_loader = ffn_utils.get_dataloader(dataset, batch_size=64, seed=43)
model, optimizer, criterion = ffn_utils.setup_model_and_optimizer(
    n_classes, n_feats, hidden_size=hyper["hidden_size"],
    layers=best_depth, activation_type=hyper["activation_type"],
    dropout=hyper["dropout"], optimizer_name="sgd",
    learning_rate=hyper["learning_rate"], l2_val=hyper["l2_val"]
)


#train the model and store metrics
train_losses = []
val_accs = []
best_val = -1
test_acc_at_best_val = None
epoch_of_best_val = None
for epoch in range(hyper["epochs"]):
    #train mode
    model.train()

    #train model and get epoch loss to store
    epoch_loss = 0.0
    for Xb, yb in train_loader:
        loss = train_batch(Xb, yb, model, optimizer, criterion)
        epoch_loss += loss
    train_losses.append(epoch_loss)

    # get valdiation accuracy and store
    val_acc = evaluate(model, dev_X, dev_y, criterion)[1]
    val_accs.append(val_acc)

    # if at last epoch, store training accuracy (last epoch training accuracy)
    if epoch == hyper["epochs"] - 1:
        last_train_acc = evaluate(model, train_X, train_y, criterion)[1]

    # update best valdiation accuracy. if updated, update the corresponding epoch and test accuracy
    # at the best epoch
    if val_acc > best_val:
        best_val = val_acc
        epoch_of_best_val = epoch + 1
        test_acc_at_best_val = evaluate(model, test_X, test_y, criterion)[1]

print(f"\nBest validation accuracy = {best_val:.4f} at epoch {epoch_of_best_val}")
print(f"Test accuracy at that epoch = {test_acc_at_best_val:.4f}")
print(f"Last training accuracy (final epoch) = {last_train_acc:.4f}")

# save all info to csv
pd.DataFrame([{
    "best_depth": best_depth,
    "epoch_best_val": epoch_of_best_val,
    "best_val_accuracy": best_val,
    "test_accuracy_at_best_val": test_acc_at_best_val,
    "last_train_acc": last_train_acc
}]).to_csv(f"{OUTPUT_DIR}/best_depth_test_accuracy.csv", index=False)

print(f"\nSaved test accuracy CSV to {OUTPUT_DIR}/best_depth_test_accuracy.csv")

#plot training loss and validation accuracy over epochs
utils.plot( "Epoch", "Training Loss", {"Training Loss": (list(range(1, hyper["epochs"] + 1)), train_losses)},
    filename=f"{OUTPUT_DIR}/best_depth_train_loss.png")

utils.plot( "Epoch", "Validation Accuracy", {"Validation Accuracy": (list(range(1, hyper["epochs"] + 1)), val_accs)},
    filename=f"{OUTPUT_DIR}/best_depth_val_acc.png")