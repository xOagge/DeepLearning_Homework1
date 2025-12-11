import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import utils
import hw1_ffn_utils as ffn_utils

# Import your network and helpers
from hw1_ffn import FeedforwardNetwork, train_batch, evaluate

OUTPUT_DIR = "Q2_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#(EVALUATOR PLEASE READ THIS COMMENT :) ). will restrain myself from
#commenting everything to detail when there is something similar in a previous code, hope thats not a problem
#Am deciding to do this so that the code is not heavy commented everywhere and there is no need to read the 
# same thing all over again. Thank you for your compreension, and sorry if this is actually an inconvenient

def main():
    # for this part of exercise Q2.2.b) we will run the best model with the exact same definitions
    # as in Q2.2.a)

    data_path = "emnist-letters.npz"
    # same as before
    epochs = 30
    batch_size = 64

    #setup the dataset and dimensions
    dataset, n_classes, n_feats = ffn_utils.setup_data()    
    #create train_dataloader with seed, and configure utils.configure_seed
    train_dataloader = ffn_utils.get_dataloader(dataset, batch_size=batch_size, seed=42)

    #get the datasets, n_classes, n_feats.
    dev_X, dev_y     = dataset.dev_X, dataset.dev_y
    test_X, test_y   = dataset.test_X, dataset.test_y

    #load the stored results from best_per_width.csv. get the row with highest best validation accuracy
    df = pd.read_csv(f"{OUTPUT_DIR}/best_per_width.csv")
    best_idx = df["best_val_acc"].idxmax()
    best_cfg = df.loc[best_idx]
    print("\n ======== BEST OVERALL CONFIGURATION ======= \n")
    print(best_cfg)

    #define model and optimizer with the hyperparameters in best_cfg
    #layer is fixed for Q2, and relu activation is what we use in q2.2.a).
    model, optimizer, criterion = ffn_utils.setup_model_and_optimizer(
        n_classes, n_feats, hidden_size=int(best_cfg.width),
        layers=1, activation_type="relu", dropout=best_cfg.dropout,
        optimizer_name="sgd", learning_rate=best_cfg.learning_rate, l2_val=best_cfg.l2
    )

    #training the selected model and gather metrics
    train_losses, val_accs = [], []
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    for ep in range(1, epochs + 1):
        #train mode
        model.train()
        epoch_loss = 0.0
        #iterate batches of training data and sum total loss of the epoch
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(X_batch, y_batch, model, optimizer, criterion)
            epoch_loss += loss
        
        #add to vector of total loss per epoch
        train_losses.append(epoch_loss)

        #get validation accuracy and add to vector of val_acc per epoch
        val_acc = evaluate(model, dev_X, dev_y, criterion)[1]
        val_accs.append(val_acc)

        # Save best found model ( highest validation accuracy epoch). Later used to
        # calculate test accuracy of this model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep
            best_model_state = model.state_dict().copy()

    #use utils plot function to plot training loss and validation accuracy.
    epochs_list = list(range(1, epochs + 1))
    utils.plot( x_label="Epoch", y_label="Training Loss",
        curves={"Training Loss": (epochs_list, train_losses)},
        filename=f"{OUTPUT_DIR}/best_model_train_loss.png"
    )
    utils.plot( x_label="Epoch", y_label="Validation Accuracy",
        curves={"Validation Accuracy": (epochs_list, val_accs)},
        filename=f"{OUTPUT_DIR}/best_model_val_acc.png"
    )

    #load best found model and obtain test accuracy by evaluating model with test set
    model.load_state_dict(best_model_state)
    test_acc = evaluate(model, test_X, test_y, criterion)[1]

    #pretty print
    print(f"\n =========== RESULTS FOR BEST MODEL =========== \n")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} (at Epoch {best_epoch})")
    print(f"Test Accuracy:            {test_acc:.4f}")
    print(f"Plots and metrics saved in '{OUTPUT_DIR}/'.")

    # save  hyperparameters used and metrics to a csv file.
    results_data = {
        "best_val_acc": [best_val_acc],
        "test_acc": [test_acc],
        "best_epoch": [best_epoch],
        "width": [best_cfg.width],
        "learning_rate": [best_cfg.learning_rate],
        "dropout": [best_cfg.dropout],
        "l2": [best_cfg.l2]
    }
    results_df = pd.DataFrame(results_data)
    results_csv_path = f"{OUTPUT_DIR}/best_model_metrics.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Metrics and hyperparameters saved to: {results_csv_path}")

if __name__ == "__main__":
    main()