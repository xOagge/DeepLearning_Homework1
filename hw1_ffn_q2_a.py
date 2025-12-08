import argparse
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import utils
import hw1_ffn_utils as ffn_utils

# import network and helper functions
from hw1_ffn import FeedforwardNetwork, train_batch, predict, evaluate


OUTPUT_DIR = "Q2_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#this function runs a model with defined hyperparameters as arguments
def run_one_config(dataset, dev_X, dev_y,
                   n_classes, n_feats, hidden_size, learning_rate, dropout, l2_val,
                   activation_type, optimizer_name, epochs):     

    #later the exact same way.
    train_dataloader = ffn_utils.get_dataloader(dataset, batch_size=64, seed=42)

    # define the model with the defined class in hw1_ffn, using this fucntions arguments
    # for hyperparameter values. Layer = 1 for whole Q2 exercise, there is no sense in even
    # trying other options like in activation and optimization, which are fixed
    model, optimizer, criterion = ffn_utils.setup_model_and_optimizer(
        n_classes, n_feats, hidden_size, layers=1,
        activation_type=activation_type, dropout=dropout,
        optimizer_name=optimizer_name, learning_rate=learning_rate, l2_val=l2_val
    )
    
    # In this part we iterate the model over the epochs, training with training data
    # and evaluating validation accuracy with dev dataset
    best_val_acc = 0.0
    for ep in range(1, epochs + 1):
        #set the model into training mode
        model.train()
        # train_dataloader only has training data, so we are iterating training data.
        #Each iteration is a batch of size 64, as defined
        for X_batch, y_batch in train_dataloader:
            #train model
            train_batch(X_batch, y_batch, model, optimizer, criterion)

        # set the model into evaluation mode
        model.eval()
        #get the validation accuracy of the model in current epoch.
        val_acc = evaluate(model, dev_X, dev_y, criterion)[1]
        #update best found validation accuracy
        best_val_acc = max(best_val_acc, val_acc)

    return best_val_acc

def main():
    #there are options to run the code with different activations and optimziers, but for the entirety of
    #the project the default options were used
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str, default='emnist-letters.npz')
    parser.add_argument('-activation', choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer', choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-epochs', type=int, default=30)
    opt = parser.parse_args()


    #load data, get datasets, n_classes, n_feats
    dataset, n_classes, n_feats = ffn_utils.setup_data()

    #extract the train dataset, to train the model
    train_X, train_y = dataset.X, dataset.y
    #extract the dev/valdiation set, to benchmark the models in the grid search
    dev_X, dev_y     = dataset.dev_X, dataset.dev_y

    #comment store other batches grid hyperparameters
    #learning_rates = [0.0005, 0.001, 0.002, 0.005] #used for first trial
    #learning_rates = [0.001, 0.005, 0.01, 0.02] #used for second trial
    #dropouts = [0.0, 0.3] #used for first trial
    #l2_vals  = [0.0, 0.001] #used for first trial
    #l2_vals  = [0.0, 0.001] #used for first trial

    # Grid hyperparameters used in the model
    widths = [16, 32, 64, 128, 256]
    learning_rates = [0.02, 0.05, 0.08, 0.15] #used for third trial
    dropouts = [0.0, 0.5] #used for second trial and third trial
    l2_vals  = [0.0, 0.01] #used for second trial and third trial

    results = []

    # do the grid seach for every combination of hyperparameters: 80 combinations
    for width in widths:
        print(f"\n=== WIDTH = {width} ===")
        for lr in learning_rates:
            for dr in dropouts:
                for l2v in l2_vals:

                    #this will return the best found validation accuracy value
                    best_val_acc = run_one_config( dataset, dev_X, dev_y, 
                        n_classes, n_feats, hidden_size=width,
                        learning_rate=lr, dropout=dr, l2_val=l2v,
                        activation_type=opt.activation,optimizer_name=opt.optimizer,
                        epochs=opt.epochs
                    )

                    #add to results the information of the hyperparameters that are variable
                    # int the grid search, and add to results with the best validation accuracy
                    results.append({"width": width, "learning_rate": lr,
                        "dropout": dr, "l2": l2v, "best_val_acc": best_val_acc
                    })

                    #print just to see progress of running code
                    print(f"    lr={lr} dr={dr} l2={l2v} → best_val_acc={best_val_acc:.4f}")

    #using panda library to store the results in a csv so we dont need to constantly run this code
    # (very time consuming), and more efficient.
    df = pd.DataFrame(results)
    df.to_csv(f"{OUTPUT_DIR}/results.csv", index=False)

    # we will also save the best combination bet width, essential for the report
    best_per_width = []

    for w in widths:
        #gets all rows with width value of w, current iteration value
        sub = df[df["width"] == w]
        #look for the row in sub with the highest best_val_acc, retrieving the 
        # the combination of hyperparameters with best performance for width = w
        best_row = sub.loc[sub["best_val_acc"].idxmax()]
        best_per_width.append(best_row) # store

    #store the best_per_width data in a csv
    pd.DataFrame(best_per_width).to_csv(
        f"{OUTPUT_DIR}/best_per_width.csv", index=False
    )

# main :)
if __name__ == "__main__":
    main()