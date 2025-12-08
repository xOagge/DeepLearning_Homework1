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

#depth configurations
depths = [1, 3, 5, 7, 9]

# get best hyperparameters of the best 32 width model (best valdiation accuracy)
df = pd.read_csv("Q2_outputs/best_per_width.csv")
best_32_unit = df[df['width'] == 32].iloc[0]

#relu and sgd remain the same. exercise specifies 30 epochs
best_hyperparams = {
    "hidden_size": 32,
    "activation_type": "relu",
    "dropout": best_32_unit["dropout"],
    "optimizer_name": "sgd",
    "learning_rate": best_32_unit["learning_rate"],
    "l2_val": best_32_unit["l2"],
    "epochs": 30
}

#run different models with the same 32-width hyperparameters, but different depth values
results = []
for L in depths:
    print(f"\nTraining depth = {L}")
    
    #reset seed before every model run. for Q2.3 we use seed 43, developer use of free will :)
    train_dataloader = ffn_utils.get_dataloader(dataset, batch_size=64, seed=43)

    #define model, optimizer, criterion
    model, optimizer, criterion = ffn_utils.setup_model_and_optimizer(
        n_classes, n_feats, hidden_size=best_hyperparams["hidden_size"],
        layers=L, activation_type=best_hyperparams["activation_type"],
        dropout=best_hyperparams["dropout"],
        optimizer_name=best_hyperparams["optimizer_name"],
        learning_rate=best_hyperparams["learning_rate"],
        l2_val=best_hyperparams["l2_val"]
    )

    #train the model over epochs and save the best validation and train accuracies.
    # final training accuracy is going to be for Q2.3.c)
    best_val_acc = 0.0
    last_train_acc = 0.0
    for ep in range(best_hyperparams["epochs"]):

        #train mode
        model.train()
        for X_batch, y_batch in train_dataloader:
            train_batch(X_batch, y_batch, model, optimizer, criterion)

        # get validation accuracy of epoch, update best found validationa ccuracy
        val_acc = evaluate(model, dev_X, dev_y, criterion)[1]
        if val_acc > best_val_acc: best_val_acc = val_acc

        # if this si last epoch, store last training accuracy
        if ep == best_hyperparams["epochs"] - 1:
            last_train_acc = evaluate(model, train_X, train_y, criterion)[1]

    #print for sanity of the developer to see things are running
    print(f"Depth {L} → Best val acc: {best_val_acc:.4f}, Last train acc: {last_train_acc:.4f}")
    
    #save hyperparameters and accuracies
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

# save all the results to a csv
pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/depth_results.csv", index=False)
print(f"Results saved in {OUTPUT_DIR}/depth_results.csv")