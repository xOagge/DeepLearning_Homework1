import os
import pandas as pd
import utils

#work is all done in Q2.3.a, where the final training accuracy is stored for each model

#read depth_results.csv
OUTPUT_DIR = "Q3_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
depth_results = pd.read_csv(f"{OUTPUT_DIR}/depth_results.csv")

#get the value of last_train_acc for each row
depths = depth_results["depth"].tolist()
train_accs = depth_results["last_train_acc"].tolist()

#plot the final training accuracy over depth of model
curves = {"Training Accuracy": (depths, train_accs)}
utils.plot( x_label="Depth", y_label="Training Accuracy (final epoch)", curves=curves,
    filename=f"{OUTPUT_DIR}/training_acc_vs_depth.png")

#sanity print
print(f"Plot saved as '{OUTPUT_DIR}/training_acc_vs_depth.png'")