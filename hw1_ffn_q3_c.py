import os
import pandas as pd
import utils

# ============================================================
OUTPUT_DIR = "Q3_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Load depth results from CSV
depth_results = pd.read_csv(f"{OUTPUT_DIR}/depth_results.csv")

# ============================================================
# Extract depths and final training accuracy
depths = depth_results["depth"].tolist()
train_accs = depth_results["last_train_acc"].tolist()  # final epoch training accuracy

# ============================================================
# Prepare curves dictionary for utils.plot
curves = {"Training Accuracy": (depths, train_accs)}

# ============================================================
# Plot using utils.plot
utils.plot(
    x_label="Depth",
    y_label="Training Accuracy (final epoch)",
    curves=curves,
    filename=f"{OUTPUT_DIR}/training_acc_vs_depth.png"
)

print(f"Plot saved as '{OUTPUT_DIR}/training_acc_vs_depth.png'")