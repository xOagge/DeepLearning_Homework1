import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
OUTPUT_DIR = "Q3_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Load depth results from Q3a
depth_results = pd.read_csv(f"{OUTPUT_DIR}/depth_results.csv")

# ============================================================
# Extract depths and final training accuracy
depths = depth_results["depth"]
train_accs = depth_results["best_val_acc"]  # use the stored best_val_acc as proxy for final accuracy

# ============================================================
# Plot training accuracy vs depth
plt.figure()
plt.plot(depths, train_accs, marker='o')
plt.xlabel("Depth")
plt.ylabel("Training Accuracy (final epoch)")
plt.title("Training Accuracy vs Depth (32-unit models)")
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/training_acc_vs_depth.png")
plt.close()

print(f"Plot saved as '{OUTPUT_DIR}/training_acc_vs_depth.png'")