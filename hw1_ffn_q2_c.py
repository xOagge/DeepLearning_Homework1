import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "Q2_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # Load CSV with best validation accuracies
    df = pd.read_csv(f"{OUTPUT_DIR}/best_per_width.csv")

    # Extract widths and accuracies
    widths = df["width"].tolist()
    accuracies = df["best_val_acc"].tolist()

    # Print for confirmation
    print("Widths:", widths)
    print("Accuracies:", accuracies)

    # Plot
    plt.figure()
    plt.plot(widths, accuracies, marker='o')
    plt.xlabel("Hidden Layer Width")
    plt.ylabel("Best Validation Accuracy")
    plt.title("Validation Accuracy vs Hidden Layer Width")
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/training_accuracy_vs_width.png")
    plt.close()

    print("\nPlot saved as 'training_accuracy_vs_width.png' in Q2_outputs folder.")

if __name__ == "__main__":
    main()