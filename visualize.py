import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_predictions(actual, predicted, save_path="results/gp_prediction.png"):
    """Plot Actual vs Predicted stock prices."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(14, 7))
    plt.plot(
        actual, label="Actual Price", color="#1f77b4", alpha=0.8, linewidth=1.5
    )
    plt.plot(
        predicted,
        label="Predicted Price",
        color="#ff7f0e",
        linestyle="--",
        alpha=0.9,
        linewidth=1.5,
    )

    plt.title(
        "GP Stock Price Prediction vs Actual (Walk-Forward Test)", fontsize=14
    )
    plt.xlabel("Test Steps / Days", fontsize=12)
    plt.ylabel("Close Price (BDT)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Chart saved to '{save_path}'")
    plt.show()


if __name__ == "__main__":
    print(
        "Run main.py with evaluation to collect actual vs predicted vectors."
    )