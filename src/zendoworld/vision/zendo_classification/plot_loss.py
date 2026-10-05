import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from scipy.optimize import curve_fit

def log_decay(x, a, b):
    return -a * np.log(x) + b

def log_converge(x, a, b, c):
    return a * np.log(x) / (x ** b) + c

def shifted_log_converge(x, a, b, c, d):
    return a * np.log(x + d) / ((x + d) ** b) + c

colors = [
    "#A8E6A3",  # pastel green
    "#7FD1B9",  # minty teal
    "#80BDE3",  # soft sky blue – good accent
    "#A3A8E6",  # lavender blue
    "#D4A3E6",  # pastel lilac
    "#F4A3C0",  # rosy pink
    "#F4A3A3",  # light coral
    "#F4CBA3",  # peach
    "#E9EB8C",  # light yellow
]

def plot_losses(csv_path):
    df = pd.read_csv(csv_path)
    epochs = df['epoch']
    loss_columns = [col for col in df.columns if col != 'epoch']

    output_dir = Path(csv_path).with_suffix('')
    output_dir.mkdir(exist_ok=True)
    fits = []

    for i, col in enumerate(loss_columns):
        x_original = epochs.values
        y_original = df[col].values

        x_all = epochs.values
        y_all = df[col].values

        mask = (x_all != 0) & (y_all != 0)
        x = x_all[mask]
        y = y_all[mask]
        plt.figure()
        try:
            popt, _ = curve_fit(
                shifted_log_converge,
                x, y,
                bounds=([0, 0, 0, 0.1], [10, 5, 1, 30]),  # adjust as needed
                maxfev=10000
            )
            y_pred = shifted_log_converge(x, *popt)
            plt.plot(x, y_pred, label=col + " (fitted)", color=colors[i % len(colors)], linewidth=2)
            fits.append((x, y_pred))
        except RuntimeError:
            print(f"Fit failed for {col}")
            plt.plot(x_original, y_original, label=col + " (raw)", linestyle='--', color=colors[i % len(colors)])
        plt.plot(x_original, y_original, marker='o', linestyle='-', label=col, color=colors[(i+1) % len(colors)])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{col} Loss over Epochs")
        plt.grid(True)
        plt.legend()
        output_path = output_dir / f"{col}_loss.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    plt.figure(figsize=(10, 6))
    for i, col in enumerate(loss_columns):
        if col != "bbox":
            [x, y] = fits[i]
            plt.plot(x, y, label=col, color=colors[i % len(colors)], linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Fitted Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    combined_path = output_dir / "all_losses.png"
    plt.savefig(combined_path)
    plt.close()
    print(f"Saved: {combined_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_losses.py <path_to_loss_csv>")
        sys.exit(1)

    plot_losses(sys.argv[1])
