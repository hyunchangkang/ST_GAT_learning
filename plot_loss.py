import numpy as np
import matplotlib.pyplot as plt
import os

def plot_history(save_dir="output/"):
    history_path = os.path.join(save_dir, "training_history.npy")
    if not os.path.exists(history_path):
        print("History file not found.")
        return

    # Load history
    history = np.load(history_path, allow_pickle=True).item()
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    print(f"Loss curve saved to {save_dir}/loss_curve.png")
    plt.show()

if __name__ == "__main__":
    plot_history()