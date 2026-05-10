
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog


def ask_to_save_model(model):
    root = tk.Tk()
    root.withdraw()

    should_save = messagebox.askyesno(
        title="Save model?",
        message="Do you want to save the trained digit recognition model?"
    )

    if not should_save:
        root.destroy()
        print("Model was not saved.")
        return False

    model_name = simpledialog.askstring(
        title="Model name",
        prompt="Enter a name for the model:"
    )

    root.destroy()

    if not model_name:
        print("No model name entered. Model was not saved.")
        return False

    # Make sure the file ends with .pth
    if not model_name.endswith(".pth"):
        model_name += ".pth"

    save_path = Path("models") / model_name
    model.save_model(save_path)

    print(f"Model saved as {save_path}")
    return True

def load_mnist(batch_size, training_mode):
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(
        root="./data",
        train=training_mode,
        download=True,
        transform = transform
    )
    return DataLoader(dataset,batch_size=batch_size, shuffle=True)