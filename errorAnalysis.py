import torch
import matplotlib.pyplot as plt

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc

def show_wrong_mnist_predictions(model, test_loader, n=16):
    model.eval()

    wrong_images = []
    wrong_true_labels = []
    wrong_pred_labels = []

    with torch.inference_mode():
        for X, y in test_loader:

            logits = model(X)
            preds = logits.argmax(dim=1)

            wrong_mask = preds != y

            wrong_images.append(X[wrong_mask])
            wrong_true_labels.append(y[wrong_mask])
            wrong_pred_labels.append(preds[wrong_mask])

            total_wrong = sum(len(labels) for labels in wrong_true_labels)

            if total_wrong >= n:
                break

    wrong_images = torch.cat(wrong_images)[:n]
    wrong_true_labels = torch.cat(wrong_true_labels)[:n]
    wrong_pred_labels = torch.cat(wrong_pred_labels)[:n]

    plt.figure(figsize=(4, 4))

    for i in range(len(wrong_images)):
        plt.subplot(4, 4, i + 1)

        plt.imshow(wrong_images[i].squeeze(), cmap="gray")

        true_label = wrong_true_labels[i].item()
        pred_label = wrong_pred_labels[i].item()

        plt.title(f"T: {true_label} | P: {pred_label}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
