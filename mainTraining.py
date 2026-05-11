import torch
from torch import nn
from model import DigitRecognizitonModel
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from errorAnalysis import accuracy_fn
from errorAnalysis import show_wrong_mnist_predictions

from helpingMethods import ask_to_save_model
from helpingMethods import load_mnist


train_loader = load_mnist(32, training_mode=True)
test_loader = load_mnist(32, training_mode=False)

model = DigitRecognizitonModel(10)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

epochs = 1

train_acc_values = []
train_batch_steps = []

test_acc_values = []
test_batch_steps = []

epoch_boundaries = []

#Create training and testing Loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}")
    train_loss = 0

    model.train()

    for batch, (X, y) in enumerate(train_loader):
        global_batch = epoch * len(train_loader) + batch

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Calculate train accuracy for this batch
        train_acc = accuracy_fn(
            y_true=y,
            y_pred=y_pred.argmax(dim=1)
        )

        train_acc_values.append(train_acc)
        train_batch_steps.append(global_batch)

        # 4. Optimizer zero grad
        optimizer.zero_grad()

        # 5. Backpropagation
        loss.backward()

        # 6. Optimizer step
        optimizer.step()

    train_loss /= len(train_loader)
    ###Testing
    # #Setup variables for accumulatvely adding up loss and accuracy
    test_loss, test_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for X, y in test_loader:
            test_pred = model(X)

            test_loss += loss_fn(test_pred, y).item()

            test_acc += accuracy_fn(
                y_true=y,
                y_pred=test_pred.argmax(dim=1)
            )

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

    epoch_end_batch = (epoch + 1) * len(train_loader) - 1

    test_acc_values.append(test_acc)
    test_batch_steps.append(epoch_end_batch)

    epoch_boundaries.append(epoch_end_batch)

    print(f'Train loss {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}:%\n')


plt.figure(figsize=(10, 5))

plt.plot(
    train_batch_steps,
    train_acc_values,
    label="Train accuracy per batch"
)

plt.plot(
    test_batch_steps,
    test_acc_values,
    marker="o",
    label="Test accuracy after each epoch"
)

for epoch_idx, boundary in enumerate(epoch_boundaries):
    plt.axvline(
        x=boundary,
        linestyle="--",
        alpha=0.5
    )

    plt.text(
        boundary,
        min(train_acc_values),
        f"Epoch {epoch_idx}",
        rotation=90,
        verticalalignment="bottom"
    )

plt.xlabel("Batch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Test Accuracy over Batches")
plt.legend()
plt.tight_layout()
plt.show()
ask_to_save_model(model)
show_wrong_mnist_predictions(model, test_loader)