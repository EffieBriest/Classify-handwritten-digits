import torch
from torch import nn
from model import DigitRecognizitonModel
from tqdm.auto import tqdm

from errorAnalysis import accuracy_fn
from errorAnalysis import show_wrong_mnist_predictions

from helpingMethods import ask_to_save_model
from helpingMethods import load_mnist


train_loader = load_mnist(32 ,training_mode=True)
test_loader = load_mnist(32, training_mode=False)

model = DigitRecognizitonModel(10)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

epochs = 3

#Create training and testing Loop
for epoch in tqdm(range(epochs)):
    ###Training
    print(f'Epoch: {epoch}')
    train_loss = 0
    #Add a loop to loop through training batches
    model.train()
    for batch, (X,y) in enumerate(train_loader):

        #1. Forward pass
        y_pred = model(X)
        #2. Caluclate loss (per batch)
        loss = loss_fn(y_pred, y)    
        train_loss += loss #accumulatively add up the loss per  epoch
        #3. Optimizer zero grad
        optimizer.zero_grad()
        #4. loss backward
        loss.backward()
        #4. Optimizer step
        optimizer.step()
        
    #Divide total train loss by length of train dataloader (average loss per epoch)
    train_loss /= len(train_loader)

    ###Testing
    # #Setup variables for accumulatvely adding up loss and accuracy
    test_loss, test_acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X,y in test_loader:
            #1. Forward pass
            test_pred = model(X)
            #2. Calculate loss (accumulatively)
            test_loss += loss_fn(test_pred, y)
            #3. Calculate accuracy
            test_acc += accuracy_fn(y_true = y, y_pred = test_pred.argmax(dim=1))

        test_loss /= len(test_loader)

        test_acc /= len(test_loader)

    print(f'Train loss {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}:%\n')

ask_to_save_model(model)
show_wrong_mnist_predictions(model, test_loader)