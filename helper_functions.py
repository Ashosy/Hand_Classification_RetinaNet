import os
import pandas as pd
import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_model(model, dataloaders, criterion, optimizer, num_epochs, is_inception=False):
    since = time.time()


    for epoch in range(num_epochs):
        losses = []
        for batch_idx, (data, targets) in enumerate(dataloaders):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            losses.append(loss.item())
            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses):.5f}")



def check_accuracy(loader, model):
    
    print("Checking accuracy on training data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()
    


   