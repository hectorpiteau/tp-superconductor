
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os 
import sys
from RDataset import RDataset
from MLPNet import MLPNet
import copy

def train_model(model, device, dataloaders, criterion, optimizer, num_epochs=230):
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    
                    outputs = model(inputs).float()
                    loss = criterion(outputs, labels).float()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # print("before loss: ", inputs.dtype, loss.dtype)
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(outputs == labels)
            
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / len(dataloaders[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = RDataset(mode="train")
    test_dataset = RDataset(mode="test")

    model = MLPNet(device=device).to(device).float()

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.Adam()

    dataloaders = {
        "train":DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=1),
        "test":DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1)
    }

    model = train_model(model, device, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer)

    return 0



if __name__ == "__main__":
    sys.exit(main())