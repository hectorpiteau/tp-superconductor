
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os 
import sys
from RDataset import RDataset
from MLPNet import MLPNet
import copy
import numpy as np

import wandb

USE_WANDB = True

def train_model(model, device, dataloaders, criterion, optimizer, scheduler, num_epochs=230):
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    min_test_loss = float("inf")

    for epoch in range(num_epochs):
        scheduler.step()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        print('lr : {:.8f}'.format(scheduler.get_lr()[0]))

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
                labels = labels.to(device).float().reshape((labels.shape[0],1))
                
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
                    # print("outputs: " )
                    # print(outputs.shape)
                    # print("labels: " )
                    # print(labels.shape)
                    # if epoch == 150 or epoch == 10:
                    #     print(outputs)
                    #     print(labels)
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
            if phase != "train" and epoch_loss < min_test_loss:
                min_test_loss = epoch_loss
                torch.save(model, "model.pt")
                print("Saving model..")
            
            if USE_WANDB:
                if phase == "train":
                    wandb.log({"train_loss": epoch_loss})
                else:
                    wandb.log({"test_loss": epoch_loss})
                    
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

def main():
    if USE_WANDB:
        wandb.init(project="superconductor")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    train_dataset = RDataset(mode="train")
    test_dataset = RDataset(mode="test")

    model = MLPNet(device=device).to(device).float()

    criterion = torch.nn.HuberLoss().float()

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.92)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=4, mode="triangular2", cycle_momentum=False)

    dataloaders = {
        "train":DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=1),
        "test":DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1)
    }

    model = train_model(model, device, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer, scheduler=scheduler)

    return 0

def load_and_evaluate():
    # model = MLPNet(device="cpu")

    # model_136_Adam_test_loss_459
    model = torch.load("model.pt")

    model.eval()
    print(model)

    dataset = RDataset(mode="test")

    test_data, test_labels = dataset[:100]

    test_data = torch.from_numpy(test_data).float()
    # test_data = test_data.reshape((test_data.shape[0], 1)).T
    print(test_data.shape)

    output = model(test_data).float()

    print("GT\t|\tPrediction " , output.shape, test_labels)
    for i in range(0, test_labels.shape[0]):
        print("{:.8f}\t\t\t{:.8f}\t\t{:.8f}".format(test_labels[i]*143.0, output[i][0]*143.0, abs(test_labels[i]*143.0 - output[i][0]*143.0)))

def get_min_max():
    dataset = RDataset(mode="train", percentage_in_train=1.0)
    data, labels = dataset[:]
    print(len(labels))
    print(min(labels))
    print(max(labels))
    lnorms = max(labels) - min(labels) 
    labels -= min(labels)
    labels /= lnorms
    print(np.min(labels))
    print(np.max(labels))

if __name__ == "__main__":
    sys.exit(main())
    # sys.exit(load_and_evaluate())
    # sys.exit(get_min_max())