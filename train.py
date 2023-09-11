
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse

from PIL import Image

def data_info(args): 
    data_dir = args.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
            "training" : transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(), 
                transforms.RandomRotation(30), 
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "validation" : transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "testing" : transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        "training" : datasets.ImageFolder(train_dir, transform = data_transforms["training"]),
        "validation" : datasets.ImageFolder(valid_dir, transform = data_transforms["validation"]),
        "testing" : datasets.ImageFolder(test_dir, transform = data_transforms["testing"])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        "training" : torch.utils.data.DataLoader(image_datasets["training"], batch_size = 16, shuffle = True),
        "validation" : torch.utils.data.DataLoader(image_datasets["validation"], batch_size = 16, shuffle = True), 
        "testing" : torch.utils.data.DataLoader(image_datasets["testing"], batch_size = 16, shuffle = True)
    }    
    
    return dataloaders['training'], dataloaders['validation'], image_datasets['training']

def train(trainloader, validloader, traindataset):
    model = models.vgg16(pretrained = True)
    
    epochs = 10
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout Feedforward Classifier
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 4096)),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(4096, 512)),
                              ('relu2', nn.ReLU()),
                              ('fc3', nn.Linear(512, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier 
    
    print_every = 40
    running_loss = 0
    steps = 0
     
    model.to(device)
    
    for e in range(epochs):
        model.train()
        
        for inputs, labels in iter(trainloader):
            steps += 1

            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)
                
            else:
                inputs, labels = inputs, labels
                
            optimizer.zero_grad()
            # front
            output = model.forward(inputs)
            # calculating the loss
            loss = criterion(output, labels)
            # back
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
                
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0 
                model.eval()
                
                with torch.no_grad(): 
                    test_loss, accuracy = validation(model, validloader, criterion)
                    
                print("Epoch: {}/{} ".format(e+1, epochs),
                        "Training Loss: {:.3f} ".format(running_loss/len(trainloader)),
                        "Validation Loss: {:.3f} ".format(test_loss/len(validloader)),
                        "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                running_loss = 0
    save_checkpoint(model, traindataset, optimizer, classifier) 
                
def validation(model, validloader, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    accuracy = 0
    test_loss = 0
    for inputs, labels in validloader:
        if torch.cuda.is_available():
            inputs, labels = inputs.to(device), labels.to(device)
        else:
            inputs, labels = inputs, labels
            
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
        
        test_loss += batch_loss.item()
        
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return test_loss, accuracy

def save_checkpoint(model, traindataset, optimizer, classifier) :
    model.class_to_idx = traindataset.class_to_idx
    model.cpu()
    model_state = {
        'epoch': 10,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'classifier': classifier,
        'class_to_idx': model.class_to_idx,
    }
    
    torch.save(model_state, "checkpoint.pth")
    print("file saved")

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    
    parser.add_argument(dest='data_directory')
    
    args = parser.parse_args()
    
    train_dataloaders, valid_dataloaders = data_info(args)
    
    train(train_dataloaders, valid_dataloaders)