# Imports here
import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
import utilize
import model_functions as fm
import argparse

def train(args):
    print("Loading the data...")
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'
    batch_size=32
    train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    # TODO: Load the datasets with ImageFolder
    train_dataset =datasets.ImageFolder(train_dir,transform=train_transforms) 
    val_dataset =datasets.ImageFolder(valid_dir,transform=val_transforms) 
    test_dataset =datasets.ImageFolder(test_dir,transform=test_transforms) 


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    valloader= torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
    testloader= torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False);
    print("Loading  data is finished")
    print("preparing the model...")
    if(args.arch=='alexnet'):
        model=models.alexnet(pretrained=True)
        for parm in model.parameters():
            parm.requires_grid=False;
        num_features = model.classifier[1].in_features
        new_network= nn.Sequential(OrderedDict([
                              ('dropout1', nn.Dropout(p=0.5)),
                              ('fc1', nn.Linear(num_features, args.hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout2', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(args.hidden_units, args.hidden_units)),  
                              ('relu2', nn.ReLU()),
                              ('dropout3', nn.Dropout(p=0.5)),
                              ('fc3', nn.Linear(args.hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1)),
                              ]))
        model.classifier=new_network
    if(args.arch=='vgg16'):
        model=models.vgg16(pretrained=True)
        for parm in model.parameters():
            parm.requires_grid=False;
        num_features = model.classifier[0].in_features
        new_network= nn.Sequential(OrderedDict([
                              ('dropout1', nn.Dropout(p=0.5)),
                              ('fc1', nn.Linear(num_features, args.hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout2', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(args.hidden_units, args.hidden_units)),  
                              ('relu2', nn.ReLU()),
                              ('dropout3', nn.Dropout(p=0.5)),
                              ('fc3', nn.Linear(args.hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1)),
                              ]))
        model.classifier=new_network
    optimizer= optim.SGD(model.classifier.parameters(),lr=args.lr)
    criterion = nn.NLLLoss()
    device="cpu"
    if(args.gpu==True):
        device=utilize.ProcessType()
    model.to(device);
    model.class_to_idx = train_dataset.class_to_idx
    print("Preparing is finished")
    print("Start training...")
    model=fm.train_model(model,optimizer,criterion,trainloader,valloader,args.epochs,device,args.save_dir)
    model.epoch=args.epochs
    print("Training End...")
    fm.saveCheckPoint(model,"_"+args.arch+"_model_epochs"+str(args.epochs),args.save_dir)
    print("checkpoint is saved in path \ "+args.save_dir);

def main():
    parser = argparse.ArgumentParser(description='Flowers Classifcation Trainer')
    parser.add_argument('--gpu', type=bool, default=False, help='Enable/Disable GPU')
    parser.add_argument('--arch', type=str, default='alexnet', help='architecture [available: alexnet, vgg16]')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='hidden units for  fc layer')
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs')
    parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory',required=True)
    parser.add_argument('--save_dir' , type=str, default='./', help='checkpoint directory path')
    args = parser.parse_args()
    print(args)
    import json
    with open('cat_to_name.json', 'r') as f:
        flower_to_name = json.load(f)
    train(args)
    print("training Finished\n")
if __name__ == "__main__":
    main()