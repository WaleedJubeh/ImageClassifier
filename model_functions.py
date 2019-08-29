
import torch
from torch import nn
from torch import optim
from collections import OrderedDict
from PIL import Image

def train_model(model,optimizer,criterion,trainloader,valloader,epochs,device,suffex):
    #training
    best_accuracy=0
    model.batch_size=trainloader.batch_size
    model.optim_state_dict=optimizer.state_dict()
    for e in range(epochs):

        running_loss=0;
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad();
            output=model(images)
            loss=criterion(output,labels)
            running_loss+=loss.item()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model.eval()
                val_loss=0;
                accuracy=0;#save best accuracy

                for images , labels in valloader:
                    images, labels = images.to(device), labels.to(device)
                    output=model(images)
                    loss=criterion(output,labels)
                    val_loss+=loss.item()
                    output_Exp=torch.exp(output)
                    top_p,top_c = output_Exp.topk(1,dim=1)
                    equals= top_c ==labels.view(*top_c.shape)
                    accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
                model.optim_state_dict=optimizer.state_dict()
                if best_accuracy<accuracy:
                    best_accuracy=accuracy
                    model.accuracy=best_accuracy
                    model.epoch=e
                    
                    saveCheckPoint(model,"best",suffex)

                print(f"Epoch {e+1}/{epochs}.. "
                  f"Train loss: {running_loss/len(trainloader):.3f}.. "
                  f"val loss: {val_loss/len(valloader):.3f}.. "
                  f"val accuracy: {accuracy/len(valloader):.3f}")
        model.train()
    return model

def evaluate(model,criterion,loader,device):
    with torch.no_grad():
            model.eval()
            val_loss=0;
            accuracy=0;
            for images , labels in loader:
                images, labels = images.to(device), labels.to(device)
                output=model(images)
                loss=criterion(output,labels)
                val_loss+=loss.item()
                output_Exp=torch.exp(output)
                top_p,top_c = output_Exp.topk(1,dim=1)
                equals= top_c ==labels.view(*top_c.shape)
                accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"test  loss: {val_loss/len(loader):.3f}.. "
                  f"test  accuracy: {accuracy/len(loader):.3f}")
    model.train()
def saveCheckPoint(model,prefix,suffex):
        
        checkpoint = {
               'state_dict': model.state_dict(),
               'epoch': model.epoch,
               'batch_size': model.batch_size,
               'optimizer_state':model.optim_state_dict,
               'class_to_idx': model.class_to_idx,
               'output_size': 102,
               'input_size':(224,224,3),
                'accuracy':model.accuracy
             }
        torch.save(checkpoint, suffex+'checkpoint'+prefix+'.pth')
def loadCheckpoint(checkpointPath):
    import torchvision.models as models
    checkpoint =torch.load(checkpointPath, map_location=lambda storage, loc: storage)
    model=models.alexnet()
    model.classifier=nn.Sequential(OrderedDict([
                              ('dropout1', nn.Dropout(p=0.5)),
                              ('fc1', nn.Linear(checkpoint['in_features'], 512)),
                              ('relu', nn.ReLU()),
                              ('dropout2', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(512, 256)),  
                              ('relu2', nn.ReLU()),
                              ('dropout3', nn.Dropout(p=0.5)),
                              ('fc3', nn.Linear(256, checkpoint['output_size'])),
                              ('output', nn.LogSoftmax(dim=1)),
                              ]))
    model.class_to_idx=checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    model.epoch=checkpoint['epoch']
    model.optimizer_state=checkpoint['optimizer_state']
    return model