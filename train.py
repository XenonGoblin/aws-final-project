import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
import argparse

def get_input_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type = str,  help = 'Path to training images') 
    parser.add_argument('--save_dir', type = str,  help = 'Path to save checkpoints') 

    parser.add_argument('--arch', type = str, default = 'densenet161', help = 'Model Architecture (densenet161|vgg11)') 
    
    parser.add_argument('--learning_rate', type = float, default = 0.003,  help = 'Learning Rate') 
    parser.add_argument('--hidden_units', type = int, default = 512,  help = 'Hidden Units') 
    parser.add_argument('--epochs', type = int, default = 3,  help = 'Epochs') 
    
    parser.add_argument('--gpu', action ='store_true', help = 'Use GPU for training') 
    
    return parser.parse_args()


def run_validation(model, data_loader: torch.utils.data.DataLoader):
    test_loss = 0
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            # Get top prediction
            top_p, top_class = ps.topk(1, dim=1)
            # Number of correct matches
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Loss: {test_loss/len(valid_loader):.3f} | Accuracy: {accuracy/len(valid_loader)*100:.3f}%")


def save_model(model, optimizer, data, name, input_nodes=2208, output_nodes=102, hidden_layers=512, arch='densenet161'):
    checkpoint = {
            'input_nodes': input_nodes,
            'output_nodes': output_nodes,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict(),
            'class_to_idx': data.class_to_idx,
            'hidden_nodes': hidden_layers,
            'arch': arch
        }
    torch.save(checkpoint, name)
    
    
        
args = get_input_args()

# GPU Check
if (args.gpu):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU Enabled")
    else:
        device = torch.device("cpu")
        print("GPU flag enabled, but GPU not found. Defaulting to CPU")
else:
    device = torch.device("cpu")

    
train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'
test_dir = args.data_dir + '/test'


# Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([
    transforms.RandomRotation(40),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(255),                                  
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])
])


train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=1)



input_units = 2208
output_units = 102
hidden_units = args.hidden_units

# Arch check and select
if args.arch == 'vgg11':
    model = models.vgg11(pretrained=True)
    input_units = 25088
elif args.arch == 'densenet161':
    model = models.densenet161(pretrained=True)
    input_units = 2208
else:
    print(f"Error, arch \'{args.arch}\' not found, quitting")
    quit()    

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False


# Model
model.classifier = nn.Sequential(
    # Layer 1
    nn.Linear(input_units, 1024),
    nn.ReLU(),
    nn.Dropout(0.3),
    # Layer 2
    nn.Linear(1024, hidden_units),
    nn.ReLU(),
    nn.Dropout(0.3),
    # Output Layer
    nn.Linear(hidden_units, output_units),
    nn.LogSoftmax(dim=1)
)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
model.to(device);



# Training model
epochs = args.epochs
steps = 0
test_every = 10

print(f'\nTraining | e={epochs} | lr={args.learning_rate} | in={input_units} | hidden={hidden_units} | out={output_units} | arch={args.arch}')

for epoch in range(epochs):
    for inputs, labels in train_loader:
        # Move to CPU or GPU
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        # Feedforward
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        # Calculate gradients
        loss.backward()
        # Backpropagate
        optimizer.step()
        
        steps += 1
        if steps % test_every == 0:
            print(f'Epoch: ({epoch}/{epochs}) | ', end='')
            model.eval()
            run_validation(model, train_loader)
            model.train()

print(f'\nTraining Complete, validation testing')
# Do validation on the test set
run_validation(model, valid_loader)

if args.save_dir is not None:
    print(f"Saving model to {args.save_dir}")
    save_model(model, optimizer, train_data, args.save_dir+f'/checkpoint_{args.arch}.pth', input_units, output_units, args.hidden_units, args.arch)
    
    
    
    
    
    
    
    





