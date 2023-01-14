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
    parser.add_argument('image', type = str,  help = 'Path to image') 
    parser.add_argument('checkpoint', type = str,  help = 'Path to checkpoint') 
    parser.add_argument('--top_k', type = int, default = 1,  help = 'Return top k results') 
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json',  help = 'Alt labels') 
    parser.add_argument('--gpu', action ='store_true', help = 'Use GPU for inference') 
    return parser.parse_args()

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(model_path: str):
    checkpoint = torch.load(model_path)
    
    arch = checkpoint['arch']
    
        # Arch check and select
    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
        input_units = 25088
    elif arch == 'densenet161':
        model = models.densenet161(pretrained=True)
        input_units = 2208
    else:
        print(f"Error, arch \'{arch}\' not found, quitting")
        quit()   
    
  
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Model
    model.classifier = nn.Sequential(
        # Layer 1
        nn.Linear(checkpoint['input_nodes'], 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        # Layer 2
        nn.Linear(1024, checkpoint['hidden_nodes']),
        nn.ReLU(),
        nn.Dropout(0.3),
        # Output Layer
        nn.Linear(checkpoint['hidden_nodes'], checkpoint['output_nodes']),
        nn.LogSoftmax(dim=1)
    )

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image: Image.Image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    
    # Resize the image so the shorter dimension is 256 pixels
    processed_image = image.thumbnail((256,256))

    # Center crop to 224x224 pixels
    width, height = image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    
    # Reorder the dimensions
    np_image = np_image.transpose((2, 0, 1))
        
    # Convert the image to a NumPy array and return it
    return torch.tensor(np_image)

def predict(image_path: str, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file    
    # Load image file
    with Image.open(image_path) as im:
        # GPU Support
        model.to(device)
        
        image_np = process_image(im)    
        image_np = image_np.float().unsqueeze_(0)
        image_np = image_np.to(device)
        
        
        # Convert labels from ImageFolder labels back to class labels
        image_np = image_np.to(device)
        model.eval()
        with torch.no_grad():
            # Eval model
            logps = model(image_np)
            
            # Log softmax to probability
            ps = torch.exp(logps)
            
            # Get K top
            top_p, top_class = ps.topk(topk, dim=1)
            
            top_p = top_p[0].cpu().numpy()
            top_class = top_class[0]
            
            # Correct top_class mapping
            output_label_mapping = model.class_to_idx
            output_label_mapping = dict((v,k) for k,v in output_label_mapping.items())

            for k in output_label_mapping:
                output_label_mapping[k] = int(output_label_mapping[k])
            top_class = [output_label_mapping[x] for x in top_class.cpu().numpy()]
             
            return top_p, top_class

args = get_input_args()

# GPU Check
if (args.gpu):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("GPU flag enabled, but gpu not found. Defaulting to cpu")
else:
    device = torch.device("cpu")

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

model = load_checkpoint(args.checkpoint)
probs, classes = predict(args.image, model, device, args.top_k)

print(probs)
print(classes)

print(f'\n{args.image.split("/")[-1]} : Results\n-------------------------')
for i in range(len(probs)):
    print(f'{probs[i]*100:0.3f}% | Output: {classes[i]} | {cat_to_name[str(classes[i])]}')

