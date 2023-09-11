# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image


def load_checkpoint(filepath) : 
    checkpoint = torch.load(filepath)
    
    model = models.vgg16(pretrained = True)
    
    for param in model.parameters(): 
        param.requires_grad = False
        
    model.class_to_index = checkpoint['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 512)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(512, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier 
    model.load_state_dict(checkpoint['state_dict'])
    
    return model 

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    expect_means = [0.485, 0.456, 0.406]
    expect_stand = [0.229, 0.224, 0.225]
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    # First, resize the images where the shortest side is 256 pixels, 
    # keeping the aspect ratio. This can be done with the thumbnail or resize methods. 
    # Then you'll need to crop out the center 224x224 portion of the image.
    process = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(expect_means, expect_stand)
    ])
    
    image = process(img)
    
    return image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
#     TODO: Implement the code to predict the class from an image file  
    model.class_to_idx = image_datasets['training'].class_to_idx
    
    image = process_image(image_path) 
    image = torch.unsqueeze(image, 0)

    image = image.float()
    
    probs, classes = torch.exp(model.forward(image)).topk(topk)
    
    idx_to_class = {} 
    
    for key, value in model.class_to_idx.items(): 
        idx_to_class[value] = key 
    
    np_classes = classes[0].numpy()
    
    top_classes = []
    
    for label in np_classes: 
        top_classes.append(int(idx_to_class[label]))
    
    return probs[0].tolist(), top_classes

# TODO: Display an image along with the top 5 classes
def display(image_path): 
    probs, classes = predict(image_path, model)
    processed_image = process_image(image_path)
    
    five_names = []

    for i in range(len(classes)): 
        five_names.append(cat_to_name[str(classes[i])])
        
    plt.figure(figsize = (5, 10))
    ax = plt.subplot(2, 1, 1)
    
    title = five_names[0]
    plt.title(title)
    imshow(processed_image, ax = ax)
    
    plt.subplot(2, 1, 2)
    sb.barplot(x = probs, y = five_names, color = sb.color_palette()[0])
    plt.show()
    


model = load_checkpoint('checkpoint.pth')
image_path = test_dir + '/28/image_05230.jpg'

display(image_path)