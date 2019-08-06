# Imports here
import torch
import numpy as np
from torch import nn
from torch import optim
from PIL import Image
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
import json

import argparse


"""
--------------------------------------------------------------------------------
This Function is used to create the model based on the initial inputs
--------------------------------------------------------------------------------
"""
def create_model(hidden_units, input_size, output_size, dropout_rate, selected_model):

    print('Creating Model...')
    #Load in VGG Model by default
    if (selected_model == 'vgg13'):
        model = models.vgg13(pretrained=True) #input 25088
        input_size = 25088

    if (selected_model == 'alexnet'):
        model = models.alexnet(pretrained=True) #input 9216
        input_size = 9216


    #Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    #creates the OrderedDict to be used in the classifer creation based on initial parameters
    classifier_struct = OrderedDict([
                        ('fc0', nn.Linear(input_size, hidden_units[0])),
                        ('dropout0', nn.Dropout(dropout_rate)),
                        ('relu0', nn.ReLU())])

    for i in range(len(hidden_units)):
        if i == (len(hidden_units) - 1):
            classifier_struct[('fc' + str(i+1))] = nn.Linear(int(hidden_units[i]), output_size)
            classifier_struct['output'] = nn.LogSoftmax(dim=1)
        else:
            classifier_struct[('fc' + str(i+1))] = nn.Linear(int(hidden_units[i]),hidden_units[i+1])
            classifier_struct[('dropout' + str(i+1))] = nn.Dropout(dropout_rate)
            classifier_struct[('relu' + str(i+1))] = nn.ReLU()

    #Create replacment classifer
    classifier = nn.Sequential(classifier_struct)

    #Create the model & load in the new classifer
    model.classifier = classifier

    print('Creating Model...Finished')
    return model


"""
--------------------------------------------------------------------------------
Function for loading a checkpoint pass
--------------------------------------------------------------------------------
"""
def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)

    loading_model = create_model(checkpoint['initial_hidden_units'],
                        checkpoint['input_size'],
                        checkpoint['output_size'],
                        checkpoint['drop_rate'],
                        checkpoint['selected_model'])

    loading_model.load_state_dict(checkpoint['state_dict'])
    loading_model.class_to_idx = checkpoint['model_index']


    return loading_model



"""
--------------------------------------------------------------------------------
Function for processing a image
--------------------------------------------------------------------------------
"""
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    #opens the images based on the passed in file path
    pil_image = Image.open(image)
    image_width, image_height = pil_image.size

    #resizes the image
    size = (256, 256)
    pil_image.thumbnail(size, Image.ANTIALIAS)

    #Crops the image
    box = find_center_crop_coords(pil_image)
    pil_image = pil_image.crop(box)


    #adjusts the color channels
    np_image = np.array(pil_image)
    np_image = np_image / 256

    #normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    #switch color channel
    np_image = np.transpose(np_image, (2,0,1))

    return np_image

"""
--------------------------------------------------------------------------------
Function for finding the center of a image
--------------------------------------------------------------------------------
"""
def find_center_crop_coords(image):
    awidth, aheight = image.size
    bwidth, bheight = (224, 224)
    l = (awidth - bwidth)/2
    t = (aheight - bheight)/2
    r = (awidth + bwidth)/2
    b = (aheight + bheight)/2
    return (l,t,r,b)


"""
--------------------------------------------------------------------------------
Function for predicting a image's catagorey
--------------------------------------------------------------------------------
"""
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file

    #place model into evaluation mode
    model.eval()
    model.to(device)


    #process & prep the image
    image = process_image(image_path)

    #image = torch.from_numpy(image).float()
    image = torch.FloatTensor(image)
    image = image.unsqueeze_(0)
    image = image.to(device)

    with torch.no_grad():
        output = model.forward(image)


    output = torch.exp(output)
    probs, classes = output.topk(topk)
    class_to_idx = model.class_to_idx
    idx_to_class = {str(value):int(key) for key, value in class_to_idx.items()}

    classes = classes.cpu()
    classes = classes.numpy()[0]

    probs = probs.cpu()
    probs = probs.detach().numpy()[0]

    mapped_classes = []

    for i in range(len(classes)):
        mapped_classes.append(idx_to_class[str(classes[i])])



    return (probs, mapped_classes)

"""
--------------------------------------------------------------------------------
End Functions
--------------------------------------------------------------------------------
"""

#Create argument parser
parser = argparse.ArgumentParser(description='This is a basic Neural network creation & training script')
parser.add_argument('--image', '--i',
                    type=str,
                    metavar='',
                    default='',
                    help='Where the test is saved & the name, defaults to current directory')
parser.add_argument('--device', '--d',
                    type=str,
                    metavar='',
                    default='gpu',
                    help='Use gpu or cpu')
parser.add_argument('--checkpoint_location', '--cl',
                    type=str,
                    metavar='',
                    default='',
                    help='Where the checkpoint is saved & the name, defaults to current directory.')
parser.add_argument('--catagorey_mapping', '--cm',
                    type=str,
                    metavar='',
                    default='cat_to_name.json',
                    help='Where the json file is saved & the name, used to map the catagoreys.')
parser.add_argument('--print_top_five', '--t',
                    type=str,
                    metavar='',
                    default='n',
                    help='Print the top 5 categories from the prediction? n/y')

#load in the user provided arguments
args = parser.parse_args()
test_image = args.image
device = args.device

if device == 'gpu':
    device = 'cuda'

# Load the provided checkpoint
new_model = load_checkpoint(args.checkpoint_location)
new_model.to(device)
probs, classes = predict(test_image, new_model)

print('Prediction results for ' + args.image)

with open(args.catagorey_mapping, 'r') as f:
    cat_to_name = json.load(f)

plant_cat = []
for i in range(len(classes)):
    plant_cat.append(cat_to_name[str(classes[i])])

if(args.print_top_five == 'y'):
    for i in range(len(plant_cat)):
        print("Prediction of " + plant_cat[i] + " is estimated to be {:.1%}".format(probs[i]) + " accurate.")
else:
    print("Prediction of " + plant_cat[0] + " is estimated to be {:.1%}".format(probs[0]) + " accurate.")
