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
Function for validation pass
--------------------------------------------------------------------------------
"""
def validation(model, validationloader, criterion):
    validation_loss = 0
    accuracy = 0



    for images, labels in validationloader:

        labels, images = labels.to(device), images.to(device)

        output = model.forward(images)
        validation_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return validation_loss, accuracy

"""
--------------------------------------------------------------------------------
End Functions
--------------------------------------------------------------------------------
"""

#Create argument parser
parser = argparse.ArgumentParser(description='This is a basic Neural network creation & training script')
parser.add_argument('--select_model', '--sel',
                    type=str,
                    metavar='',
                    default='alexnet',
                    help='Select a base model vgg13 or alexnet ( note alexnet takes less memory )')
parser.add_argument('--epochs', '--e',
                    type=int,
                    metavar='',
                    default=5,
                    help='Select the number of epochs to use in training ( int )')
parser.add_argument('--learn_rate', '--lr',
                    type=float,
                    metavar='',
                    default=0.001,
                    help='Select the learn rate ( Float )')
parser.add_argument('--hidden_units', '--hu',
                    type=int,
                    metavar='',
                    nargs='+',
                    default=[5000, 1000, 500],
                    help='Select the hidden unit layers seperated by spaces in descending order, for alexnet between 9216 & 102, for vgg13 between 25088 & 102')
parser.add_argument('--device', '--d',
                    type=str,
                    metavar='',
                    default='gpu',
                    help='Use gpu or cpu')
parser.add_argument('--save_dir', '--s',
                    type=str,
                    metavar='',
                    default='',
                    help='Where to save the checkpoint, defaults to current directory.')


#load in the user provided arguments
args = parser.parse_args()

#Default Parameters for the network
selected_model = args.select_model
input_size = 0
output_size = 102
hidden_units = args.hidden_units
dropout_rate = .01
epochs = args.epochs
learn_rate = args.learn_rate
print_every = 40
steps = 0
device = args.device

if device == 'gpu':
    device = 'cuda'


#Load in the dataset
print('Starting...')
print('Loading image data...')
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


#Setup transformations
training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

testing_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
training_data = datasets.ImageFolder(train_dir, transform=training_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
testing_data = datasets.ImageFolder(test_dir, transform=testing_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainingloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)
testingloader = torch.utils.data.DataLoader(testing_data, batch_size=32)

#Open the catagorey mapping file
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#Call the create model function
model = create_model(hidden_units, input_size, output_size, dropout_rate, selected_model)


#Train the model with the pretrained VGG model
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)


#Train the network printing out status as it progresses
print('Starting Training useing the ' + args.device)
model.to(device)

for i in range(epochs):

    model.train()
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainingloader):

        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        #forward & backward pass
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        #Print out the steps to track progress
        if steps % print_every == 0:

            model.eval()

            with torch.no_grad():
                validation_loss, accuracy = validation(model, validationloader, criterion)

            print("Epoch: {}/{}... ".format(i+1, epochs),
                  "Training Loss: {:.4f}".format(running_loss/print_every),
                  "Validation Loss: {:.3f}.. ".format(validation_loss/len(validationloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validationloader)))

            running_loss = 0

            model.train()


#Test the networks Accuracy
print('Training complete, testing accuracy...')
correct = 0
total = 0


with torch.no_grad():
    for data in testingloader:
        images, labels = data

        labels, images = labels.to(device), images.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


#Save the checkpoint
print('Saveing the model...')
model.class_to_idx = training_data.class_to_idx

checkpoint = {'input_size' : input_size,
              'output_size' : output_size,
              'selected_model' : selected_model,
              'learn_rate' : learn_rate,
              'drop_rate' : dropout_rate,
              'num_epochs' : epochs,
              'initial_hidden_units' : hidden_units,
              'optimizer_state' : optimizer.state_dict(),
              'model_index' : model.class_to_idx,
              'state_dict' : model.state_dict()}
torch.save(checkpoint, (args.save_dir + 'checkpoint.pth'))
print('Save complete')
