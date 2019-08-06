"""
--------------------------------------------------------------------------------
This Function is used to create the model based on the initial inputs
--------------------------------------------------------------------------------
"""
def create_model(hidden_units, input_size, output_size, dropout_rate, selected_model):

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

        labels, images = labels.to('cuda'), images.to('cuda')

        output = model.forward(images)
        validation_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return validation_loss, accuracy


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


    #process & prep the image
    image = process_image(image_path)

    #image = torch.from_numpy(image).float()
    image = torch.FloatTensor(image)
    image = image.unsqueeze_(0)
    #image = image.to('cuda')

    with torch.no_grad():
        output = model.forward(image)


    output = torch.exp(output)
    probs, classes = output.topk(topk)

    probs = probs.cpu()
    probs = probs.detach().numpy()[0]

    classes = classes.cpu()
    classes = classes.numpy()[0]
    classes = [model.class_to_idx[str(i)] for i in classes]



    return (probs, classes)
