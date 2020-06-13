import argparse # needed to parse command line arguments
from time import time
from matplotlib import pyplot as plt # for graphs
import numpy as np
import json # for the labels saved in a json file
import torch # for PyTorch
from torch import nn # to make it easier making sequential models
from torch import optim # for the optimization method
from torchvision import datasets, transforms, models # to load the data and models
from PIL import Image # for image file further below


def get_args():
    """
    Obtain the command line arguments and parse them to run the program
    """
    parser = argparse.ArgumentParser()

    # data_dir = image data directory
    # --save_dir = saving the directory
    #

    # defaults of argparse, unless stated otherwise:
    #   action = 'store'
    #   type = str (strings)

    parser.add_argument('--data_dir', default = 'flowers/', help = 'Location of the images data folder, that contains the "train", "valid" and "test" subforlders')
    parser.add_argument('--predict', default = 'flowers/train/73/image_00287.jpg', help = 'Location of image to predict the label of')
    parser.add_argument('--train', action = 'store_true', default = False, help = 'Places the model into training mode')
    parser.add_argument('--device', default = 'gpu', help = 'CUDA (Compute Unified Device Architecture) is used to quickly train the network; by default all training calculations are on the GPU')
    parser.add_argument('--lr', type = int, default = 1, help = 'Defines the training learning rate')
    parser.add_argument('--epochs', type = int, default = 1, help = 'Defines the number of training cycles')
    parser.add_argument('--checkpoint', default = 'checkpoint.pth', help = 'File name to store the checkpoint of the network model')
    # parser.add_argument('--arch', default = 'vgg19', help = 'Defines the model Architecture' ) # I'll look into implementing other models and varied nodes at another time
    
    args = parser.parse_args()

    return args
    

def set_dir(dir):
    directory = {'train':dir + 'train', # 8189 files
                 'valid':dir + 'valid', # 818 files
                 'test' :dir + 'test'} # 8189 files

    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                       'valid': transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                       'test' : transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                      }

    # TODO: Load the datasets with ImageFolder
    image_datasets = { data : datasets.ImageFolder(directory[data], transform=data_transforms[data]) for data in ['train', 'valid', 'test']}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    data_loaders = { data: torch.utils.data.DataLoader(image_datasets[data], batch_size=50, shuffle = True) for data in ['train','valid','test']}

    return data_loaders


def build_model():
    model = models.vgg19(pretrained=True) # VGG-19 model
    for values in model.parameters():
        values.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Linear(25088,1020), # Hidden layer 1
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(1020,510), # Hidden layer 2
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(510,102),  # Output of 102 choices
                                     nn.LogSoftmax(dim=1))
    return model

def modelstate(model, data_loaders, state='valid', device='cpu', epochs=1, lr = 0.03):
    '''Loading the model within the 3 different states'''
    
    model.to(device) # moving the model to the GPU or CPU
    criterion = nn.NLLLoss() # Negative Log Likelihood Loss is best when the output is a a log-softmax function
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    if state == 'train':
        train_losses = []
        valid_losses = []
        accuracy_list = []
        spotcheck = 20
        index = 0
        trainloss = 0
        
        for e in range(epochs):
            step = 0
            for images, labels in data_loaders[state]:
                images, labels = images.to(device),labels.to(device) # moving the images and labels to the GPU or CPU, whichever is available
        
                optimizer.zero_grad()
        
                output = model(images)
        
                loss = criterion(output, labels)

                loss.backward()
        
                optimizer.step()
        
                trainloss += loss.item()

                step += 1
        
                if step % spotcheck == 0: # after every 10 steps we can view the progress being made
                    
                    validloss, accuracy  = modelstate(model, data_loaders, 'valid', device, epochs)
                    
                    train_losses.append(trainloss/spotcheck) # since we are spotchecking and not running through the whole trainloader before printing the below, we need to gather the average at that point in time
                    valid_losses.append(validloss)
                    accuracy_list.append(accuracy)
                    
                    print(f'Epoch = {e+1} of {epochs} | Spot Check ={(index+1):3.0f} | Training Loss = {train_losses[index]:6.3f} | Validation Loss = {valid_losses[index]:6.3f} | Accuracy = {((accuracy_list[index])*100):6.3f}%')                                             
                    
                    index += 1   
                    trainloss = 0 # resets the training loss so an accurate value can be obtained per spot check
        
        return model
    
    if state == 'valid':    
                
        model.eval() # switching to evaluation mode
        validloss = 0
        accuracy = 0
        
        with torch.no_grad(): # ensuring we don't receive the gradients during this spot check
            for valid_images, valid_labels in data_loaders[state]:
                valid_images, valid_labels = valid_images.to(device), valid_labels.to(device) # moving these to the GPU or CPU also
                    
                output = model(valid_images) # forward pass
                    
                ps = torch.exp(output) # obtain probabilites by passing the outputs through an exponential
                    
                topprob, topclass = ps.topk(1, dim=1) # first largest value in our probabilites
                    
                equals = topclass == valid_labels.view(*topclass.shape) # I had to edit the labels batch size to 18 so that it can be transformed
                    
                validloss += criterion(output, valid_labels) # sum of validation loss during this spot check
                    
                accuracy += torch.mean(equals.type(torch.FloatTensor)) # accuracy during this spot check
        
        validloss = validloss/len(data_loaders[state])
        accuracy = accuracy/len(data_loaders[state])
        
        model.train() # switching back to training mode
        
        return validloss, accuracy   
    
    if state == 'test':
        
        model.eval() # switching to evaluation mode
        
        testloss = 0
        accuracytest = 0
        
        for test_images, test_labels in data_loaders[state]:
            test_images, test_labels = test_images.to(device), test_labels.to(device) # moving these to the GPU or CPU also
                            
            output = model(test_images)
                            
            ps = torch.exp(output)
                            
            topprob, topclass = ps.topk(1, dim=1)
                            
            equals = topclass == test_labels.view(*topclass.shape)
                            
            testloss += criterion(output, test_labels)
                            
            accuracytest += torch.mean(equals.type(torch.FloatTensor))
        
        testloss = testloss / len(data_loaders[state])
        accuracytest = (accuracytest / data_loaders[state]) * 100
        
        model.train()
        
        return testloss, accuracytest

def validation(model):
    
    testloss, accuracytest = modelstate(model, data_loaders, 'valid')
    
    print(f'Test Loss = {test_loss:6.3f} | Accuracy = {(accuracy_test):6.3f}%')

    return None

def save_checkpoint(model, checkpoint):
    model.to('cpu')
    model.class_to_idx = image_datasets['train'].class_to_idx
    torch.save({'input': 25088,
                'hidden': (1020,510),
                'output': 102,
                'arch': 'vgg19',
                'classifier': model.classifier,
                'state_dict': model.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                # 'epochs':epochs,
                'class_to_idx':model.class_to_idx
                }, checkpoint)
    return None

def loading(path):
    
    checkpoint = torch.load(path)
    model2use = getattr(models, checkpoint['arch'])(pretrained=True) # get's the value of the objects attribute and assigns it to model2use
    model2use.classifier = checkpoint['classifier']
    model2use.load_state_dict(checkpoint['state_dict'])
    # epochs = checkpoint['epochs']
    # optimizer = optimizer.load_state_dict(checkpoint['optimizer'])
    model2use.class_to_idx = checkpoint['class_to_idx']
    
    return model2use

def process_image(pic):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(pic).resize((256,256)).crop((6,6,250,250)) # this would crop it by 244 from the centre
    # left, upper, right, lower - order of co-ordinates form the top left pixel as 0,0 | therefore the right can be thought of as the width and the lower can be thought of as the height
    
    mean = np.array([0.485, 0.456, 0.406])
    
    std = np.array([0.229, 0.224, 0.225])
    
    img = np.array(pil_image)/255 # since there are 0-255 pixel values, I would want to obtain the values for this array
    
    np_image = torch.from_numpy(((img - mean) / std).transpose((2, 0, 1))).type(torch.FloatTensor) # changing it from (c,a,b) to (a,b,c) | turning it from a numpy array to a torch tensor
    
    return np_image

def predict(image_path, modelpath, topk=5, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file    
    with torch.no_grad():
        image = process_image(image_path)
    
        image = torch.unsqueeze(image,0) # so that it shows a 4D tensor input instead of a 3D one | this will show as a batch of size 1 now
    
        # print('img', image.shape) # confirming shape is now 4D
        
        modelpredict = loading(modelpath)
    
        image = image.to(device)
        modelpredict = modelpredict.to(device)
    
        modelpredict.eval() # to avoid changing it's state
         
        output = modelpredict(image)
                    
        ps = torch.exp(output)
                    
        prob5, class5 = ps.topk(5, dim = 1)
    
    return prob5, class5

def confirm(imagepath, modelpath, device='cpu'):
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    probs, classes = predict(imagepath, modelpath, device)
    classes = classes[0].numpy()
    probs = probs[0].numpy()
    
    topprob_i = np.argmax(probs)
    topprob = probs[topprob_i]
    topclass = classes[topprob_i]

    fig = plt.figure(figsize = (10,10))
    plot1 = plt.subplot2grid((15,9), (0,0), colspan = 9, rowspan = 9)
    plot2 = plt.subplot2grid((15,9), (9,2), colspan = 6, rowspan = 9)


    #top plot
    image = Image.open(imagepath)
    plot1.set_title(cat_to_name[str(topclass)])
    plot1.imshow(image)
    plot1.axis('off') # removes the axis from shwoing

    classtypes = []
    for classtype in classes:
        classtypes.append(cat_to_name[str(classtype)])  # converting to string so that the key can be read

    # bottom plot
    y = np.arange(5)
    plot2.set_yticks(y)
    plot2.set_yticklabels(classtypes)
    plot2.invert_yaxis()
    plot2.barh(y, probs, align='center')

    plt.show()

    return None

def main():
    start_time = time()    
    
    args = get_args()
   
    if args.train:
        data_loaders = set_dir(args.data_dir)
        model = build_model()
        model = modelstate(model, data_loaders, 'train', args.device, args.epochs, args.lr)
        save_checkpoint(model, args.checkpoint)
    
    loaded_model = loading(args.checkpoint)
    
    confirm(args.predict, args.checkpoint, args.device)
    
    end_time = time()
    
    tot_time = end_time-start_time #calculate difference between end time and start time
    hours = int(tot_time / 3600) # 3600 seconds in an hour
    hourless = tot_time % 3600
    minutes = int(hourless / 60) # 60 seconds in a minute
    seconds = round(hourless % 60) # rounding to the nearest second
    print("\n** Total Runtime = {}h:{}m:{}s".format(hours,minutes,seconds))


main()