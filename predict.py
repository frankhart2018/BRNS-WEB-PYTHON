import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

transform_ori = transforms.Compose([transforms.RandomResizedCrop(64),   # create 64x64 image
                                    transforms.RandomHorizontalFlip(),    # flipping the image horizontally
                                    transforms.ToTensor(),                 # convert the image to a Tensor
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # normalize the image

def predict_function(img_name, model):
    image = cv2.imread(img_name)   #Read the image
    img = Image.fromarray(image)      #Convert the image to an array
    img = transform_ori(img)     #Apply the transformations
    img = img.view(1,3,64,64)       #Add batch size
    img = Variable(img)
    #Wrap the tensor to a variable

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        img = img.cuda()

    output = model(img)
    _, predicted = torch.max(output, 1)
    print(predicted)
    if predicted==0:
        p = 'Explosive'
    else:
        p = 'Non Explosive'
    return  p
