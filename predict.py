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

transform_ori = transforms.Compose(
    [
        transforms.RandomResizedCrop(64),  # create 64x64 image
        transforms.RandomHorizontalFlip(),  # flipping the image horizontally
        transforms.ToTensor(),  # convert the image to a Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)  # normalize the image


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(
            in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=8192, out_features=4000)
        self.droput = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=4000, out_features=2000)
        self.droput = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=2000, out_features=500)
        self.droput = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(in_features=500, out_features=50)
        self.droput = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(in_features=50, out_features=2)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        out = out.view(-1, 8192)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc5(out)
        return out


model = CNN()
model.load_state_dict(
    torch.load("static/model/vanilla-cnn-colored.pth", map_location=torch.device("cpu"))
)


def predict_function(img_name):
    global model

    image = cv2.imread(img_name)  # Read the image
    img = Image.fromarray(image)  # Convert the image to an array
    img = transform_ori(img)  # Apply the transformations
    img = img.view(1, 3, 64, 64)  # Add batch size
    img = Variable(img)
    # Wrap the tensor to a variable

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        img = img.cuda()

    output = model(img)
    _, predicted = torch.max(output, 1)
    print(predicted)
    if predicted == 0:
        p = "Explosive"
    else:
        p = "Non Explosive"
    return p
