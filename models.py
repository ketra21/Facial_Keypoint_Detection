    
## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        # input 1x224x224
        self.conv1 = nn.Conv2d(1, 32, 5) # 32×220×220
        self.pool1 = nn.MaxPool2d(4, 4) #32×55×55
        self.dropout1 = nn.Dropout(p=0.1)
        
        self.conv2 = nn.Conv2d(32, 64, 4) #64×52×52
        self.pool2 = nn.MaxPool2d(2, 2) #64×26×26
        self.dropout2 = nn.Dropout(p=0.2)
        
        self.conv3 = nn.Conv2d(64,128, 3) #128×24×24
        self.pool3 = nn.MaxPool2d(2, 2) #128×12×12
        self.dropout3 = nn.Dropout(p=0.3)
        
        self.conv4 = nn.Conv2d(128, 256, 2) #256×11×11
        self.pool4 = nn.MaxPool2d(2, 2) #256×5×5
        self.dropout4 = nn.Dropout(p=0.4)
        
        self.fc1 = nn.Linear(256*5*5, 1000) #1000
        self.fc1_drop = nn.Dropout(p=0.5)
                
        self.fc2 = nn.Linear(1000, 1000)  #1000
        self.fc2_drop = nn.Dropout(p=0.6)
        
        self.fc3 = nn.Linear(1000, 136) #136
         
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        
        
        x = self.dropout4(self.pool1(F.relu(self.conv1(x))))
        x = self.dropout4(self.pool2(F.relu(self.conv2(x))))
        x = self.dropout4(self.pool3(F.relu(self.conv3(x))))
        x = self.dropout4(self.pool4(F.relu(self.conv4(x))))
        
        x = x.view(x.size(0), -1) 
        
        x = self.fc1_drop(F.relu(self.fc1(x)))
        x = self.fc2_drop(self.fc2(x))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
