#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:58:32 2022

@Author:     Alessio Borgi
@Contact :   borgi.1952442@studenti.uniroma1.it

@Filename:   Spam_Detector.py
"""

#Importing the several libraries we will use.
from cgi import test
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset                                #Dataset stores the samples and their corresponding labels.
from torchvision.transforms import ToTensor
import pandas as pd
import torch.utils.data as data_utils
import numpy as np


class MyDataset(Dataset):
    def __init__(self, email_file, batch_size = 8):   
        #Labels are stored in a csv file, each line has the form namefile,label - Ex:img1.png, dog; transforms is a list of transformations we need to apply to the data.
        x = email_file['Email'].tolist()
        
        #Mapping Labels to code.
        email_file['Label'] = email_file.Label.map({'ham':0, 'spam':1})
        y = email_file['Label'].tolist()
        res = 0
        for i in y:
            #print(y)
            if i == 0:
                res += 1
        #print(f"Num Ham is: {res} Num spam: is: {len(y) - res}")
        # print(type(y))
        self.label = torch.tensor(y, dtype = torch.int8)
        # self.label = y

        # print(x)
        
        #Mapping String Samples to int.
        # print(len(x))
        max_l = 0
        ts_list = []
        for w in x:
            phrase = ''
            for char in w:
                if char != ' ':
                    phrase += char
            # print(len(phrase), len(w))
            ts_list.append(torch.ByteTensor(list(bytes(phrase, 'utf8'))))
            max_l = max(ts_list[-1].size()[0], max_l)
        w_t = torch.zeros((len(ts_list), max_l), dtype=torch.uint8)
        for i, ts in enumerate(ts_list):
            w_t[i, 0:ts.size()[0]] = ts        
        #print(w_t.size()[1])
        print(f"The size is:{w_t.size()}")
        w_t = torch.unsqueeze(w_t, dim=0)
        print(f"The size is:{w_t.size()}")
        size = list(w_t.size())
        # print(type(size))
        self.email = torch.tensor(w_t).view(batch_size,1,size[1],size[2])  
        #self.email = w_t
        
        
    def __len__(self):
        return len(self.label)                                         #Return the length of the labels. 
    
    def __getitem__(self, idx):             
        #Getting the label.
        return self.email[idx], self.label[idx]                                               #Return the image with the corresponding label.

#Downloading Step:
#Getting a Dataset containing small images of clothes(t-shirt, shoes etc...). 
df = pd.read_csv('Dataset_SpamHam.csv')
training_size = int(0.7 * len(df))
#print("The training size is:", training_size)
training_data = df.iloc[:training_size,:]
# print(len(training_data))
test_data = df.iloc[training_size:,:]
#print(type(test_data))

df_train = MyDataset(training_data, len(training_data))
df_test = MyDataset(test_data, len(test_data))
#print(df_train)

#Get the elements.
#Creation of a DataLoader that allows us to create an Iterable over the Dataset. Note that we need to indicate a Batch_Size. Indeed, in most cases, you don't have
#the capability to save the whole data, and thus you make use of Batches. The more the batch_size is bigger, the faster is the epoque(iteration through the whole 
# dataset). However, a too large batch size could cause the break of our training step.

train_dataloader = DataLoader(df_train, batch_size = 8, shuffle = True)
test_dataloader = DataLoader(df_test, batch_size = 8, shuffle = True)

#Define the device for the computation.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Define our Multilayer Perceptron(MLP).
class MLP(nn.Module):                       #We extend the "nn.Module" package if we want to create an MLP class.
    #Definition of all the things we want to use in the "__init__". In this case, we create the model using the Sequential Approach. 
    def __init__(self):
        super(MLP,self).__init__()
        self.flatten=nn.Flatten()           #It takes a multidimensional tensor and converts to a monodimentional tensor. Since we are using images we can consider 
                                            #the values of the pixels as features. Our dataset is a two dimentional (height and weidth), so we flatten it.
        self.conv1 = nn.Conv2d(1, 6, 3)             #Starting Layer of the CNN. Note that we have to give the input channel, output channels and the Kernel Size.
        self.conv2 = nn.Conv2d(6, 10, 3)            #The second Layer of our CNN will have to take in input channels the output channel of the prevoius layer.
        
        self.fc1 = nn.Linear(3900*740*2, 20)          #Defining the Fully-Connected layer.
        self.fc2 = nn.Linear(20, 15)
        self.fc3 = nn.Linear(15, 2)                #Final Layer, outputting 36 corresponding to number of classes-1.
        self.relu = nn.Mish()                       #We have used Mish as         
    #This function allows to establish th reorder in which layers should be used.
    def forward(self,x):                    
        #First Convolution.
        x = self.conv1(x)
        x = self.relu(x)                            #Applying the Activation Function.
            
        #Second Convolution.
        x = self.conv2(x)
        x = self.relu(x)                            #Applying the Activation Function. 
            
        #Fully Connected
        x = torch.flatten(x, 1)                     #Flatten all dimensions except the batch. We have to put "1" beacuse the input data is a tensor of 4 elements
                                                    #[batch,chanels, width, height]. Flatten the data starting from index one.
        #FC1
        x = self.fc1(x)
        x = self.relu(x)                            #Applying the Activation Function.
            
        #FC2
        x = self.fc2(x)
        x = self.relu(x)                            #Applying the Activation Function.
            
        #FC out
        x = self.fc3(x)
        #We don't need to use the mish. It will provide an array that has the same length of the classes.
            
        return x                     


def train_and_test():
        #The train_and_test function is made of two inner functions, defining respectively the Training and Testing Loop.
        def trainingLoop(train_dataloader, model, loss_fn, optimizer):
            '''Training Loop Function'''
        
            for batch, (X,y) in enumerate(train_dataloader):
                pred= model(X)                                              #Getting the prediction.
                loss = loss_fn(pred,y)                                      #Distance with the batch containing the input image.

                #Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch%100==0:                                            #Every 100 batches, we store our reconstruction.
                    loss= loss.item()
                    print(f'The Current Loss is: {loss}')
    
        def testLoop(test_dataloader, model, loss_fn):
            '''Test Loop Function'''
            print_size = len(test_dataloader.dataset)                                 
            num_batches = len(test_dataloader)                                         #It is the lenght of the test_dataloader. 
            test_loss = 0                                                              #Intialize the test loss to zero. 
            correct = 0                                                                #Iniziatile the number of correct labels to zero.
            
            with torch.no_grad():                                                      #Do not modify the weights of the model.
                for X,y in test_dataloader:
                    X, y = X.to(device), y.to(device)
                    
                    pred = model(X)                                                    #Getting the prediction.
                    test_loss += loss_fn(pred, y).item()                               #With the ".item", we just get the value of the loss.
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    
                test_loss = test_loss / num_batches                                    #The test loss is defined as the value of the test loss divided by the number of 
                                                                                       #batches. 
                correct = correct / print_size                                         #Correct is defined as the value of correct divided by print_size.
                print(f"The Current Accuracy is: {correct * 100}, with the Average Loss being:{test_loss}") 
        
        #instance of our model
        model=MLP().to(device)

        #hyperparameter settings
        learning_rate= 1e-3
        epochs = 3

        #loss function definition
        loss_fn =nn.CrossEntropyLoss()

        #optimizer definition
        optimizer = torch.optim.Adam(model.parameters(),learning_rate)  
        
                
        #model = torch.load('Models/CNN_7_Version_92.194.pt')                           #At this point, we load the model. This acts as a sort of weight, allowing us to start 
                                                                                        #doing the training and testing from a better accuracy. Thanks to it, we were able to 
                                                                                        #reach an accuracy near to 96% .. It was a big improvement, but not all the license 
                                                                                        #plates were recognized correctly. Reaching a higher accuracy may cause overfitting 
                                                                                        #problems. 
        for e in range(epochs):                                                         #For each epoch. 
            trainingLoop(train_dataloader,model,loss_fn,optimizer)                      #Call the Training Loop. 
            testLoop(test_dataloader, model, loss_fn)                                   #Call the Test Loop. 
        torch.save(model,'Models/MLP.pt')                                               #Save the model. 
        return model
    
    
    
if __name__ == "__main__":
    train_and_test()