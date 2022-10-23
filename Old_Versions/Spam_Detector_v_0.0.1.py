#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:58:32 2022

@Author:     Alessio Borgi
@Contact :   borgi.1952442@studenti.uniroma1.it

@Filename:   Spam_Detector.py
"""

#Importing the several libraries we will use.
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
    def __init__(self, csv_path):   
        #Labels are stored in a csv file, each line has the form namefile,label - Ex:img1.png, dog; transforms is a list of transformations we need to apply to the data.
        email_file = pd.read_csv(csv_path)
        x = email_file['Email']
        email_file['Label'] = email_file.Label.map({'ham':0, 'spam':1})
        y = email_file['Label']
        print(y)
                
        self.email = torch.tensor(x)
        self.label = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.label)                                         #Return the length of the labels. 
    
    def __getitem__(self, idx):             
        #Getting the label.
        return self.email[idx], self.label[idx]                                               #Return the image with the corresponding label.

#Downloading Step:
#Getting a Dataset containing small images of clothes(t-shirt, shoes etc...). 
df = pd.read_csv('Dataset_SpamHam.csv')
mydataset = MyDataset(csv_path = "Dataset_SpamHam.csv")

# training_data = datasets.FashionMNIST(root = "data", train = True, download = True, transform = ToTensor())       
# test_data = datasets.FashionMNIST(root = "data", train = False, download = True, transform = ToTensor())

#Get the elements.
#Creation of a DataLoader that allows us to create an Iterable over the Dataset. Note that we need to indicate a Batch_Size. Indeed, in most cases, you don't have
#the capability to save the whole data, and thus you make use of Batches. The more the batch_size is bigger, the faster is the epoque(iteration through the whole 
# dataset). However, a too large batch size could cause the break of our training step.

#train_dataloader = DataLoader(training_data, batch_size = 8, shuffle = True)
test_dataloader = DataLoader(mydataset, batch_size = 8, shuffle = True)

#Define the device for the computation.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Define our Multilayer Perceptron(MLP).
class MLP(nn.Module):                       #We extend the "nn.Module" package if we want to create an MLP class.
    #Definition of all the things we want to use in the "__init__". In this case, we create the model using the Sequential Approach. 
    def __init__(self):
        super(MLP,self).__init__()
        self.flatten=nn.Flatten()           #It takes a multidimensional tensor and converts to a monodimentional tensor. Since we are using images we can consider 
                                            #the values of the pixels as features. Our dataset is a two dimentional (height and weidth), so we flatten it.

        #Creation of the Sequential Approach.
        self.ann = nn.Sequential(           #It puts sequentially the things we pass as arguments, ex: connects layers.
                nn.Linear(28*28,512),       #It creates a layer specifying input an output number of nodes. Note that the input number of nodes correspond to the 
                                            #size of the flattened.
                nn.ReLU(),                  #Declaration of the Activation Function. It computes the maximum of the values. We have that all the values from 0 to 
                                            #+inf remains the same, whilst all the values that are negatives are set to 0.
                nn.Linear(512,512),         #We add up another Hidden Layer. Note that the input size that the second layer takes is the output size of the previous
                                            #Hidden Layer.
                nn.ReLU(),                  #We specify its Activation Function.
                nn.Linear(512,10)           #Last Hidden Layer, that allows to apply Classification. Here we have that the output size is 10, conciding to the number of classes.       
        )
    #This function allows to establish th eorder in which layers should be used.
    def forward(self,x):                    
        x=self.flatten(x)                   #We flatten the image.
        logits = self.ann(x)                #Here we pass our image to our model.
        return logits                       #we return the value.


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
        
        train_dataloader = DataLoader(training_data, batch_size = 8, shuffle = True)   #DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
        test_dataloader = DataLoader(test_data, batch_size = 8, shuffle = True)     #The batch size takes 8 samples at each iteration,until the entire dataset has been seen.
        
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