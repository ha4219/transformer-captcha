import os
import glob
import math

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from sklearn.model_selection import train_test_split
from torch.autograd import Variable

from models.transformer_captcha import Model
from utils.dataloaders import create_dataloader

from conf import *

all_path = glob.glob(path)
train_path, test_path = train_test_split(all_path)

train_loader, _ = create_dataloader(
    path=train_path,
    image_size=image_size,
    vocab_size=vocab_size,
    augment=True,
    batch_size=32,
)

test_loader, _ = create_dataloader(
    path=test_path,
    image_size=image_size,
    vocab_size=vocab_size,
    augment=True,
    batch_size=32,
)


model = Model(
    image_size=image_size,
    patch_size=patch_size,
    d_model=d_model,
    n_head=n_head,
    n_layers=n_layers,
    vocab_size=vocab_size,
)

# lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
# lrf: 0.1  # final OneCycleLR learning rate (lr0 * lrf)

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

epochs = 1000
lr0 = 5e-5
lrf = 5e-4
lf = one_cycle(1, lrf, epochs)
optimizer = torch.optim.Adam(model.parameters(), lr=lr0)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
loss_fn = torch.nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to save the model
def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels, text = data
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            # run the model on the test set to predict labels
            outputs = model(images, labels)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, -1)
            _, labels = torch.max(labels, -1)
            # print(predicted[:,6], labels[:,:6])
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item() / labels.size(1)
    
    # compute the accuracy over all test images
    accuracy = (accuracy / total)
    return(accuracy)


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    
    best_accuracy = 0.0
    scheduler.last_epoch = num_epochs - 1
    # Define your execution device
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)
    nb = len(train_loader)

    train_loss = []
    train_acc = []
    test_acc = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0
        total = 0

        model.train()

        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}')
        
        for i, (images, labels, text) in pbar:
            
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            total += 1

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images, labels)
            # print(outputs.shape)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            _, predicted = torch.max(outputs.data, -1)
            _, labels = torch.max(labels, -1)
            # print(predicted[:,6], labels[:,:6])
            running_acc += (predicted == labels).sum().item() / labels.size(1)

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value

            scheduler.step()
            if i % 1000 == 999:    
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0
                running_acc = 0.0
            pbar.set_description(f'train_loss: {running_loss / total}, train_acc: {running_acc / total}')
        
        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
        
        train_loss.append(running_loss / nb)
        train_acc.append(running_acc / nb)
        test_acc.append(accuracy)

        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

    from utils.plot import train_test_loss_acc_plot
    train_test_loss_acc_plot(train_loss, train_acc, test_acc)

if __name__ == '__main__':
    train(epochs)