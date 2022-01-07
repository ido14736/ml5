from gcommand_loader import GCommandLoader
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import math
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainDataset = GCommandLoader('C:/Users/magshimim/PycharmProjects/ml5/data/train')
validDataset = GCommandLoader('C:/Users/magshimim/PycharmProjects/ml5/data/valid')
testDataset = GCommandLoader('C:/Users/magshimim/PycharmProjects/ml5/data/test')

#TODO: fix to the reqired format
#creating test_y
def write_res(results):
    out_file = open("test_y", "w")
    for batch in results:
        for pred in batch:
            out_file.write("%s\n" % pred.numpy()[0])
    out_file.close()


# Training the model
def train(model, train_loader, optimizer, loss_func):
    model.train()
    for batch_idx, (data, labels) in tqdm(enumerate(train_loader)):
       # print(data.size())
       # print(data[0])
       # print(labels.size())
       # print(labels)

        # converting the data into GPU format
        data = data.to(device)
        labels = labels.to(device)
        #if torch.cuda.is_available():
         #   data = data.cuda()
          #  labels = labels.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


#TODO: fix to our model
#testing the model
def test(model, test_loader):
    model.eval()
    pred_list = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim = True)[1]

            #adding the prediction to the file
            pred_list.append(pred)
            correct += pred.eq(target.view_as(pred)).cpu().sum()  # counting how many correct predictions
    # print the average loss
    test_loss /= len(test_loader.dataset)
    print('\nTest Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

    return pred_list


# Defining the Model
# Convolutional Neural Networks with five layers
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            # 1st convolution layers set
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 32 x 161 x 101
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 32 x 161 x 101
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 32 x 80 x 50
            # 2nd convolution layers set
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 x 80 x 50
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 64 x 80 x 50
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 x 40 x 25
            # 3rd convolution layers set
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 128 x 40 x 25
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 128 x 40 x 25
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.ZeroPad2d((1, 0, 0, 0)),  # 128 x 40 x 26
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128 x 20 x 13
            # 4th convolution layers set
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 256 x 20 x 13
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 256 x 20 x 13
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.ZeroPad2d((1, 0, 0, 0)),  # 256 x 20 x 14
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256 x 10 x 7
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(256 * 10 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 30) #try 30
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return F.log_softmax(x, dim=1)

if __name__ == "__main__":
    batchSize = 64
    # Load the data and converting to tensor
    #training_set = GCommandLoader('gcommands/train')
    #validation_set = GCommandLoader('gcommands/valid')
    #test_set = GCommandLoader('gcommands/test')
    print("start loading")
    #TODO: updade the Parameters in the DataLoader function
    train_loader = torch.utils.data.DataLoader(
        trainDataset, batch_size=batchSize, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)
    print("train")
    #print(torch.max(trainDataset[100][0]))
    #print(trainDataset[999][0])
    #print(len(trainDataset))
    #for k, (input, label) in enumerate(train_loader):
     #   print(input.size(), len(label))

    validation_loader = torch.utils.data.DataLoader(
        validDataset, batch_size=batchSize, shuffle=False, pin_memory=True, sampler=None)
    print("valid")
    test_loader = torch.utils.data.DataLoader(
        testDataset, batch_size=batchSize, shuffle=False, pin_memory=True, sampler=None)
    print("test")
    #TODO: update the Parameters

    # defining the learning rate
    lr = 0.001

    # defining the epoches number
    epochs = 10
    device = "cpu"
     #checking if GPU is available
    if torch.cuda.is_available():
        device = "cude:0"
        print("using gpu")
    else:
        print("using cpu")
    #    model = model.cuda()
     #   criterion = criterion.cuda()

    # defining the model
    model = Net().to(device)

    # defining the loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # defining the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        # training
    for i in range(epochs):
        train(model, train_loader, optimizer, criterion)
        test(model, validation_loader)

    # shuffeling after every epoch
        #train_loader = torch.utils.data.DataLoader(labeled_training_examples_dataset, batch_size=64, shuffle=True)

    # testing + creating test_y
    #pred_list = test(model, test_loader)
    #write_res(pred_list)
 # plotting the training and validation loss
 #plt.plot(train_losses, label='Training loss')
 #plt.plot(val_losses, label='Validation loss')
 #plt.legend()
 #plt.show()