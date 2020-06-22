from __future__ import print_function
import os
import time
import math
import torch
import random
import logging
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torchvision.models as models
from torchvision import datasets, transforms
from scipy import *

os.environ["CUDA_VISIBLE_DEVICES"] =  "0"
store_name = "../../results/UIUC/UIUC_1_1_FC_epoch_50"
time_str = time.strftime("%m-%d-%H-%M", time.localtime())
exp_dir = store_name + "_" + time_str + "_" + "{}".replace(".", "p")

try:
    os.stat(exp_dir)
except:
    os.makedirs(exp_dir)

logging.info("OPENING " + exp_dir + '/results_train.csv')
logging.info("OPENING " + exp_dir + '/results_test.csv')

results_train_file = open(exp_dir + '/results_train.csv', 'w')
results_train_file.write('epoch, train_acc, train_loss\n')
results_train_file.flush()

results_test_file = open(exp_dir + '/results_test.csv', 'w')
results_test_file.write('epoch, test_acc, test_loss\n')
results_test_file.flush()


#seed
torch.manual_seed(1)
torch.cuda.manual_seed(1)

L2 = 5e-4
new_net = True
nb_epochs = 50
Last_FC_num = 32
use_cuda = True
class_num = 8
dataset_fold = "../dataset/UIUC/"


#Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


trainset    = torchvision.datasets.ImageFolder(root=dataset_fold + "train", transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

testset = torchvision.datasets.ImageFolder(root=dataset_fold + "test", transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)


class New_Net(nn.Module):
    def __init__(self, model):
        super(New_Net, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.classifier_2 = Classifier_2()


    def forward(self, train_samples):
        x_train = self.features(train_samples)
        x = x_train.view(x_train.size(0), -1)
        x_2 = self.classifier_2(x)
        return x_2


class Classifier_2(nn.Module):
    def __init__(self):
        super(Classifier_2, self).__init__()              
        self.fc1 = nn.Linear(25088, Last_FC_num)
        self.fc2 = nn.Linear(Last_FC_num, class_num)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def train(epoch, net, dataset):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    idx = 0
    for batch_idx, (train_samples, targets) in enumerate(trainloader):
        idx = batch_idx
        optimizer.zero_grad()
        if use_cuda:
            train_samples, targets = train_samples.cuda(), targets.cuda()

        train_samples, targets = Variable(train_samples), Variable(targets)
        outputs = net(train_samples)
        loss =  criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()


    train_acc = 1.*correct.numpy()/total
    train_loss = train_loss/(idx+1)
    results_train_file.write('%d,  %.4f, %.4f\n' % (epoch,  train_acc, train_loss))
    results_train_file.flush()


    print('loss: {}, acc: {}%'.format(
        train_loss, 100. * correct / total))
    return train_acc, train_loss


def test(epoch, net, dataset):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    for batch_idx, (test_samples, targets) in enumerate(dataset):
        idx = batch_idx
        if use_cuda:
            test_samples, targets = test_samples.cuda(), targets.cuda()

        test_samples, targets = Variable(test_samples), Variable(targets)

        outputs = net(test_samples)
        loss =  criterion(outputs,targets)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        test_loss += loss.item()


    test_loss = test_loss/(idx+1)
    test_acc = 1.*correct.numpy()/total
    results_test_file.write('%d, %.4f, %.4f\n' % (epoch, test_acc, test_loss))
    results_test_file.flush()

    print('loss: {},  acc: {}%'.format(
        test_loss, 100. * correct / total))

    return test_acc



for step in range(15):
    net_vgg = models.vgg16(pretrained=True)
    net = New_Net(net_vgg).cuda()
    criterion =  nn.CrossEntropyLoss()
    min_loss = 1000000
    optimizer =optim.RMSprop([ 
                               {'params': net_vgg.features.parameters(), 'lr':0.00001},
                               {'params': net.classifier_2.parameters(), 'lr':0.0001}
        ],
                            weight_decay=L2  )        


    for epoch in range(0, nb_epochs):
        train_acc, train_loss = train(epoch=epoch,net=net,dataset=trainloader)
        test_acc= test(epoch=epoch,net=net, dataset=testloader)

        if train_loss < min_loss :
            min_loss = train_loss
            torch.save(net.state_dict(), store_name+'.pth')
            
    if  os.path.exists(store_name+'.pth'):
        net.load_state_dict(torch.load(store_name+'.pth'))
        test_acc = test(epoch=0,net=net,dataset=testloader)
        f = file(store_name + ".txt","a")
        f.write(str(test_acc) + ',' + '\n')
        f.close()

if os.path.exists(store_name + '.txt'):
    fr = open(store_name + '.txt')
    accs = fr.readlines()
    accs_list = []
    if len(accs) == 15:
        for acc in accs:
            acc = float(acc.strip().split(',')[0])
            accs_list.append(acc)
    mean = np.mean(accs_list)
    std = np.std(accs_list)
    f1 = file(store_name + "result.txt","a")
    f1.write('mean:' + str(mean) + "," + 'std:' + str(std) + '\n')
    f1.close()
   

