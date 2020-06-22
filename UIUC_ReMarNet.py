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

os.environ["CUDA_VISIBLE_DEVICES"] =  "1"
store_name = "../../results/UIUC/UIUC_1_9_Ours_0.001"
time_str = time.strftime("%m-%d-%H-%M", time.localtime())
exp_dir = store_name + "_" + time_str + "_" + "{}".replace(".", "p")

try:
    os.stat(exp_dir)
except:
    os.makedirs(exp_dir)

logging.info("OPENING " + exp_dir + '/results_train.csv')
logging.info("OPENING " + exp_dir + '/results_test.csv')

results_train_file = open(exp_dir + '/results_train.csv', 'w')
results_train_file.write('epoch, train_acc_RN, train_acc_FC, train_acc_Ours, train_loss\n')
results_train_file.flush()

results_test_file = open(exp_dir + '/results_test.csv', 'w')
results_test_file.write('epoch, test_acc_RN, test_acc_FC, test_acc_Ours, test_loss\n')
results_test_file.flush()

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

transform_query = transforms.Compose([
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


queryset    = torchvision.datasets.ImageFolder(root=dataset_fold + "query", transform=transform_query)
queryloader = torch.utils.data.DataLoader(queryset, batch_size=class_num, shuffle=False, num_workers=0)

testset = torchvision.datasets.ImageFolder(root=dataset_fold + "test", transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

class New_Net(nn.Module):
    def __init__(self, model):
        super(New_Net, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.classifier_1 = RelationNetwork()
        self.classifier_2 = Classifier_2()


    def forward(self, query_samples, train_samples):
        x_train = self.features(train_samples)
        x_query = self.features(query_samples)
        x_1=self.classifier_1(x_query, x_train)
        x = x_train.view(x_train.size(0), -1)
        x_2 = self.classifier_2(x)
        return x_1, x_2


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1024, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64,32)
        self.fc2 = nn.Linear(32,1)


    def forward(self, query_samples, train_samples):
        x = self.Calculation(query_samples, train_samples)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out


    def Calculation(self, query_samples, train_samples):
        sample_features_ext = query_samples.unsqueeze(0).repeat(len(train_samples), 1, 1, 1, 1)
        batch_features_ext  = train_samples.unsqueeze(0).repeat(len(query_samples), 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, 512*2, 7, 7)
        return relation_pairs

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
    correct_1 = 0
    correct_2 = 0
    total = 0
    idx = 0
    for batch_idx, (train_samples, targets) in enumerate(trainloader):
        idx = batch_idx
        optimizer.zero_grad()

        if use_cuda:
            train_samples, targets = train_samples.cuda(), targets.cuda()

        train_samples, targets = Variable(train_samples), Variable(targets)
        query_samples, query_labels = queryloader.__iter__().next()

        if use_cuda:
            query_samples, query_labels = query_samples.cuda(), query_labels.cuda()

        query_samples, query_labels = Variable(query_samples), Variable(query_labels)
        one_hot_labels = Variable(torch.zeros(len(train_samples), class_num).scatter_(1, targets.cpu().view(-1, 1), 1)).cuda()


        outputs_1, outputs_2 = net(query_samples, train_samples)    # RN , FC
        outputs_1 = outputs_1.view(-1, class_num)
        outputs = outputs_1 + outputs_2

        loss1 = mse(outputs_1, one_hot_labels)
        loss2 = criterion(outputs_2, targets)
        loss = loss1 + loss2
        loss.backward()

        optimizer.step()
        train_loss += loss.item()

        _, predicted_1 = torch.max(outputs_1.data, 1)
        _, predicted_2 = torch.max(outputs_2.data, 1)
        _, predicted = torch.max(outputs.data, 1)

        total += targets.size(0)


        correct_1 += predicted_1.eq(targets.data).cpu().sum()
        correct_2 += predicted_2.eq(targets.data).cpu().sum()
        correct += predicted.eq(targets.data).cpu().sum()

    train_acc_1 = 1.*correct_1.numpy()/total
    train_acc_2 = 1.*correct_2.numpy()/total
    train_acc = 1.*correct.numpy()/total
    train_loss = train_loss/(idx+1)

    results_train_file.write('%d, %.4f, %.4f, %.4f, %.4f\n' % (epoch, train_acc_1, train_acc_2, train_acc, train_loss))
    results_train_file.flush()


    print('loss: {}, acc_1: {}%, acc_2: {}%, acc: {}%'.format(
        train_loss, 100. * correct_1 / total, 100. * correct_2 / total, 100. * correct / total))
    return train_acc, train_loss


def test(epoch, net, dataset):
    net.eval()
    test_loss = 0
    correct = 0
    correct_1 = 0
    correct_2 = 0
    total = 0
    idx = 0
    for batch_idx, (test_samples, targets) in enumerate(dataset):
        idx = batch_idx
        if use_cuda:
            test_samples, targets = test_samples.cuda(), targets.cuda()

        test_samples, targets = Variable(test_samples), Variable(targets)
        query_samples, query_labels = queryloader.__iter__().next()

        if use_cuda:
            query_samples, query_labels = query_samples.cuda(), query_labels.cuda()

        query_samples, query_labels = Variable(query_samples), Variable(query_labels)
        one_hot_labels = Variable(torch.zeros(len(test_samples), class_num).scatter_(1, targets.cpu().view(-1,1), 1)).cuda()

        outputs_1, outputs_2 = net(query_samples,test_samples)
        outputs_1 = outputs_1.view(-1,class_num)
        outputs = outputs_1 + outputs_2

        loss1 = mse(outputs_1,one_hot_labels)
        loss2 =  criterion(outputs_2,targets)
        loss = loss1 + loss2

        _, predicted_1 = torch.max(outputs_1.data, 1)
        _, predicted_2 = torch.max(outputs_2.data, 1)
        _, predicted = torch.max(outputs.data, 1)

        total += targets.size(0)

        correct_1 += predicted_1.eq(targets.data).cpu().sum()
        correct_2 += predicted_2.eq(targets.data).cpu().sum()
        correct += predicted.eq(targets.data).cpu().sum()

        test_loss += loss.item()


    test_loss = test_loss/(idx+1)
    test_acc_1 = 1.*correct_1.numpy()/total
    test_acc_2 = 1.*correct_2.numpy()/total
    test_acc = 1.*correct.numpy()/total
    results_test_file.write('%d, %.4f, %.4f, %.4f, %.4f\n' % (epoch, test_acc_1, test_acc_2, test_acc, test_loss))
    results_test_file.flush()

    print('loss: {}, acc_1: {}%, acc_2: {}%, acc: {}%'.format(
        test_loss, 100. * correct_1 / total, 100. * correct_2 / total, 100. * correct / total))

    return test_acc, test_acc_1, test_acc_2



for step in range(15):
    net_vgg = models.vgg16(pretrained=True)
    net = New_Net(net_vgg).cuda()
    criterion =  nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    min_loss = 1000000
    optimizer =optim.RMSprop([ 
                               {'params': net_vgg.features.parameters(), 'lr':0.00001},
                               {'params': net.classifier_1.parameters(), 'lr':0.001},
                               {'params': net.classifier_2.parameters(), 'lr':0.0001}
        ],
                            weight_decay=L2  )        




    for epoch in range(0, nb_epochs):

        train_acc,train_loss = train(epoch=epoch,net=net,dataset=trainloader)
        test_acc, test_acc_1, test_acc_2 = test(epoch=epoch,net=net, dataset=testloader)

        if train_loss < min_loss :
            min_loss = train_loss
            torch.save(net.state_dict(), store_name+'.pth')

    if  os.path.exists(store_name+'.pth'):
        net.load_state_dict(torch.load(store_name+'.pth'))
        test_acc, test_acc_1, test_acc_2 = test(epoch=0,net=net,dataset=testloader)
        f = file(store_name + ".txt","a")
        f.write(str(test_acc)+","+str(test_acc_1)+","+str(test_acc_2)+'\n')
        f.close()

