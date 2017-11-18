from dataGenerator import *
import gzip
import cPickle as pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

class dataGenerator(dataGeneratorPrototip):

    def __init__(self):
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='/home/guglielmo/torchExercise/data', train=True, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

        testset  = torchvision.datasets.CIFAR10(root='/home/guglielmo/torchExercise/data', train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)

        #testset  = torchvision.datasets.CIFAR10(root='/home/guglielmo/torchExercise/data', train=False, download=True, transform=transform)
        self.validloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)

        

    def returnGen(self, sets):
        if sets == "test":
            return self.testloader
        if sets == "valid":
            return self.validloader
        if sets == "train":
            return self.trainloader


    def returnConfig(self):
        conf = {'datafile':'cifar10'}
        return conf
