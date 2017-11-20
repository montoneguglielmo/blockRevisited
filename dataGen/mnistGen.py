from dataGenerator import *
import gzip
import cPickle as pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class dataGenerator(dataGeneratorPrototip):

    def __init__(self, **kwargs):
        self.datafile = '/home/guglielmo/dataset/mnist.pkl.gz'
        with gzip.open(self.datafile, 'rb') as f:
            data = pickle.load(f)
        
        images = data[0].reshape(data[0].shape[0],1,28,28)
        labels = data[1]

        self.n_test_samples  = 10000
        self.n_valid_samples = 10000
        self.n_train_samples = 50000
        
        images_test  = images[:self.n_test_samples]
        images_valid = images[self.n_test_samples:self.n_test_samples+self.n_valid_samples]
        images_train = images[self.n_test_samples+self.n_valid_samples:]

        labels_test  = labels[:self.n_test_samples]
        labels_valid = labels[self.n_test_samples:self.n_test_samples+self.n_valid_samples]
        labels_train = labels[self.n_test_samples+self.n_valid_samples:]
        mnistPartTest   = mnist(images_test, labels_test, transform=ToTensor())
        mnistPartValid  = mnist(images_valid, labels_valid, transform=ToTensor())
        mnistPartTrain  = mnist(images_train, labels_train, transform=ToTensor())

        self.testloader  = DataLoader(mnistPartTest, batch_size=500, shuffle=False, num_workers=1)
        self.validloader = DataLoader(mnistPartValid, batch_size=500, shuffle=False, num_workers=1)
        self.trainloader = DataLoader(mnistPartTrain, batch_size=20, shuffle=True, num_workers=1)

    
    def returnGen(self,sets,**kwargs):
        if sets == "test":
            return self.testloader
        if sets == "valid":
            return self.validloader
        if sets == "train":
            return self.trainloader


    def returnConfig(self,**kwargs):
        conf = {'datafile': self.datafile, 'n_test_samples': self.n_test_samples, 'n_valid_samples': self.n_valid_samples, 'n_train_samples': self.n_train_samples}
        return conf
        


class mnist(Dataset):

    def __init__(self, inp, trg, transform=None):

        self.inp       = inp
        self.trg       = trg
        self.transform = transform
        
    def __len__(self):
        return self.inp.shape[0]

    def __getitem__(self, idx):
        img    = self.inp[idx]
        trg    = self.trg[idx]

        #print img.shape
        img = np.reshape(img, (img.shape[1] * img.shape[2]))
        
        if self.transform:
            img = self.transform(img)

            
        return img, trg


class ToTensor(object):
    def __call__(self, image):
        return torch.from_numpy(image).float()
