import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from utils import topologyOrder, nameToActFun, nameToOptim, vanilla
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
#from visualize import *
import collections
import argparse
#import json
import os
import time
import numpy as np
import atexit
import gc
import h5py
import time

class resAdaptMod(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, n_tasks=1, stride=1, padding=0, dilatation=1, groups=1, bias=True):
        super(resAdaptMod, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilatation, groups, bias)

        self.listTaskParams = []
        for _ in range(n_tasks):
            self.listTaskParams.append(nn.Conv2d(out_channels, out_channels, 1))

        self.listTaskParams = torch.nn.ModuleList(self.listTaskParams)
            
    def forward(self, input, task_id=0):
        output_conv = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        output      = output_conv +  self.listTaskParams[task_id](output_conv)
        return output



class resNetCar(nn.Module):
    def __init__(self):
        super(resNetCar, self).__init__()
        self.resAdaptMod1 = resAdaptMod(12, 48, 7, stride=2)
        self.resAdaptMod2 = resAdaptMod(48, 96, 5)
        self.resAdaptMod3 = resAdaptMod(96, 128, 4)
        self.resAdaptMod4 = resAdaptMod(128, 256, 3)
        self.outlayer1    = nn.Linear(3072,5)
        self.outlayer2    = nn.Linear(3072,5)

    def forward(self, x, task_id=0):
        x  = F.max_pool2d(F.relu(self.resAdaptMod1(x, task_id)), 2)
        x  = F.max_pool2d(F.relu(self.resAdaptMod2(x, task_id)), 2)
        x  = F.relu(self.resAdaptMod3(x, task_id))
        x  = F.relu(self.resAdaptMod4(x, task_id))
        x  = F.max_pool2d(x,2)
        x  = x.view(-1, np.prod(x.shape[1:]))
        x1 = self.outlayer1(x)
        x2 = self.outlayer2(x)
        return [x1, x2]


def buildVariable(inputs):
    if torch.cuda.is_available():
        if isinstance(inputs, list):
            inputs = [Variable(input.cuda()) for input in inputs]
        else:
            inputs = Variable(inputs.cuda())
    else:
        if isinstance(inputs, list):
            inputs = [Variable(input) for input in inputs]
        else:
            inputs = Variable(inputs)

    return inputs


def exit_handler():
    print 'Closing hdf5 files...'
    for obj in gc.get_objects():
        if isinstance(obj, h5py.File):   # Just HDF5 files
            try:
                obj.close()
            except:
                pass # Was
    print 'done'

                         

if __name__ == "__main__":

    atexit.register(exit_handler)
    parser = argparse.ArgumentParser()
    #parser.add_argument("--config", help="json file describing each layer in the network and the structure of the network")
    parser.add_argument("--dataGenerator", help="dataGenerator, one of the .py file in the folder dataGen")
    args = parser.parse_args()

    results                      = {}
    confFile                     = args.config
    dataGeneratorName            = args.dataGenerator
    results['confFile']          = confFile
    results['dataGeneratorFile'] = dataGeneratorName
    
    if not os.path.exists("/home/guglielmo/torchclassifier/"):
        os.makedirs("/home/guglielmo/torchclassifier/")
    results['name'] = time.strftime("/home/guglielmo/torchclassifier/%H%M%S")
    

    ##CREATING THE DATA GENERATOR
    fileName    = dataGeneratorName.split('.')[0].replace('/', '.')
    dataGen     = getattr(__import__(fileName, fromlist=['dataGenerator']), 'dataGenerator')()
    testloader  = dataGen.returnGen('test')
    validloader = dataGen.returnGen('valid')
    trainloader = dataGen.returnGen('train')

    net = resNetCar()
    if torch.cuda.is_available():
        net.cuda()

    criterion    = nn.CrossEntropyLoss()                        
    optimizer    = optim.SGD(net.parameters(), lr=1e-4, momentum = .9)
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    n_epochs      = 100
    epc_tolerance = 10
    oldAcc        = -np.inf
    epoch         = -1
    while epoch < n_epochs:  #loop over the dataset multiple times
        start_time   = time.time()
        epoch += 1
        #if 'lr_scheduler' in confJson:
        lr_scheduler.step()
        running_loss = 0.0
        cnt_b        = 0
        total        = 0
        correct      = 0
        for inputs, labels in trainloader:
            inputs = buildVariable(inputs)
            labels = buildVariable(labels)
            
            outputs = net(inputs)
            if isinstance(outputs, list):
                loss  = 0
                if torch.cuda.is_available():
                    predicted = torch.ones(1, outputs[0].size()[0]).cuda()
                else:
                    predicted = torch.ones(1, outputs[0].size()[0])                    
                for out, lab in zip(outputs, labels):
                    loss   += criterion(out, lab)
                    _, pred = torch.max(out.data, 1)
                    predicted *= (pred == lab.data).float()
                total     += outputs[0].size(0)
                correct   += torch.sum(predicted)
            else:
                loss         = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total       += labels.size(0)
                correct     += (predicted  == labels.data).sum()

                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.data[0]
            cnt_b += 1
        running_loss /= float(cnt_b)
        accTrain = 100. * (1. - float(correct)/float(total))
        print('Epoch %d,  loss: %.4f' % (epoch, running_loss))    
        

        correct = 0
        total   = 0
        for inputs, labels in validloader:
            inputs  = buildVariable(inputs)
            outputs = net(inputs)
            if isinstance(outputs, list):
                if torch.cuda.is_available():
                    predicted = torch.ones(1, outputs[0].size()[0]).cuda()
                else:
                    predicted = torch.ones(1, outputs[0].size()[0])

                for out, lab in zip(outputs, labels):
                    if torch.cuda.is_available():
                        lab = lab.cuda()
                    _, pred = torch.max(out.data, 1)
                    predicted *= (pred == lab).float()
                total     += outputs[0].size(0)
                correct   += torch.sum(predicted)
            else:
                if torch.cuda.is_available():
                    labels = labels.cuda()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted  == labels).sum()
        accValid = 100. * (1. - float(correct)/float(total))            
        
        correct = 0
        total = 0
        for inputs, labels in testloader:
            inputs = buildVariable(inputs)
            outputs = net(inputs)
            if isinstance(outputs, list):
                if torch.cuda.is_available():
                    predicted = torch.ones(1, outputs[0].size()[0]).cuda()
                else:
                    predicted = torch.ones(1, outputs[0].size()[0])
                
                for out, lab in zip(outputs, labels):
                    if torch.cuda.is_available():
                        lab = lab.cuda()
                    _, pred = torch.max(out.data, 1)
                    predicted *= (pred == lab).float()
                total   += outputs[0].size(0)
                correct += torch.sum(predicted)
            else:
                if torch.cuda.is_available():
                    labels = labels.cuda()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
        accTest = 100. * (1.- float(correct)/float(total))

        lastTime = (time.time() - start_time)/60.0
        print('Epoch:%d, Train(Miss) %.3f%%, Valid(Miss) %.3f%%, Test(Miss) %.3f%%. Done in %.1f' %(epoch, accTrain, accValid, accTest, lastTime))

        
        if accValid < oldAcc and (epoch-lastEpcBestAcc)>epc_tolerance:
             break
        else:
            oldAcc         = accValid
            lastEpcBestAcc = epoch
            results['runningTime']= lastTime
            results['validAcc']   = accValid
            results['testValid']  = accTest
            results['trainError'] = running_loss
            results['epoch']      = epoch
            results['configNet']  = net.returnNetCfg()
            results['strctNet']   = net.returnNetStrct()
            results['dataset']    = dataGen.returnConfig()
            #results['optimizer']  = {}
            #results['optimizer']['name']  = nameToOptim.inverse[type(optimizer)]
            #keysOpt = optimizer.state_dict()['param_groups'][0].keys()
            #paramOpt = {key:optimizer.state_dict()['param_groups'][0][key] for key in keysOpt if key is not 'params'}
            #results['optimizer']['params']= paramOpt

            #if 'lr_scheduler' in confJson:
            #    results['lr_scheduler']              = {}
            #    results['lr_scheduler']['step_size'] = lr_scheduler.step_size
            #    results['lr_scheduler']['gamma']     = lr_scheduler.gamma
                

            torch.save(net.returnNetParams(), net.fileName + '.pt')

            with open(net.fileName + 'res.json', 'w') as f:
                json.dump(results, f, indent=3, sort_keys=True)
