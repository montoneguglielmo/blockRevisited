import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from utils import topologyOrder, nameToActFun
import torch.optim as optim
import copy
#from visualize import *
import collections
import argparse
import json
import os
import time
import numpy as np
import atexit
import gc
import h5py

class Net(nn.Module):

    def __init__(self, **kwargs):
        super(Net, self).__init__()

        self.fileName   = kwargs['fileName'] 
        self.lstSubNets = kwargs['lstSubNets']
        self.netStrc    = kwargs['netStrc']
        
        self.netNames   = self.lstSubNets.keys()
        self.netNames.sort()

        
        self.subNets = {}
        self.actFuns = {}
        for netName in self.netNames:
            netConf = self.lstSubNets[netName]
            if netConf['type'] == 'conv':
                self._modules[netName]      = nn.Conv2d(**netConf['params'])
                self._modules[netName].type = 'conv'
            if netConf['type'] == 'fc':
                self._modules[netName]      = nn.Linear(**netConf['params'])
                self._modules[netName].type = 'fc'
            if netConf['type'] == 'max_pool':
                self._modules[netName] = nn.MaxPool2d(**netConf['params'])
                self._modules[netName].type = 'max_pool'
                
            if 'init' in netConf and netConf['init'] == 'load':
                mod        = torch.load(netConf['file_name'] + '.pt')
                moduleName = netConf['moduleName']
                state_dict={'weight': mod[moduleName + '.weight'], 'bias': mod[moduleName + '.bias']}
                self._modules[netName].load_state_dict(state_dict)

            if 'requires_grad' in netConf and not netConf['requires_grad']:
                self._modules[netName].weight.requires_grad = False
                self._modules[netName].bias.requires_grad   = False
                
            if 'actFun' in netConf:
                self.actFuns[netName] = nameToActFun[netConf['actFun']]
            else:
                self.actFuns[netName] = None
        
        
    def forward(self, x, **kwargs):

        self.topOrd = topologyOrder(copy.deepcopy(self.netStrc))        
        outputs     = {}

        for tp in self.topOrd:
            if self.netStrc[tp]['input'] == ['input']:
                outputs[tp] = self._modules[tp](x)
                if self.actFuns[tp] is not None:
                    outputs[tp] = self.actFuns[tp](outputs[tp])
            else:
                x = []
                for incNet in self.netStrc[tp]['input']:
                    x.append(outputs[incNet])
                    
                x = torch.cat(x,1)
                if self.lstSubNets[tp]['type'] == 'fc':
                    x = x.view(x.size(0), -1)

                outputs[tp] = self._modules[tp](x)
                if self.actFuns[tp] is not None:
                    outputs[tp] = self.actFuns[tp](outputs[tp])

        out_net = []
        for tp in self.topOrd:
            if 'output' in tp:
                out_net.append(outputs[tp])
        if len(out_net) == 1:
            out_net = out_net[0]
                    
        return out_net
                        
    
    def returnNetParams(self):
        return self.state_dict()

    
    
    def returnNetCfg(self, **kwargs):
        lstSubNets = {name: {} for name in self._modules.keys()}
        for md in self._modules.keys():
            lstSubNets[md]['type']   = self._modules[md].type
            lstSubNets[md]['init']   = self.fileName
            lstSubNets[md]['params'] = {}
            if self._modules[md].type == 'conv':
                lstSubNets[md]['params']['in_channels']  = self._modules[md].in_channels
                lstSubNets[md]['params']['out_channels'] = self._modules[md].out_channels
                lstSubNets[md]['params']['kernel_size']  = self._modules[md].kernel_size
                lstSubNets[md]['params']['stride']       = self._modules[md].stride
            if self._modules[md].type == 'max_pool':
                lstSubNets[md]['params']['kernel_size']  = self._modules[md].kernel_size
            if self._modules[md].type == 'fc':
                lstSubNets[md]['params']['in_features']  = self._modules[md].in_features
                lstSubNets[md]['params']['out_features'] = self._modules[md].out_features
            
            if len(self._modules[md]._parameters) > 0:
                lstSubNets[md]['requires_grad'] =  self._modules[md]._parameters['weight'].requires_grad
                      
            if self.actFuns[md] is not None:
                lstSubNets[md]['actFun'] = nameToActFun.inverse[self.actFuns[md]][0]
        return lstSubNets


    
    def returnNetStrct(self, **kwargs):
        return self.netStrc


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
    parser.add_argument("--config", help="json file describing each layer in the network and the structure of the network")
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
    fileName = dataGeneratorName.split('.')[0].replace('/', '.')
    dataGen = getattr(__import__(fileName, fromlist=['dataGenerator']), 'dataGenerator')()
    testloader  = dataGen.returnGen('test')
    validloader = dataGen.returnGen('valid')
    trainloader = dataGen.returnGen('train')


    ## BUILDING THE NETWORK
    with open(confFile) as f:
        confJson = json.load(f)

    configNet = confJson['configNet']
    strctNet  = confJson['strctNet']

    net = Net(lstSubNets=configNet, netStrc=strctNet, fileName=results['name'])

    if torch.cuda.is_available():
        net.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    optimizer.zero_grad()

    n_epochs      = 100
    ecp_tolerance = 10
    results['n_epochs']     = n_epochs
    results['ecp_tolerance']= ecp_tolerance    
    lastEpcBestAcc = 0
    oldAcc      = -np.inf
    epoch       = 0
    while epoch < n_epochs:  # loop over the dataset multiple times
        epoch += 1
        
        running_loss = 0.0
        cnt_b        = 0
        for inputs, labels in trainloader:
            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda())
                if isinstance(labels, list):
                    labels = [Variable(lab.cuda()) for lab in labels]
                else:
                    labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                if isinstance(labels, list):
                    labels = [Variable(lab) for lab in labels]
                else:
                    labels = Variable(labels)
                    
            cnt_b += 1
            
            optimizer.zero_grad()
            outputs = net(inputs)
            
            loss = 0
            if isinstance(outputs, list):
                for out, lab in zip(outputs, labels):
                    loss += criterion(out, lab)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
        running_loss /= float(cnt_b) 
        print('Epoch %d,  loss: %.4f' % (epoch, running_loss))    
        
    
        correct = 0
        total = 0
        for inputs, labels in validloader:
            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)
            
            outputs = net(inputs)
            if isinstance(outputs, list):
                predicted = np.ones(outputs[0].size()[0])
                for out, lab in zip(outputs, labels):
                    _, pred = torch.max(out.data, 1)
                    predicted *= (pred.cpu() == lab).numpy()
                total     += outputs[0].size(0)
                correct   += predicted.sum()
            else:
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.cpu() == labels).sum()
        accValid = 100. * float(correct)/float(total)            

        
        correct = 0
        total = 0
        for inputs, labels in testloader:
            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)    
            outputs = net(inputs)
            if isinstance(outputs, list):
                predicted = np.ones(outputs[0].size()[0])
                for out, lab in zip(outputs, labels):
                    _, pred = torch.max(out.data, 1)
                    predicted *= (pred.cpu() == lab).numpy()
                total   += output[0].size(0)
                correct += predicted.sum()
            else:
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.cpu() == labels).sum()
        accTest = 100. * float(correct)/float(total)
        print('Epoch:%d, Valid(Acc) %.4f%%, Test(Acc) %.4f%%' %(epoch, accValid, accTest))

        
        if accValid < oldAcc and (epoch-lastEpcBestAcc)>epc_tolerance:
            break
        else:
            oldAcc         = accValid
            lastEpcBestAcc = epoch
            results['validAcc']   = accValid
            results['testValid']  = accTest
            results['trainError'] = running_loss
            results['epoch']      = epoch
            results['configNet']  = net.returnNetCfg()
            results['strctNet']   = net.returnNetStrct()
            results['dataset']    = dataGen.returnConfig()
            torch.save(net.returnNetParams(), net.fileName + '.pt')

            with open(net.fileName + 'res.json', 'w') as f:
                json.dump(results, f, indent=3, sort_keys=True)
