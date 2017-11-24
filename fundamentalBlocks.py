import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from utils import topologyOrder, nameToActFun, nameToOptim, vanilla
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
import time

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
                self._modules[netName]      = nn.MaxPool2d(**netConf['params'])
                self._modules[netName].type = 'max_pool'
                
            if 'init' in netConf and netConf['init'] == 'load':
               mod        = torch.load(netConf['file_name'] + '.pt')
               moduleName = netConf['moduleName']
               state_dict = {'weight': mod[moduleName + '.weight'], 'bias': mod[moduleName + '.bias']}
               self._modules[netName].load_state_dict(state_dict)

            if 'requires_grad' in netConf and not netConf['requires_grad']:
                for p in self._modules[netName].parameters():
                    p.requires_grad = False
                
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
    params    = filter(lambda p: p.requires_grad, net.parameters())

    if 'optim' in confJson:
        optimName = confJson['optim']['name']
        if 'params' in confJson['optim']:
            paramsOpt = confJson['optim']['params']
            optimizer   = nameToOptim[optimName](params, **paramsOpt)
        else:
            optimizer   = nameToOptim[optimName](params)
            
    n_epochs      = 100
    epc_tolerance = 10
    results['n_epochs']     = n_epochs
    results['epc_tolerance']= epc_tolerance    
    lastEpcBestAcc = 0
    oldAcc      = -np.inf
    epoch       = -1
    while epoch < n_epochs:  # loop over the dataset multiple times
        start_time   = time.time()
        epoch += 1
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
                    loss += criterion(out, lab)
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
            results['optimizer']  = {}
            results['optimizer']['name']  = nameToOptim.inverse[type(optimizer)]
            keysOpt = optimizer.state_dict()['param_groups'][0].keys()
            paramOpt = {key:optimizer.state_dict()['param_groups'][0][key] for key in keysOpt if key is not 'params'}
            results['optimizer']['params']= paramOpt

            torch.save(net.returnNetParams(), net.fileName + '.pt')

            with open(net.fileName + 'res.json', 'w') as f:
                json.dump(results, f, indent=3, sort_keys=True)
