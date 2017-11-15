import torch.nn.functional as F
import torch.nn as nn


class bidict(dict):
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.iteritems():
            self.inverse.setdefault(value,[]).append(key) 

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key) 
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value,[]).append(key)        

    def __delitem__(self, key):
        self.inverse.setdefault(self[key],[]).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]: 
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)

nameToActFun = bidict({'rlu': F.relu, 'softmax':nn.Softmax()})
        

def topologyOrder(netStr):

    netOrd   = []
    noIncEdg = []
    netNames = netStr.keys()
    netNames.sort()
    
    for netName in netNames:
        if netStr[netName]['input'] == ['input']:
            noIncEdg.append(netName)

    while len(noIncEdg) > 0:
        netOrd.append(noIncEdg.pop())
        
        removed = netOrd[-1]
        for netName in netNames:
            lstInp = netStr[netName]['input']
            if removed in lstInp:
                lstInp.remove(removed)
                netStr[netName]['input'] = lstInp

                if len(lstInp) == 0:
                    noIncEdg.append(netName)

    for netName in netNames:
         if len(netStr[netName]['input']) != 0 and (netStr[netName]['input'] != ['input']):
             raise ValueError('The graph has at least one cycle')

    return netOrd



if __name__ == "__main__":

    netStr = {
        'net1' :{'input':['input'], 'output':['net3']},
        'net2' :{'input':['input'], 'output':['net4']},
        'net3' :{'input':['net1'],  'output':['net4']},
        'net4' :{'input':['net3', 'net2'], 'output':[]}
    }

    
    print topologyOrder(netStr)
