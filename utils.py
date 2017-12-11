import torch.nn.functional as F
import torch.nn as nn
import torch.optim

##TRY IT ALSO WITH WEIGHT THAT SHOULD NOT BE UPDATED
class vanilla(torch.optim.Optimizer):

    def __init__(self, params, lr=0.1, weight_decay=0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(vanilla, self).__init__(params, defaults)
                      
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss=closure()
                      
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group['weight_decay'] != 0:
                    d_p.add_(group['weight_decay'], p.data)
                p.data.add_(-group['lr'], d_p)
                
        return loss


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

nameToActFun = bidict({'rlu': F.relu, 'softmax':nn.Softmax})
nameToOptim  = bidict({'Vanilla': vanilla, 'SGD': torch.optim.SGD})

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
