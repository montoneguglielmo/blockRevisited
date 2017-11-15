    lstSubNets = {'net1':
              {
                  'init': 'random',
                  'type': 'conv',
                  'params':{'in_channels':1, 'out_channels': 6, 'kernel_size': (3,3), 'stride': 1},
                  'actFun':'rlu'
              },
              'net2':
              {
                  'type': 'max_pool',
                  'params':{'kernel_size':2}
              },
              'net3':
              {
                  'init': 'random',
                  'type': 'conv',
                  'params':{'in_channels':6, 'out_channels': 16, 'kernel_size': (4,4), 'stride': 1},
                  'actFun':'rlu'
              },
              'net4':
              {
                  'init': 'random',
                  'type': 'conv',
                  'params':{'in_channels':16, 'out_channels': 16, 'kernel_size': (4,4), 'stride': 1},
                  'actFun':'rlu'
              },
              'output':
              {
                  'init': 'random',
                  'type': 'fc',
                  'params':{'in_features':7*7*16, 'out_features': 10},
                  'actFun':'softmax'
              }
    }

    
netStrc = {
        'net1'  :{'input':['input'], 'output':['net2']},
        'net2'  :{'input':['net1'], 'output':['net3']},
        'net3'  :{'input':['net2'], 'output':['net4']},
        'net4'  :{'input':['net3'], 'output':['output']},
        'output':{'input':['net4']}
}




    # lstSubNets = {'net1':
    #                {
    #                 'init': 'load',
    #                 'file_name' :'model',
    #                 'moduleName': 'net1',
    #                 'type': 'conv',
    #                 'params':{'in_channels':1, 'out_channels': 4, 'kernel_size': (3,3), 'stride': 1},
    #                 'actFun':'rlu'
    #                },
    #          'net2':
    #               {
    #                 'init': 'load',
    #                 'file_name' :'model',
    #                 'moduleName': 'net2',
    #                 'type': 'conv',
    #                 'params':{'in_channels':1, 'out_channels': 2, 'kernel_size': (3,3), 'stride': 1},
    #                 'actFun':'rlu'
    #                },
    #          'output':
    #                {
    #                 'init': 'load',
    #                 'file_name' :'model',
    #                 'moduleName': 'output',
    #                 'type': 'fc',
    #                 'params':{'in_features':30*30*(4+2), 'out_features': 10},
    #                 'actFun':'softmax'
    #                 }
    #         }
