from fundamentalBlocks import *
from torch.autograd import Variable


if __name__ == "__main__":

    atexit.register(exit_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="json file describing each layer in the network and the structure of the network")
    parser.add_argument("--dataGenerator", help="dataGenerator, one of the .py file in the folder dataGen")
    args = parser.parse_args()

    confFile          = args.config
    dataGeneratorName = args.dataGenerator
    fileName          = dataGeneratorName.split('.')[0].replace('/', '.')
    dataGen           = getattr(__import__(fileName, fromlist=['dataGenerator']), 'dataGenerator')(outputKind='regression')
    testloader        = dataGen.returnGen('test')
    
    
    with open(confFile) as f:
        confJson = json.load(f)

    configNet  = confJson['configNet']
    strctNet   = confJson['strctNet']

    net        = Net(lstSubNets=configNet, netStrc=strctNet, fileName='NoName')
    trg_values = np.linspace(0, 1, dataGen.test_dts.n_classes)
    trg_values[0]  = trg_values[1]/4.
    trg_values[-1] = trg_values[-2] + (trg_values[-1]-trg_values[-2])*3./4. 
    
    rms_error = 0
    cnt_b     = 1
    for inputs, labels in testloader:
        inputs  = Variable(inputs) 
        outputs = net(inputs)
        _, predicted_motor = torch.max(outputs[0], 1)
        _, predicted_steer = torch.max(outputs[1], 1)
        
        output_motor      = trg_values[predicted_motor.data.numpy()]
        output_steer      = trg_values[predicted_steer.data.numpy()]
        
        trg_motor   = labels[0].numpy()
        trg_steer   = labels[1].numpy()

        rms_error   += (np.mean(np.abs(output_motor - trg_motor)) + np.mean(np.abs(output_steer - trg_steer)))/2.
        cnt_b +=1

    print "RMS error:", rms_error/float(cnt_b)
