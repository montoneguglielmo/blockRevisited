from torch.utils.data import DataLoader
from torch.utils.data import Dataset as DatasetTorch
import re
import h5py
import numpy as np
import torch
from dataGenerator import *

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


class dataGenerator(dataGeneratorPrototip):

    def __init__(self, **kwargs):

        dtFold = '/home/guglielmo/dataset/car/Mr_Blue/toKill/'
        
        self.datafile  = "direct.hdf5"
        self.datafile  = dtFold + self.datafile

        self.batch_size = 100

        self.n_test_samples  = 25000#30000
        self.n_valid_samples = 25000#30000
        self.n_train_samples = 206000
        
        self.dtm  = datasetManagerCar(self.datafile)
        print "Number of total data present in the file:", self.dtm.n_data
        test_dts  = self.dtm.createDataset(n_samples = self.n_test_samples, name="test", datasetType=datasetCar)
        valid_dts = self.dtm.createDataset(n_samples = self.n_valid_samples, name="valid", datasetType=datasetCar)
        train_dts = self.dtm.createDataset(n_samples = self.n_train_samples, name="train", datasetType=datasetCar)
        
        self.testCar  = DataLoader(test_dts, batch_size=self.batch_size, shuffle=False, num_workers=1)
        self.validCar = DataLoader(valid_dts, batch_size=self.batch_size, shuffle=False, num_workers=1)
        self.trainCar = DataLoader(train_dts, batch_size=self.batch_size, shuffle=True, num_workers=1)
        
    
    def returnGen(self,sets,**kwargs):
        if sets == "test":
            return self.testCar
        if sets == "valid":
            return self.validCar
        if sets == "train":
            return self.trainCar


    def returnConfig(self,**kwargs):
        conf = {'datafile': self.datafile, 'n_test_samples': self.n_test_samples, 'n_valid_samples': self.n_valid_samples, 'n_train_samples': self.n_train_samples, 'batch_sz': self.batch_size}
        return conf




class datasetManager(object):

    def __init__(self, file_hdf):

        self.file_hdf     = file_hdf
        self._openFile()
        
        self.dts_names    = []
        self.dts_ids      = []
        self.n_data       = self._n_data()
        self.unused_ids   = range(self.n_data)
        self.preprocess()
        
    def createDataset(self, n_samples, **kwargs):
        if n_samples > len(self.unused_ids):
            warnings.warn("Then number of sample exceed the number of data left in the dataset.")
            
        name      =  kwargs['name']
        dataset   =  kwargs['datasetType']
        dts       = dataset(n_samples, unused_ids=self.unused_ids, **kwargs)
        self.dts_names.append(kwargs['name'])
        self.dts_ids.append(dts.ids)
        to_remove       = dts.ids
        self.unused_ids = self._remove_ids(to_remove)
        return dts

    def _openFile(self):
        pass

    def closeFile(self):
        pass
    
    def _remove_ids(self,to_remove):
        unused_ids = [id_ for id_ in self.unused_ids if id_ not in to_remove]
        return unused_ids

    def _n_data(self):
        return 0
        
    def preprocess(self):
        pass

    
class dataset(DatasetTorch):

    def __init__(self, n_samples, **kwargs):
        self.n_samples = n_samples
        unused_ids     = kwargs['unused_ids']
        self.ids       = self._get_ids(unused_ids)
        
    def _get_ids(self, unused_ids):
        unused_ids = np.random.permutation(unused_ids)
        return unused_ids[:self.n_samples]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pass


    

class datasetManagerCar(datasetManager):

    def __init__(self, file_hdf):
        super(datasetManagerCar, self).__init__(file_hdf)
        self.fields = ['left','left_timestamp', 'right', 'motor', 'state', 'steer']
        self._addrs = self._dataAddressList()

    def _openFile(self):
        self.fh5  = h5py.File(self.file_hdf, "r")
            
    def closeFile(self):
        self.fh5.close()
            
    def _n_data(self):
        n_data = 0
        segs   = self.fh5['segments'].keys()
        segs.sort(key=natural_keys)
        for seg in segs:
            n_data += self.fh5['segments/' + seg + '/left'].shape[0]
        return n_data

    
    def createDataset(self, n_samples, **kwargs):
        dts = super(datasetManagerCar, self).createDataset(n_samples, **kwargs)
        dts._addrs = self._addrs
        dts.fh5    = self.fh5
        return dts
        
        
    def _dataAddressList(self):
        addLst = []

        segs = self.fh5['segments'].keys()
        segs.sort(key=natural_keys)

        for seg in segs:
            n_data = self.fh5['segments/' + seg + '/motor'].shape[0]
            for cnt_n in range(n_data):
                label = [seg, np.int16(cnt_n)]
                addLst.append(label)
        return addLst




class datasetCar(dataset):

    def __init__(self, n_samples, **kwargs):
        super(datasetCar, self).__init__(n_samples, **kwargs)
        self.motor_max =  np.asarray(70, dtype=np.float32)
        self.motor_min =  np.asarray(50, dtype=np.float32)
        self.steer_max =  np.asarray(100, dtype=np.float32)
        self.steer_min =  np.asarray(-1.5, dtype=np.float32)

        self.time_tol    = 0.7
        self.n_classes   = 5
        self.range_motor = np.linspace(0,1,self.n_classes, dtype=np.float32)
        self.range_steer = np.linspace(0,1,self.n_classes, dtype=np.float32)
        self.toll        = 1./(2.*(self.n_classes-1.))
        
                               
    def get_data(self,label, name):
            data = self.fh5['segments/' + str(label[0]) + '/' + name][label[1]]
            return np.asarray(data, dtype=np.float32)                   

    def __getitem__(self, idx):

        idx = self.ids[idx]

        if idx == 0:
            label_    = self._addrs[idx]
            label_prv = self._addrs[idx]
        else:
            label_    = self._addrs[idx]
            label_prv = self._addrs[idx-1]

        consecutive = True
        if np.abs(self.get_data(label_,"left_timestamp") - self.get_data(label_prv,"left_timestamp")) > self.time_tol:
            consecutive=False
            label_prv  = label_
            
        inp = []
        inp.append(self.get_data(label_, 'left'))
        inp.append(self.get_data(label_prv, 'left'))
        inp.append(self.get_data(label_, 'right'))
        inp.append(self.get_data(label_prv, 'right'))
        inp = (np.concatenate(inp, axis=2) - 127.5)/127.5
        inp = np.rollaxis(inp, 2)

        inp = torch.from_numpy(inp)

        if consecutive:
            trg_m = (self.get_data(label_, 'motor') - self.motor_min)/(self.motor_max-self.motor_min)
            trg_s = (self.get_data(label_, 'steer') - self.steer_min)/(self.steer_max-self.steer_min)
        else:
            trg_m = 60.0
            trg_s = 50.0
            
        trg_m = np.argmax(np.isclose(self.range_motor, trg_m, atol=self.toll) >.5)
        trg_s = np.argmax(np.isclose(self.range_steer, trg_s, atol=self.toll) >.5)

        #trg = [trg_m, trg_s]
        trg = int(trg_m + 5*trg_s)
        
        return inp, trg
