import h5py
import numpy as np
import time
import re
import sys
import warnings
import torch
from torch.utils.data import DataLoader, Dataset
from dataGenerator import *


class dataGenerator(dataGeneratorPrototip):

    def __init__(self, **kwargs):

        dtFold = '/home/guglielmo/dataset/car/Mr_Blue/'
        
        #self.datafiles  = ["caffe_direct_local_sidewalks_05Dec16_15h03m35s_Mr_Blue.hdf5", "direct_local_17Dec16_16h29m16s_Mr_Blue.hdf5", "direct_home_06Dec16_16h01m42s_Mr_Blue.hdf5", "direct_31Aug2016_Mr_Blue_sidewalks_2.hdf5", "direct_31Aug2016_Mr_Blue_sidewalks_1.hdf5", "direct_home_06Dec16_08h10m47s_Mr_Blue.hdf5", "direct_racing_Tilden_27Nov16_11h08m39s_Mr_Blue.hdf5", "direct_racing_Tilden_27Nov16_12h20m21s_Mr_Blue.hdf5", "direct_racing_Tilden_27Nov16_13h28m00s_Mr_Blue.hdf5", "direct_racing_Tilden_27Nov16_10h20m51s_Mr_Blue.hdf5", "direct_racing_Tilden_27Nov16_10h41m05s_Mr_Blue.hdf5"]

        #self.datafiles  = ["direct_racing_Tilden_27Nov16_10h20m51s_Mr_Blue.hdf5"]
        self.datafiles  = ["caffe_z2_color_direct_Smyth_tape_single_transmitter_11Feb17_14h06m16s_Mr_Blue.hdf5", "caffe_z2_color_direct_Smyth_tape_single_transmitter_11Feb17_15h07m11s_Mr_Blue.hdf5"]

        

        self.datafiles  = [dtFold + dt for dt in self.datafiles]

        self.batch_size = 100

        self.n_test_samples  = 2000#30000
        self.n_valid_samples = 2000#30000
        self.n_train_samples = 15000
        
        dtm  = datasetManagerCar(self.datafiles)
        print "Number of total data present in the file:", dtm.n_data
        dtm.createDataset(n_samples = self.n_test_samples, name="test", datasetType=datasetCar)
        dtm.createDataset(n_samples = self.n_valid_samples, name="valid", datasetType=datasetCar)
        dtm.createDataset(n_samples = self.n_train_samples, name="train", datasetType=datasetCar)
        
        self.testCar  = DataLoader(dtm.dts['test'], batch_size=self.batch_size, shuffle=False, num_workers=1)
        self.validCar = DataLoader(dtm.dts['valid'], batch_size=self.batch_size, shuffle=False, num_workers=1)
        self.trainCar = DataLoader(dtm.dts['train'], batch_size=self.batch_size, shuffle=True, num_workers=1)
        
    
    def returnGen(self,sets,**kwargs):
        if sets == "test":
            return self.testCar
        if sets == "valid":
            return self.validCar
        if sets == "train":
            return self.trainCar


    def returnConfig(self,**kwargs):
        conf = {'datafiles': self.datafiles, 'n_test_samples': self.n_test_samples, 'n_valid_samples': self.n_valid_samples, 'n_train_samples': self.n_train_samples, 'batch_sz': self.batch_size}
        return conf




    
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


    
class datasetManager(object):

    def __init__(self, file_list):

        self.file_lst     = file_list
        self.dts          = {}
        
        self.n_data       = self._n_data()
        self.unused_ids   = range(self.n_data)
        
        self.preprocess()

        
    def createDataset(self, n_samples, **kwargs):

        if n_samples > len(self.unused_ids):
            warnings.warn("Then number of sample exceed the number of data left in the dataset.")
            
        name    = kwargs['name']
        dataset = kwargs['datasetType']
        self.dts[name]  = dataset(n_samples, unused_ids=self.unused_ids, **kwargs)
        to_remove       = self.dts[name].ids
        self.unused_ids = self._remove_ids(to_remove)
        
    def _remove_ids(self,to_remove):
        unused_ids = [id_ for id_ in self.unused_ids if id_ not in to_remove]
        return unused_ids

    def _n_data(self):
        return 0
        

    def preprocess(self):
        pass
        


class dataset(Dataset):

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

    def __init__(self, file_list):
        super(datasetManagerCar, self).__init__(file_list)
        self.fields = ['left','left_timestamp', 'right', 'motor', 'state', 'steer']
        self._addrs = self._dataAddressList()
        
        
    def _openFile(self):
        self.fh5_lst = []
        for f in self.file_lst:
            self.fh5_lst.append(h5py.File(f, "r"))

            
    def closeFile(self):
        for f in self.fh5_lst:
            f.close()

            
    def _n_data(self):
        self._openFile()
        n_data = 0
        for f in self.fh5_lst:
            segs = f['segments'].keys()
            segs.sort(key=natural_keys)
            for seg in segs:
                n_data += f['segments/' + seg + '/left'].shape[0]
        return n_data

    
    def createDataset(self, n_samples, **kwargs):
        super(datasetManagerCar, self).createDataset(n_samples, **kwargs)
        self.dts[kwargs['name']].fh5_lst    = self.fh5_lst
        self.dts[kwargs['name']]._addrs     = self._addrs

    def _dataAddressList(self):
        addLst = []
        cnt_f = 0
        for f in self.fh5_lst:
            segs = f['segments'].keys()
            segs.sort(key=natural_keys)

            cnt_s = 0
            for seg in segs:
                n_data = f['segments/' + seg + '/motor'].shape[0]
                
                for cnt_n in range(n_data):
                    label = [np.int16(cnt_f), np.int16(cnt_s), np.int16(cnt_n)]
                    addLst.append(label)
                cnt_s += 1
            cnt_f += 1            
        return addLst
        
                
class datasetCar(dataset):

    def __init__(self, n_samples, **kwargs):
        super(datasetCar, self).__init__(n_samples, **kwargs)
        self.motor_max =  np.asarray(70, dtype=np.float32)
        self.motor_min =  np.asarray(50, dtype=np.float32)
        self.steer_max =  np.asarray(100, dtype=np.float32)
        self.steer_min =  np.asarray(-1.5, dtype=np.float32)

        self.n_classes   = 5
        self.range_motor = np.linspace(0,1,self.n_classes, dtype=np.float32)
        self.range_steer = np.linspace(0,1,self.n_classes, dtype=np.float32)
        self.toll        = 1./(2.*(self.n_classes-1.))
        
                               
    def get_data(self,label, name):
            data = self.fh5_lst[label[0]]['segments/' + str(label[1]) + '/' + name][label[2]]
            return np.asarray(data, dtype=np.float32)                   

    def __getitem__(self, idx):

        idx = self.ids[idx]

        if idx == 0:
            label_    = self._addrs[idx]
            label_prv = self._addrs[idx]
        else:
            label_    = self._addrs[idx]
            label_prv = self._addrs[idx-1]
            
        inp = []
        inp.append(self.get_data(label_, 'left'))
        inp.append(self.get_data(label_prv, 'left'))
        inp.append(self.get_data(label_, 'right'))
        inp.append(self.get_data(label_prv, 'right'))
        inp = np.concatenate(inp, axis=2)/255.
        inp = np.rollaxis(inp, 2)

        inp = torch.from_numpy(inp)
       
        trg_m = (self.get_data(label_, 'motor') - self.motor_min)/(self.motor_max-self.motor_min)
        trg_s = (self.get_data(label_, 'steer') - self.steer_min)/(self.steer_max-self.steer_min)

        trg_m = np.argmax(np.isclose(self.range_motor, trg_m, atol=self.toll) >.5)
        trg_s = np.argmax(np.isclose(self.range_steer, trg_s, atol=self.toll) >.5)

        #trg = [trg_m, trg_s]
        trg = int(trg_m + 5*trg_s)
        
        return inp, trg

