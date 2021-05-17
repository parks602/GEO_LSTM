import numpy as np
import pandas as pd
import sys
import random
from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from utils import isExists

class CustomDataset(Dataset):
    def __init__(self, xdata, ydata, dlist):
        nt, ns, nw, nf = xdata.shape
        xdata = xdata.reshape((nt * ns, nw, nf))
        ydata = ydata.flatten()
        dlist = dlist.flatten()
        self._xdata = torch.from_numpy(xdata).type(torch.FloatTensor)
        self._ydata = torch.from_numpy(ydata).type(torch.FloatTensor)
        self._dlist = dlist

    def __len__(self):
        return self._xdata.shape[0]

    def __getitem__(self, idx):
        return self._xdata[idx], self._ydata[idx], self._dlist[idx]

class MinMaxScaler:
    def __init__(self, var, pad=3):
        self.pad = pad
        self.var = var
        if self.var == "T3H":
            self.min=-38-pad
            self.max=43+pad
        elif self.var == "REH":
            self.min=0
            self.max=100
        elif self.var == "LAT":
            self.min=31
            self.max=44
        elif self.var == "LON":
            self.min=123
            self.max=133
        elif self.var == "HGT":
            self.min=0
            self.max=2600
        else:
            print("Error : Check Variable Option")
            sys.exit(1)

    def fit(self, dt):
        self.min= np.min(dt) 
        self.max= np.max(dt)

    def transform(self, dt):
        return (dt - self.min) / (self.max - self.min)

    def rev_transform(self, dt):
        return dt * (self.max - self.min) + self.min

def splitDataset(args, dataset):
    ntrain = int(args.split_frac * len(dataset))
    nvalid = len(dataset) - ntrain
    train_dataset, valid_dataset = random_split(dataset, [ntrain, nvalid])
    return train_dataset, valid_dataset

def splitByStation(x, y, d, split_frac):
    ndata = x.shape[0]
    ntrain = int(ndata * split_frac)
    print("ntrain : (%d/%d)" %(ntrain,ndata))
    print("nvalid : (%d/%d)" %(ndata-ntrain,ndata))
    train_idx = sorted(random.sample(range(ndata), ntrain))
    valid_idx = list(set(range(ndata)) - set(train_idx))

    train, valid = {}, {}
    train['x'], valid['x'] = x[train_idx], x[valid_idx]
    train['y'], valid['y'] = y[train_idx], y[valid_idx]
    train['d'], valid['d'] = d[train_idx], d[valid_idx]
    print("train shape : ", train['x'].shape)
    return train, valid

def getTestDataset(args, scaler=None):
    preproc = data_preprocess(args.var, args.dataf, args.infof, args.targf, args.windowSize, args.nearestK, scaler)
    x, _, d = preproc.getData()
    nt, ns, nw, nf = x.shape
    x = x.reshape((nt * ns, nw, nf))
    x = torch.from_numpy(x).type(torch.FloatTensor)
    d = d.flatten()
    return x, d

class MyDataModule(LightningDataModule):
    def __init__(self, args, scaler=None):
        ### for data_preprocess
        self.var       = args.var
        self.datafile  = args.dataf
        self.infofile  = args.infof
        self.targetfile= args.targf
        self.WSize     = args.windowSize
        self.nearK     = args.nearestK
        self.split_frac= args.split_frac  
        self.batchSize = args.batchSize
        self.sdate     = args.sdate
        self.edate     = args.edate
        self.scaler    = scaler
      
    def prepare_data(self):
        preproc = data_preprocess(self.var, 
                                  self.datafile, self.infofile, self.targetfile, 
                                  self.sdate, self.edate,
                                  self.WSize, self.nearK, self.scaler)
        x, y, d = preproc.getData()
        # split dataset
        train, valid = splitByStation(x, y, d, self.split_frac)
        self.train_dataset = CustomDataset(train['x'], train['y'], train['d'])
        self.valid_dataset = CustomDataset(valid['x'], valid['y'], valid['d'])

    def setup(self):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        # things to do on every accelerator in distributed mode
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=False, 
                          batch_size=self.batchSize, num_workers=20)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, shuffle=False, 
                          batch_size=2, num_workers=20)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # Data specific
        parser = ArgumentParser(parents=[parent_parser],add_help=False)
        parser.add_argument('--dataf', help='spatio-temporal data file in csv', required=True)
        parser.add_argument('--infof', help='station info file in csv', required=True)
        parser.add_argument('--targf', help='target info file in csv', required=True)
        parser.add_argument('--sdate', type=int, help='start datetime[yyyymmddhh]', required=True)
        parser.add_argument('--edate', type=int, help='end datetime[yyyymmddhh]', required=True)
        parser.add_argument('--windowSize', type=int, default=8, help='rolling window size for timeseries')
        parser.add_argument('--nearestK', type=int, default=10, help='K-nearest station')
        parser.add_argument('--split_frac', type=float, default=0.7, help='ratio of train set to whole dataset')
        parser.add_argument('--batchSize', type=int, default=16, help='input batch size')

        return parser

class data_preprocess:
    def __init__(self, var: str, datafile: str, locfile: str, targetfile: str, 
                 sdate: str, edate: str, nwindow: int, K: int, scaler=None):
        # read data
        if isExists(datafile) and isExists(locfile) and isExists(targetfile):
            df    = pd.read_csv(f"{datafile}",index_col=False)            # ntime, nloc
            loc   = pd.read_csv(f"{locfile}",index_col=False)             # nloc,    info (stnid, lon, lat, ...)
            tloc  = pd.read_csv(f"{targetfile}",index_col=False)          # ntarget, info (lon, lat)
        else:
            sys.exit(1)
        assert int(sdate) <= int(edate), print("Error : [ sdate > edate ]")
        # preprocess
        stnid = list(map(str, loc['stnid'].tolist()))
        df['datetime'] = pd.to_datetime(df['datetime'])
        sdate_dt = pd.to_datetime(sdate, format="%Y%m%d%H")
        edate_dt = pd.to_datetime(edate, format="%Y%m%d%H")
        df = df[ (df['datetime'] >= sdate_dt) & (df['datetime'] <= edate_dt) ]
        assert df.empty == False, f"No Data between {sdate} and {edate}"
        df['datetime'] = df['datetime'].dt.strftime("%Y%m%d%H")
        df.set_index(['datetime'], inplace=True)
        df = df[ stnid ]        # re-order

        self._nloc = len(loc)
        if K > self._nloc - 1: 
            print("K is greater than number of station : K = nloc - 1")
            self._K = self._nloc - 1
        else:
            self._K  = K        # Nearest K statation 
        self._nwindow= nwindow  # for rolling window
        self._dlist  = df.index.values
        self._data   = df.values
        self._loc    = loc[['lat','lon']].values
        self._tloc   = tloc[['lat','lon']].values     # target location

        # Scaling
        if scaler is not None:
            scaler_lat = MinMaxScaler("LAT")
            self._loc[:,0] = scaler_lat.transform(self._loc[:,0])
            scaler_lon = MinMaxScaler("LON")
            self._loc[:,1] = scaler_lon.transform(self._loc[:,1])
            self._data = scaler.transform(self._data)

        self._RollingWindow()
        self._GeoLayer()

    def _RollingWindow(self):
        sampl= []
        nx, ny = self._data.shape
        for rw in range(self._nwindow, nx+1):
            si = rw - self._nwindow 
            series = np.transpose(self._data[si:rw, :])  # nwindow, nloc > nloc, nwindow
            sampl.append(series)
        xdata = np.array(sampl).reshape(-1, self._nloc, self._nwindow)  # nsample(ntimeseries - nwindow + 1), nloc, nwindow
        self._xdata = np.transpose(xdata, [0,2,1])  # nsample, nwindow, nloc
        self._ydata = np.transpose(self._data[self._nwindow-1:,:])              # nloc(target), nsample
        self._dlist = np.tile(self._dlist[self._nwindow-1:],(self._nloc,1))     # nloc(target), nsample

    def _GeoLayer(self):
        ntarget = self._tloc.shape[0]
        #R = np.zeros((ntarget,self._nloc,2))
        #D = np.zeros((ntarget,self._nloc))
        dset = []
        for i in range(ntarget):
            RLatLng = self._tloc[i] - self._loc[:]
            Dist    = RLatLng[:,0]**2. + RLatLng[:,1]**2.
            # make 0 to inf
            Dist    = np.where(Dist == 0, np.inf, Dist)
            # Rank
            idx     = np.argpartition(Dist, self._K)[:self._K]
            # A_g function
            R_A     = RLatLng[idx,:]
            #print('0:',R_A.shape)
            # g_in x R_A
            '''
            1. extract k-nearest xdata from target point
            2. copy R_A for shape of xdata
            3. reshape R_A :  (nsample, nwindow, K, 2[lat,lng]) >  (nsample, nwindow, K*2)
            4. concatenate xdata & R_A 
            5. stack data for ntarget
            '''
            xdata = self._xdata[:,:,idx]   # nsample, nwindlw, K
            #print('1:',xdata.shape)
            R_A   = np.broadcast_to(R_A, (xdata.shape[0],self._nwindow)+R_A.shape) 
            #print('2:',R_A.shape)
            R_A   = R_A.reshape((xdata.shape[0], xdata.shape[1], -1))  
            #print('3:',R_A.shape)
            dset.append(np.concatenate((xdata, R_A),axis=2))  # nsample, nwindow, nfeature(K x 3[RLat,Rlng,S_t)
            #print('4:',np.concatenate((xdata, R_A),axis=2).shape)
        # update xdata
        self._xdata = np.array(dset)  # ntarget, nsample, nwindow, nfeature(K x 3)
        #print('5:',self._xdata.shape)

    def getData(self):
        #nt, ns, nw, nf = self._xdata.shape
        #self._xdata = self._xdata.reshape((nt * ns, nw, nf))
        #self._ydata = self._ydata.flatten()
        #self._dlist = self._dlist.flatten()
        '''
        xdata : (ntarget, nsample for time, nseq, nfeature) 
        ydata : (ntarget, nsample) 
        dlist : (ntarget, nsample) 
        '''
        return self._xdata, self._ydata, self._dlist
