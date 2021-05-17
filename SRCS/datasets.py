from scipy.io import FortranFile
from datetime import timedelta, datetime
import argparse
import os, sys, json
import pandas as pd
import numpy as np
from utils import isExists
from sys import argv

class LMODEL:
    def __init__(self):
        self.null = -999.
        self._data = []
        self._dlist= []

    def read_data(self, fname, dtime):
        if isExists(fname):
            dt = np.load(fname).flatten()
            dt = np.round(dt,1)
        else:
            dt = np.full(self.data[0].shape, self.null)
        self._data.append(dt)
        kst_time = datetime.strptime(dtime, "%Y%m%d%H") + timedelta(hours=9) ## UTC to KST
        self._dlist.append(kst_time.strftime("%Y%m%d%H"))

    def get_data(self):
        return np.array(self._data), np.array(self._dlist)

def make_ldaps_data(args):
    start = datetime.strptime(args.sdate,"%Y%m%d%H")
    end   = datetime.strptime(args.edate,"%Y%m%d%H")
    ldaps = LMODEL()
    now   = start
    while now <= end:
        dtime = now.strftime("%Y%m%d%H")
        fname = f"{args.ipath}/{args.var}_{dtime}.npy"
        ldaps.read_data(fname, dtime)
        now += timedelta(hours=3)
    data, dlist = ldaps.get_data()
    np.save(f"{args.dain}/{args.var}_ldaps.npy", data)
    np.save(f"{args.dain}/{args.var}_dlist.npy", dlist)

def sampling_ldaps(args, only_land=False):
    if args.make_raw:
        make_ldaps_data(args)

    ldata = f"{args.dain}/{args.var}_ldaps.npy"
    flist = f"{args.dain}/{args.var}_dlist.npy"
    fgrid = args.grid1km
    fdem  = args.dem1km
    egrid = args.egrid   ## excluded grid for ldaps

    if isExists(ldata) and isExists(flist) and isExists(fgrid) \
       and isExists(fdem) and isExists(egrid): 
        data  = np.load(ldata)
        dlist = np.load(flist)
        egrid = np.load(egrid).flatten()  
        ginfo = np.load(fgrid).reshape([-1,2])
        ghgt  = np.load(fdem).flatten()
    else:
        sys.exit(1)

    print(f"Samples = {args.nsample}, only_land = {only_land}")
    # remove null points
    idx_ = np.where(egrid == 0)  # 1 == null
    data = np.squeeze(data[:,idx_])
    ghgt = np.squeeze(ghgt[idx_])
    ginfo= np.squeeze(ginfo[idx_,:])
    if only_land:
        land_idx = np.where(ghgt > 0)
        ghgt = ghgt[land_idx]
        data = np.squeeze(data[:,land_idx])
        ginfo= np.squeeze(ginfo[land_idx, :])
    nx, ny = data.shape
    # sampling
    idx = np.random.choice(range(ny), args.nsample, replace=False) # replace=False : not duplicated
    sub_sample = data[:,idx]
    sub_info   = ginfo[idx,:]  # nsample, nfeature
    sub_hgt    = ghgt[idx].reshape([-1,1])
    # dataframe
    sample_idx = args.nsample+50000
    grid_num   = np.arange(sample_idx, sample_idx+sub_sample.shape[1]).reshape([-1,1])
    df = pd.DataFrame(sub_sample, columns=range(sample_idx, sample_idx+sub_sample.shape[1]))
    info = pd.DataFrame(np.concatenate((grid_num, sub_info, sub_hgt),axis=1), columns=['stnid','lon','lat','hgt'])
    info['stnid'] = info['stnid'].astype(int)
    # imputation for datetime
    df[ df < -990. ] = np.nan
    df = df.fillna(method='bfill')
    df['datetime'] = pd.to_datetime(dlist.tolist(),format="%Y%m%d%H")
    # save
    df.to_csv(f"{args.ldata}",index=False)
    info.to_csv(f"{args.linfo}",index=False)
    return 

def merge_model_obs(args):
    fmodel = f"{args.ldata}"
    minfo  = f"{args.linfo}"
    fobs   = f"{args.odata}"
    oinfo  = f"{args.oinfo}"

    if isExists(fmodel) and isExists(fobs) and isExists(minfo) and isExists(oinfo):
        obs   = pd.read_csv(fobs)
        model = pd.read_csv(fmodel) 
        m_info= pd.read_csv(minfo)
        o_info= pd.read_csv(oinfo)
    else:
        sys.exit(1)
    # Merge
    print("Merge Model & OBS Data")
    print(model.head(5))
    print(obs.head(5))
    df = pd.merge(model, obs, on='datetime', how='left')
    info = o_info.append(m_info, ignore_index=True)
    # re-order
    cols = df.columns.tolist()
    cols.remove('datetime')
    ordered_cols = ['datetime'] + cols
    df = df[ ordered_cols ]
    # Save
    df.to_csv(f"{args.dain}/{args.var}_model+obs.csv",index=False)
    info.to_csv(f"{args.dain}/{args.var}_model+obs_info.csv", index=False)

class OBS:
    def __init__(self):
        self.null = -999.
        self.n    = 0
        self.oloc = []
        self.opnt = []
        self.hgt  = []
        self.stn  = []
        self.data = None

    def readOBS(self, stn_fname, obs_fname):
        if not isExists(stn_fname): return
        if not isExists(obs_fname): return

        stn_dt = pd.read_csv(stn_fname)
        ### Get OBS
        with FortranFile(obs_fname,'r') as f:
            info = f.read_ints(np.int32)
            stnlist = f.read_ints(np.int32)
            data = f.read_reals(np.float32)
            data = np.reshape(data, info[:7:-1]) ### nstn, nyear, nmonth, nday, nhour
        data = np.transpose(data)[:,:,:,:,:24]  # 24hour
        if self.data is None:
            self.data = data
        else:
            try:
                self.data = np.concatenate((self.data, data), axis=0)
            except:
                print("Error : Check OBS data :: %s" %(obs_fname))
                sys.exit(1)
        ### Get STN
        stnlist = stnlist.tolist()
        for stnid in stnlist:
            temp = stn_dt[ stn_dt['stnid'] == stnid ]
            if temp.empty:
                data[stnlist.index(stnid),...] = self.null
                #print("Skip : %d" %(stnid))
                self.oloc.append([self.null,self.null])
                self.opnt.append([self.null,self.null])
                self.hgt.append(self.null)
                self.stn.append(self.null)
                continue
            self.oloc.append(temp[['longitude','latitude']].values.squeeze().tolist())
            #self.opnt.append(temp[['ixx','iyy']].values.squeeze().tolist())
            self.hgt.append(temp[['altitude']].values.squeeze().tolist())
            self.stn.append(str(stnid))
        self.n += 1

    def removeNull(self):
        oloc = np.array(self.oloc)
        #opnt = np.array(self.opnt,dtype=int)
        hgt  = np.array(self.hgt)
        stn  = np.array(self.stn)

        chk1 = oloc[:,0]
        idx = np.where(chk1 != self.null)
        self.data = np.squeeze(self.data[idx])
        self.oloc = np.squeeze(oloc[idx,:])
        #self.opnt = np.squeeze(opnt[idx,:])
        self.stn  = np.squeeze(stn[idx])
        self.hgt  = np.squeeze(hgt[idx]) # m

    def getData(self, utctime):
        obs_time = utctime + timedelta(hours=9)
        iy, im, id, ih = obs_time.year - 2016, obs_time.month - 1, obs_time.day - 1, obs_time.hour
        res = self.data[:,iy,im,id,ih]
        idx = np.where(res != self.null)
        #return res[idx], np.squeeze(self.oloc[idx,:]), np.squeeze(self.opnt[idx,:]), np.squeeze(self.hgt[idx])
        return res[idx], np.squeeze(self.oloc[idx,:]), np.squeeze(self.hgt[idx])

def make_obs(args):
    obs_cls = OBS()
    start = datetime.strptime(args.sdate,"%Y%m%d%H")
    end   = datetime.strptime(args.edate,"%Y%m%d%H")
    for pre in ['kma', 'kfs', 'rda']:
        info= "{}/{}_aws_info.csv".format(args.ipath, pre)
        obs_= "{}/{}_obs_{}.{}-{}".format(args.ipath, pre, args.var, args.sdate, args.edate)
        obs_cls.readOBS(info, obs_)
    ### Buoy
    info= "{}/buoy_info.csv".format(args.ipath)
    obs_= "{}/buoy_obs_{}.{}-{}".format(args.ipath,args.var,args.sdate,args.edate)
    obs_cls.readOBS(info, obs_)
    obs_cls.removeNull()
    ### Get Data
    data = obs_cls.data
    oloc = obs_cls.oloc
    hgt  = obs_cls.hgt
    stn  = obs_cls.stn
    ### make spatio-temporal data
    data = np.transpose(np.reshape(data, [data.shape[0], data.shape[1]* data.shape[2] * data.shape[3] * data.shape[4]]))
    dt = pd.DataFrame(data, columns=stn)
    dt[ dt < -990 ] = np.nan        ## input nan
    dt = dt.dropna(how='all', axis=0)  ## remove fake datetime
    dt = dt.fillna(method='ffill')  ## imputation
    dt = dt.dropna(axis=1)          ## remove nan columns
    dt_idx = pd.DatetimeIndex(pd.date_range(start=start, end=end, freq='60min'))
    dt.index = dt_idx
    ### make Station Info
    stn = np.reshape(np.array(stn), (len(stn),1))
    hgt = np.reshape(np.array(hgt), (len(hgt),1))
    stninfo = pd.DataFrame(np.concatenate((stn, oloc, hgt),axis=1), columns=['stnid','lon','lat', 'hgt'])
    stninfo = stninfo[stninfo['stnid'].isin(dt.columns)]
    ### Save
    stninfo.to_csv(f"{args.oinfo}",index=False)
    dt.to_csv(f"{args.odata}",index_label='datetime')

def make_mesh_dataframe(args):
    fgrid = args.grid1km
    fdem  = args.dem1km
    if isExists(fgrid) and isExists(fdem):
        grid = np.load(fgrid)
        dem  = np.load(fdem)
    dem = np.expand_dims(dem,axis=2)
    dt  = np.concatenate((grid, dem), axis=2)
    dt_reshape = np.reshape(dt, [dt.shape[0]*dt.shape[1], dt.shape[2]])
    df = pd.DataFrame(dt_reshape, columns=['lon','lat','hgt'])
    print(df.head(5))
    df.to_csv(f"{args.dain}/mesh_1km.csv",index=False)

def set_arguments():
    parser = argparse.ArgumentParser(description="Make Dataset")
    action_choices=['obs','ldps','ldps_raw','mesh']
    # mode
    parser.add_argument('--mode', choices=action_choices, default=action_choices[1], help='set mode')
    parser.add_argument('--make_raw', action='store_true', help='make RAW LDPS between sdate and edate')
    parser.add_argument('--obs', action='store_true', help='make OBS')
    parser.add_argument('--mesh', action='store_true', help='make mesh')
    parser.add_argument('--var', type=str, default="T3H", help='variable T3H or REH')
    parser.add_argument('--dain', type=str, default='../DAIN', help='location of data saved')
    parser.add_argument('--grid1km', type=str, help='location of grid 1km info file', required=True)
    parser.add_argument('--dem1km', type=str, help='location of grid 1km info file', required=True)
    
    # ldps_raw
    parser.add_argument('--sdate', type=str, help='start datetime[YYYYMMDDHH]',required=((action_choices[2] or action_choices[0]) in argv))
    parser.add_argument('--edate', type=str, help='end datetime[YYYYMMDDHH]',required=((action_choices[2] or action_choices[0]) in argv))
    parser.add_argument('--ipath', type=str, help='location of data path',required=((action_choices[2] or action_choices[0]) in argv))
    # ldps
    parser.add_argument('--ldata', type=str, help='location of ldaps sampling file',required=(action_choices[1] in argv))
    parser.add_argument('--linfo', type=str, help='location of ldaps info file',required=(action_choices[1] in argv))
    parser.add_argument('--odata', type=str, help='location of obs file',required=(action_choices[1] in argv))
    parser.add_argument('--oinfo', type=str, help='location of obs info file',required=(action_choices[1] in argv))
    parser.add_argument('--egrid', type=str, help='grid data for excluded ldps grid',required=(action_choices[1] in argv))
    parser.add_argument('--nsample', type=int, default=10000, help='number of sample for dataset', required=(action_choices[1] in argv))

    args = parser.parse_args()
    print(args)
    return args

def main():
    args = set_arguments()
    if 'ldps' in args.mode:
        if args.mode == 'ldps_raw':
            make_ldaps_data(args)
        sampling_ldaps(args)
        merge_model_obs(args)
    elif args.mode == "obs":
        make_obs(args)
    elif args.mode == "mesh":
        make_mesh_dataframe(args)

if __name__ == "__main__":
    main()
