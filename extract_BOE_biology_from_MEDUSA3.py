import numpy as np
from cmocean import cm
import cartopy as cp
import cartopy.crs as ccrs
import netCDF4 as nc
import matplotlib.pyplot as plt
import xarray as xr
import sys
sys.path.append('/gpfs/home/mep22dku/scratch/SOZONE')
#list of models
sys.path.append('/gpfs/home/mep22dku/scratch/SOZONE/UTILS')
import lom
import utils as ut
from scipy.interpolate import interp1d
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')
import cartopy.feature as cfeature
from importlib import reload
import matplotlib.path as mpath
import glob
import pickle
import pandas as pd
import seawater
import time
from scipy.interpolate import interp1d
sys.path.append('/gpfs/home/mep22dku/scratch/SOZONE/mocsy-master')
import mocsy
import gsw
#put python script here
##\def get_weighted(ts,bgc, diadaux):
  
def get_weighted(bgc):
    
    tmask = xr.open_dataset(f'/gpfs/data/greenocean/software/resources/MEDUSA/mesh_mask.nc')
    tmask_surf = tmask.tmask[0,0,:,:] * tmask.e1t[0,:,:] * tmask.e2t[0,:,:]

    PHD_mean = bgc.PHD.isel(y=slice(0,114),deptht = 0).weighted(tmask_surf.isel(y=slice(0,114))).mean(dim = ['x','y'])

   
    return PHD_mean

## surface diatom concentration
def get_grads_fer(bgc, verbose = True):
    
    grad_FER_0_100_ar = np.zeros([12])
    grad_FER_0_200_ar = np.zeros([12])
    grad_FER_100_200_ar = np.zeros([12])
    grad_avg2_ar = np.zeros([12])
    FER_1000_minus_0_ar = np.zeros([12])
    FER_2000_minus_0_ar = np.zeros([12])

    for m in range(12):
        FER = bgc.FER.isel(y=slice(0,114)).mean(dim = ['x','y']) * 1000 #convert mmol/m3 to nmol/L
        FER_0 = FER[m,0]
        if np.isnan(FER_0):
            pass

        else:

            depth_0 = bgc.deptht[0]
            depths_first100 = bgc.deptht[0:25]
            FERs_first100 = FER[m,0:25]
            f = interp1d(depths_first100, FERs_first100, kind='linear')
            FER_100 = f(100)

            depths_second100 = bgc.deptht[24:32]
            FERs_second100 = FER[m,24:32]
            f = interp1d(depths_second100, FERs_second100, kind='linear')
            FER_200 = f(200)

            grad_FER_0_100 = ((FER_100-FER_0)/(100-depth_0)).values
            grad_FER_0_200 = ((FER_200-FER_0)/(200-depth_0)).values
            
            grad_FER_100_200 = ((FER_200-FER_100)/(200-100))
            grad_avg2 = (grad_FER_100_200+grad_FER_0_100)/2
            if verbose:
                print(grad_FER_0_100)
                print(grad_FER_0_200)
                print(grad_FER_100_200)
                print(grad_avg2)

            #bgc.deptht[44:50]
            depths_1000 = bgc.deptht[44:50]
            FERs_1000 = FER[m,44:50]
            f = interp1d(depths_1000, FERs_1000, kind='linear')
            FER_1000 = f(1000)    
            FER_1000_minus_0 = (FER_1000-FER_0).values
            
            depths_2000 = bgc.deptht[53:57]
            #print(depths_2000)
            FERs_2000 = FER[m,53:57]
            f = interp1d(depths_2000, FERs_2000, kind='linear')
            FER_2000 = f(2000)    
            FER_2000_minus_0 = (FER_2000-FER_0).values

            if verbose:
                plt.plot(FER[m,0:32],-bgc.deptht[0:32])
                plt.plot(FER_100, -100, 'ro')
                plt.plot(FER_200, -200, 'r+')
                
                print(m)
                print(f'grad_FER_0_100 {grad_FER_0_100}')
                print(f'grad_FER_0_200 {grad_FER_0_200}')
                print(f'grad_FER_100_200 {grad_FER_100_200}')
                print(f'grad_avg2 {grad_avg2}')
                print(f'FER_1000_minus_0 {FER_1000_minus_0}')
                print(f'FER_1000_minus_0 {FER_2000_minus_0}')
                print('---')

            grad_FER_0_100_ar[m] = grad_FER_0_100
            grad_FER_0_200_ar[m] = grad_FER_0_200
            grad_FER_100_200_ar[m] = grad_FER_100_200
            grad_avg2_ar[m] = grad_avg2
            FER_1000_minus_0_ar[m] = FER_1000_minus_0
            FER_2000_minus_0_ar[m] = FER_2000_minus_0
            
    return grad_FER_0_100_ar, grad_FER_0_200_ar, grad_FER_100_200_ar, \
grad_avg2_ar, FER_1000_minus_0_ar, FER_2000_minus_0_ar


def make_yearlist(yr, scen):
    
    tdir = '/gpfs/data/greenocean/software/resources/MEDUSA/PROC2/'

    if yr < 2015:
        runid = 'bc370'
    else:
        if scen == '1A':
            runid = 'be682'
        if scen == '1B':
            runid = 'ce417'

    bgc = xr.open_dataset(f'{tdir}medusa_{runid}_1y_{yr}_ptrc-T-FERPHD.nc')
    
    return bgc #, ts, diadaux

def save_biol_param(scen,yr):
    
    bgc = make_yearlist(yr, scen)
    
    savenam = f'./data/biol-param-medusa_scen-{scen}-{yr}.nc'
    times = pd.date_range(f"{yr}/01/01",f"{yr}/12/11",freq='MS')#,closed='left')

    data_vars = {
                ### bgc gradients
        'diatconc': (['time_counter', 'quantity'], np.zeros([12,4]),
                {'units': ''}),
        'dmudFe': (['time_counter', 'quantity'], np.zeros([12,4]),
                {'units': ''}),

                ### carbonate system variables
        'grad_dFev': (['time_counter', 'quantity'], np.zeros([12,4]),
                {'units': ''}),

        }

    coords = {'time_counter': (['time_counter'], times),
            'quantity': (['mean', 'stdev', 'max', 'min']),

    }
    # define global attributes
    attrs = {'made in':'scratch/BOE-SOcarbon/extract_BOE_biology_from_MEDUSA.py',
    'desc': ''
    }
    ds = xr.Dataset(data_vars=data_vars,
    coords=coords,
    attrs=attrs)

    PHD_mean = get_weighted(bgc)
    
    grad_FER_0_100_ar, grad_FER_0_200_ar, grad_FER_100_200_ar, \
    grad_avg2_ar, FER_1000_minus_0_ar, FER_2000_minus_0_ar = get_grads_fer(bgc, False)

    ds['diatconc'][:,0] = PHD_mean.values 
    ds['dmudFe'][:,0] = grad_avg2_ar; 
    ds['grad_dFev'][:,0] = FER_1000_minus_0_ar; 


    print(savenam)
    ds.to_netcdf(savenam)

    return ds
       

for yr in range(2050,2100):
    ds = save_biol_param('1A',yr)
    ds = save_biol_param('1B',yr)
    
