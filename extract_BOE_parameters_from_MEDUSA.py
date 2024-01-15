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
  
def get_weighted(ts,bgc, diadaux):
    
    tmask = xr.open_dataset(f'/gpfs/data/greenocean/software/resources/MEDUSA/mesh_mask.nc')
    tmask_surf = tmask.tmask[0,0,:,:] * tmask.e1t[0,:,:] * tmask.e2t[0,:,:]

    T_mean = ts.votemper.isel(y=slice(0,114),deptht = 0).weighted(tmask_surf.isel(y=slice(0,114))).mean(dim = ['x','y'])
    DIC_mean = bgc.DIC.isel(y=slice(0,114),deptht = 0).weighted(tmask_surf.isel(y=slice(0,114))).mean(dim = ['x','y'])
    ALK_mean = bgc.ALK.isel(y=slice(0,114),deptht = 0).weighted(tmask_surf.isel(y=slice(0,114))).mean(dim = ['x','y'])
    OCNPCO2_mean = diadaux.OCN_PCO2.isel(y=slice(0,114)).weighted(tmask_surf.isel(y=slice(0,114))).mean(dim = ['x','y'])
    OCNDPCO2_mean = diadaux.OCN_DPCO2.isel(y=slice(0,114)).weighted(tmask_surf.isel(y=slice(0,114))).mean(dim = ['x','y'])
    CO2FLUX_mean = diadaux.CO2FLUX.isel(y=slice(0,114)).weighted(tmask_surf.isel(y=slice(0,114))).mean(dim = ['x','y'])
   
    return T_mean, DIC_mean, ALK_mean, OCNPCO2_mean, OCNDPCO2_mean, CO2FLUX_mean

def get_revelle(ts,bgc):
    
        
    tmask = xr.open_dataset(f'/gpfs/data/greenocean/software/resources/MEDUSA/mesh_mask.nc')
    tmask_surf = tmask.tmask[0,0,:,:] * tmask.e1t[0,:,:] * tmask.e2t[0,:,:]
    
    DIC = bgc.DIC.isel(y=slice(0,114),deptht = 0).values
    ALK = bgc.ALK.isel(y=slice(0,114),deptht = 0).values
    votemper = ts.votemper.isel(y=slice(0,114),deptht = 0).values
    vosaline = ts.vosaline.isel(y=slice(0,114),deptht = 0).values
    
    tdra = np.ravel(DIC * 1e-3) #DIC, convert mol/L to mol/m3 (1000 L/m3)
    ttara = np.ravel(ALK * 1e-3) #Alkalinity, convert mol/L to mol/m3 (1000 L/m3)
    tsra = np.ravel(vosaline) #salt
    tsra_psu = np.ravel(vosaline) *35/35.16504 #g/kg to psu (salt)
    ttera = np.ravel(votemper)

    tdepth = np.zeros_like(ttera)
    tdepth[:] = 0
    ttera_is = gsw.t_from_CT(tsra,ttera,tdepth)
    tzero = np.zeros_like(tdepth)
    tpressure = np.zeros_like(tdepth)



    response_tup = mocsy.mvars(temp=ttera_is, sal=tsra_psu, alk=ttara, dic=tdra, 
                       sil=tzero, phos=tzero, patm=tpressure, depth=tdepth, lat=tzero, 
                        optcon='mol/m3', optt='Tinsitu', optp='m',
                        optb = 'l10', optk1k2='m10', optkf = 'dg', optgas = 'Pinsitu')
    pH,pco2,fco2,co2,hco3,co3,OmegaA,OmegaC,BetaD,DENis,p,Tis = response_tup

    BetaD = BetaD.reshape(12,114, 362)
    BetaD[BetaD>100] = np.nan
    Revelle = bgc.DIC.isel(y=slice(0,114),deptht = 0).copy()
    Revelle[:] = BetaD
    Revelle_mean = Revelle.isel(y=slice(0,114)).weighted(tmask_surf.isel(y=slice(0,114))).mean(dim = ['x','y'])

    return Revelle_mean 

def get_grads(bgc, verbose = False):
    
    grad_dic_0_100_ar = np.zeros([12])
    grad_dic_0_200_ar = np.zeros([12])
    grad_dic_100_200_ar = np.zeros([12])
    grad_avg2_ar = np.zeros([12])
    DIC_1000_minus_0_ar = np.zeros([12])

    for m in range(12):
            
        tmask = xr.open_dataset(f'/gpfs/data/greenocean/software/resources/MEDUSA/mesh_mask.nc')
        tmask_surf = tmask.tmask[0,0,:,:] * tmask.e1t[0,:,:] * tmask.e2t[0,:,:]

        DIC = bgc.DIC.isel(y=slice(0,114)).weighted(tmask_surf.isel(y=slice(0,114))).mean(dim = ['x','y'])
        dic_0 = DIC[m,0]
        
        if np.isnan(dic_0):
            pass

        else:

            depth_0 = bgc.deptht[0]
            depths_first100 = bgc.deptht[0:25]
            dics_first100 = DIC[m,0:25]
            f = interp1d(depths_first100, dics_first100, kind='linear')
            dic_100 = f(100)

            depths_second100 = bgc.deptht[24:32]
            dics_second100 = DIC[m,24:32]
            f = interp1d(depths_second100, dics_second100, kind='linear')
            dic_200 = f(200)

            grad_dic_0_100 = ((dic_100-dic_0)/(100-depth_0)).values
            grad_dic_0_200 = ((dic_200-dic_0)/(200-depth_0)).values
            
            grad_dic_100_200 = ((dic_200-dic_100)/(200-100))
            grad_avg2 = (grad_dic_100_200+grad_dic_0_100)/2
            if verbose:
                print(grad_dic_0_100)
                print(grad_dic_0_200)
                print(grad_dic_100_200)
                print(grad_avg2)

            #bgc.deptht[44:50]
            depths_1000 = bgc.deptht[44:50]
            dics_1000 = DIC[m,44:50]
            f = interp1d(depths_1000, dics_1000, kind='linear')
            dic_1000 = f(1000)    
            DIC_1000_minus_0 = (dic_1000-dic_0).values

            if verbose:
                plt.plot(DIC[m,0:32],-bgc.deptht[0:32])
                plt.plot(dic_100, -100, 'ro')
                plt.plot(dic_200, -200, 'r+')

                print(f'grad_dic_0_100 {grad_dic_0_100}')
                print(f'grad_dic_0_200 {grad_dic_0_200}')
                print(f'grad_dic_100_200 {grad_dic_100_200}')
                print(f'grad_avg2 {grad_avg2}')
                print(f'DIC_1000_minus_0 {DIC_1000_minus_0}')

            grad_dic_0_100_ar[m] = grad_dic_0_100
            grad_dic_0_200_ar[m] = grad_dic_0_200
            grad_dic_100_200_ar[m] = grad_dic_100_200
            grad_avg2_ar[m] = grad_avg2
            DIC_1000_minus_0_ar[m] = DIC_1000_minus_0
            
    return grad_dic_0_100_ar, grad_dic_0_200_ar, grad_dic_100_200_ar, grad_avg2_ar, DIC_1000_minus_0_ar

def make_yearlist(yr, scen):
    
    tdir = '/gpfs/data/greenocean/software/resources/MEDUSA/PROC2/'

    if yr < 2015:
        runid = 'bc370'
    else:
        if scen == '1A':
            runid = 'be682'
        if scen == '1B':
            runid = 'ce417'

    bgc = xr.open_dataset(f'{tdir}medusa_{runid}_1y_{yr}_ptrc-T-CHLTADIC.nc')
    ts = xr.open_dataset(f'{tdir}nemo_{runid}_1y_{yr}_grid-T-TS.nc')
    diadaux = xr.open_dataset(f'{tdir}medusa_{runid}_1y_{yr}_diad-T-aux.nc')
    
    return bgc, ts, diadaux

def save_cchem_param(scen,yr):
    
    bgc,ts, diadaux = make_yearlist(yr, scen)
    
    savenam = f'./data/cchem-param-medusa_scen-{scen}-{yr}.nc'
    times = pd.date_range(f"{yr}/01/01",f"{yr}/12/11",freq='MS')#,closed='left')

    data_vars = {
                ### bgc gradients
        'grad_dDICdz': (['time_counter', 'quantity'], np.zeros([12,4]),
                {'units': ''}),
        'grad_DIC_surfdeep': (['time_counter', 'quantity'], np.zeros([12,4]),
                {'units': ''}),

                ### carbonate system variables
        'mn_kg': (['time_counter', 'quantity'], np.zeros([12,4]),
                {'units': ''}),
        'mn_SST': (['time_counter', 'quantity'], np.zeros([12,4]),
                {'units': ''}),
        'mn_Revfact': (['time_counter', 'quantity'], np.zeros([12,4]),
                {'units': ''}),
        'mn_DIC': (['time_counter', 'quantity'], np.zeros([12,4]),
                {'units': ''}),
        'mn_TA': (['time_counter', 'quantity'], np.zeros([12,4]),
                {'units': ''}),
        'mn_pCO2': (['time_counter', 'quantity'], np.zeros([12,4]),
                {'units': ''}),
        }

    coords = {'time_counter': (['time_counter'], times),
            'quantity': (['mean', 'stdev', 'max', 'min']),

    }
    # define global attributes
    attrs = {'made in':'scratch/BOE-SOcarbon/extract_BOE_parameters_from_MEDUSA.py',
    'desc': ''
    }
    ds = xr.Dataset(data_vars=data_vars,
    coords=coords,
    attrs=attrs)

    Rev_mean = get_revelle(ts,bgc)
    grad_dic_0_100_ar, grad_dic_0_200_ar, grad_dic_100_200_ar, grad_avg2_ar, DIC_1000_minus_0_ar = \
    get_grads(bgc)
    T_mean, DIC_mean, ALK_mean, OCNPCO2_mean, OCNDPCO2_mean, CO2FLUX_mean = get_weighted(ts,bgc, diadaux)

    ds['grad_dDICdz'][:,0] = grad_avg2_ar 
    ds['grad_DIC_surfdeep'][:,0] = DIC_1000_minus_0_ar; 
    ds['mn_kg'][:,0] = 0.065; 
    ds['mn_SST'][:,0] = T_mean.values; 
    ds['mn_Revfact'][:,0] = Rev_mean.values; 
    ds['mn_DIC'][:,0] = DIC_mean.values; 
    ds['mn_TA'][:,0] = ALK_mean.values; 
    ds['mn_pCO2'][:,0] = OCNPCO2_mean.values; 

    print(savenam)
    ds.to_netcdf(savenam)

    return ds
       

for yr in range(1950,2100):
    ds = save_cchem_param('1A',yr)
    ds = save_cchem_param('1B',yr)
    