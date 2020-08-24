# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:54:21 2020

@author: bjkmo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime

from scipy.interpolate import interp1d
from scipy.stats import ttest_1samp

import PIL


#%% Functions

def RMSE(obs, truth):
    return np.sqrt( np.mean((obs-truth)**2) )

def MARE(obs, truth):
    top = (obs+1e-7) - (truth+1e-7)
    bot = (truth+1e-7)
    return np.mean(np.abs(top/bot))

def BIAS(obs, truth):
    return np.mean(truth-obs)

#%%


    
bath_dir = r'C:\Users\bjkmo\Desktop\12326 Field Course in Applied Geophysics\GPR'
bath_file_gps = 'GPR_data_final.xlsx'
bath_file = 'Bathymetry_Tools_Comparison.xlsx'

readfile = os.path.join(bath_dir, bath_file)
readfile_gps = os.path.join(bath_dir, bath_file_gps)



gpr1 = pd.read_excel(readfile, sheet_name='GPR_38')[['Corrected distance across', 'Depth' ]]  # 'Correlation correction']]
gpr1.columns = ['join_cngmeters','Depth']
gpr1.join_cngmeters = gpr1.join_cngmeters[::-1].values

gpr1_gps = pd.read_excel(readfile_gps, sheet_name='xs38_chain')[['join_cngmeters','Depth']]
gpr1_gps = gpr1_gps.groupby('join_cngmeters').aggregate('mean')
gpr1_gps = gpr1_gps.reset_index()
#gpr1['Dataset'] = 'GPR 38'


gpr2 = pd.read_excel(readfile, sheet_name='GPR_39')[['Corrected distance across',  'Depth' ]]  #'Correlation correction']]
gpr2.columns = ['join_cngmeters','Depth']

gpr2_gps = pd.read_excel(readfile_gps, sheet_name='xs39_chain')[['join_cngmeters','Depth']]
gpr2_gps = gpr2_gps.groupby('join_cngmeters').aggregate('mean')
gpr2_gps = gpr2_gps.reset_index()
#gpr1['Dataset'] = 'GPR 38'

rtk =  pd.read_excel(readfile_gps, sheet_name='rtk_chain')[['join_cngmeters','Depth']]
#rtk['Dataset'] = 'RTK GPS'


stick =  pd.read_excel(readfile, sheet_name='Stick')[['Corrected distance','height [m]']]
stick.columns = ['join_cngmeters','Depth']
stick.join_cngmeters += 0.3

sonar =  pd.read_excel(readfile,
                      sheet_name='Sonar',
#                      header=None,
                      names=['join_cngmeters','depth_neg','Depth']
                      )[['join_cngmeters','Depth']]
#sonar['Dataset'] = 'Sonar'
sonar = sonar.dropna()



error_filename = 'bathymetry_error_uncorrected.xlsx'
#
## GPR profiles are corrected by using static measurements for calibrating depth
for dff in [gpr1, gpr2, gpr1_gps, gpr2_gps]:
    dff.Depth = dff.Depth*1.0727 + 0.0285
    error_filename = 'bathymetry_error_corrected.xlsx'








#%%


plt.figure()
plt.gca().invert_yaxis()


#plt.plot(gpr1.join_cngmeters, gpr1.Depth*100, '-^r', label='GPR 38', markerfacecolor='None')
#plt.plot(gpr2.join_cngmeters, gpr2.Depth*100, '-sr', label='GPR 39', markerfacecolor='None' )

plt.plot(gpr1_gps.join_cngmeters, gpr1_gps.Depth*100, '-^b', label='GPR 38 - GPS') #, markerfacecolor='None' )
plt.plot(gpr2_gps.join_cngmeters, gpr2_gps.Depth*100, '-sb', label='GPR 39 - GPS') #, markerfacecolor='None' )


plt.plot(sonar.join_cngmeters, sonar.Depth*100, '-sk', ms=5, label='Sonar')

plt.plot(stick.join_cngmeters, stick.Depth*100, '-sk', label='Stick', ms=8, markerfacecolor='white')

plt.plot(rtk.join_cngmeters, rtk.Depth*100, '-ok', label='RTK GPS', ms=8, markerfacecolor='yellow')


plt.legend()
plt.xlabel('Distance from left bank [m]')
plt.ylabel('Depth [cm]')

plt.tight_layout()
plt.show()





#%% Errors



fdict = {}
rmse = {}
mare = {}
bias = {}
bias_pval = {}

labels = ['gpr1', 'gpr2', 'gpr1_gps', 'gpr2_gps', 'sonar', 'rtk', 'stick']
dfs = [gpr1, gpr2, gpr1_gps, gpr2_gps, sonar, rtk, stick]

for i in range(len(dfs)):
    dff = dfs[i]
    lab = labels[i]
    fdict[lab] = interp1d(dff.join_cngmeters, dff.Depth*100)


xnew = np.arange(1.0,9.0,0.1)

for i in range(len(dfs)-1):
    dff = dfs[i]
    lab = labels[i]

    obs = fdict[lab](xnew)
    truth = fdict['stick'](xnew)

    
    rmse[lab] = RMSE(obs, truth)
    mare[lab] = MARE(obs, truth)
    bias[lab] = BIAS(obs, truth)
    bias_pval[lab] = ttest_1samp(obs-truth, 0).pvalue
    




bathymetry_error = pd.DataFrame.from_dict({'RMSE': rmse, 'MARE': mare, 'BIAS':bias, 'BIAS pval':bias_pval}).round({'RMSE':1, 'MARE':2, 'BIAS':1})
bathymetry_error.to_excel(error_filename)






