# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 13:45:09 2020

@author: bjkmo
"""




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime

from scipy.interpolate import UnivariateSpline as spline

#%% Read data
wse_dir = r'C:\Users\bjkmo\Desktop\12326 Field Course in Applied Geophysics\Surface elevation'
wse_file = 'WSE_RESULTS.xlsx'

readfile = os.path.join(wse_dir, wse_file)


radar = pd.read_excel(readfile,
                      sheet_name='RADAR'
                      )

egtved = pd.read_excel(readfile,
                      sheet_name='Radar_Egtved'
                      )

rtk = pd.read_excel(readfile,
                    sheet_name='RTK'
                    )

dem = pd.read_excel(readfile,
                    sheet_name='DEM'
                    )

# 32.30 Vejle å, ns Egtved å
# 32.03 Vejle å, Refsgårdslund dambrug
vandstand = np.array([0.720, 0.810] )
nulpunkt = np.array([14.38, 14.07] )
in_situ_coordinates = np.array([12846, 13290] )
in_situ = np.array([in_situ_coordinates, vandstand+nulpunkt] )

#%% Mask data

# mask radar
powermask = radar['return pow'] >= 70
distancemask = radar['distance'] <= 10
chainage_mask = radar['join_cngme'] >= 12805

radar_mask = powermask & distancemask & chainage_mask



#mask1 = radar_mask & (radar['Day'] == 1)
#mask2 = radar_mask & (radar['Day'] == 2)
#mask3 = radar_mask & (radar['Day'] == 3)
#mask4 = radar_mask & (radar['Day'] == 4)

# mask DEM
dem_mask = dem.DEM.between(dem.DEM.quantile(.05), dem.DEM.quantile(.95))


# RTK GPS mask
rtk_mask = rtk.join_cngmeters >= 12805


#%%


#plt.figure()
#plt.plot(radar.loc[mask1,'join_cngme'],radar.loc[mask1,'W.S.E. abo'], '.' )
#plt.plot(radar.loc[mask2,'join_cngme'],radar.loc[mask2,'W.S.E. abo'], '.' )
#plt.plot(radar.loc[mask3,'join_cngme'],radar.loc[mask3,'W.S.E. abo'], '.' )
#plt.plot(radar.loc[mask4,'join_cngme'],radar.loc[mask4,'W.S.E. abo'], '.' )
#plt.show()




plt.figure()

scatter = plt.scatter(radar.loc[radar_mask,'join_cngme'],
                      radar.loc[radar_mask,'W.S.E. abo'], 
                      c=radar.loc[radar_mask,'Day'], 
                      alpha=0.5,
                      s=8,
                      cmap='tab20b',
                      
                      )

plt.plot(dem.join_cngmeters[dem_mask], dem.DEM[dem_mask],'k', alpha=0.5, linewidth=0.5, label='Photogrammetry DEM')
plt.plot(rtk.join_cngmeters[rtk_mask], rtk.Elevation[rtk_mask], 'ok', label='RTK GPS', ms=8, markerfacecolor='yellow')
plt.plot(in_situ[0], in_situ[1], 'ok', ms=8, markerfacecolor='b', label = 'In situ stations')

legend1 = plt.legend(*scatter.legend_elements(),
                     loc="lower left", title="Radar \n         Day", 
                     fontsize=12)
plt.gca().add_artist(legend1)

plt.legend(loc="center left", title="", )

#plt.title('Water surface elevation along Vejle Å')
plt.xlabel('River chainage [m]')
plt.ylabel('Water surface elevation [m.a.m.s.l.]')
plt.tight_layout()

plt.show()



#%% Interpolation 

# new 'looser' mask
powermask = radar['return pow'] >= 55
distancemask = radar['distance'] <= 15
chainage_mask = radar['join_cngme'] >= 12935

radar_mask = powermask & distancemask  & chainage_mask



#splines = {}
#for i in range(4):
   
df1 = (radar.loc[radar_mask & (radar.Day==3), ['join_cngme', 'W.S.E. abo']].
             groupby('join_cngme').
             agg('mean').
             reset_index().
             sort_values('join_cngme'))
outlier_mask = df1['W.S.E. abo'].between(df1['W.S.E. abo'].quantile(.05), df1['W.S.E. abo'].quantile(.95))

x1 = df1['join_cngme'].values[outlier_mask]
y1 = df1['W.S.E. abo'].values[outlier_mask]
    
#    splines['f'+str(i)] =  


#%% Plot

f1 = spline(x1, y1, k=3, s=2)
residual_mask = abs(f1(x1)-y1) < 0.05 

f2 = spline(x1[residual_mask],y1[residual_mask] , k=3, s=0.5)
residual_mask2 = abs(f2(x1)-y1) < 0.03 

f3 = spline(x1[residual_mask2],y1[residual_mask2] , k=3, s=0.5)


plt.figure()
#plt.plot(x1,y1, '.')
#plt.plot(x1[residual_mask],y1[residual_mask], '.k')
plt.plot(x1[residual_mask2],y1[residual_mask2], '.', label='Radar - day 3')
#plt.plot(x1, f1(x1))
#plt.plot(x1, f2(x1), 'r')
plt.plot(x1, f3(x1), 'k', label='Radar - spline')
plt.plot(rtk.join_cngmeters[rtk_mask], rtk.Elevation[rtk_mask], 'ok', label='RTK GPS', ms=8, markerfacecolor='yellow')

#plt.text(2,4,'This text starts at point (2,4)')


plt.legend()
plt.show()




#%% Errors

x2, y2 =  x1[residual_mask2],y1[residual_mask2]
rtk1 = rtk.loc[rtk_mask, ['join_cngmeters','Elevation']]
rtk1.columns=['x','y']
rtk1 = rtk1.sort_values('x').reset_index(drop=True)


mean1=np.zeros(len(rtk1))
std1=np.zeros(len(rtk1))


for i in range(len(rtk1)):
    point = rtk1.iloc[i]
    m = np.where( np.abs(x2-point.x ) < 2)
    if len(m[0])==0:
        continue

    mean1[i] = np.mean(y2[m])
    std1[i] = np.std(y2[m])




error_df = pd.DataFrame({'RTK_No.': np.arange(len(rtk1)),'RTKx': rtk1.x.values, 'RTKy': rtk1.y.values,'Mean': mean1, 'sdev':std1} ).drop(labels=0)
error_df.to_excel('WSE_errors.xlsx')





#%%

plt.figure()
plt.plot(rtk.join_cngmeters[rtk_mask], rtk.Elevation[rtk_mask], 'ok', label='RTK GPS', ms=8, markerfacecolor='yellow')

#plt.plot(x1,y1, '.')
#plt.plot(x1[residual_mask],y1[residual_mask], '.k')
plt.plot(x1[residual_mask2],y1[residual_mask2], '.', label='Radar - day 3')
#plt.plot(x1, f1(x1))
#plt.plot(x1, f2(x1), 'r')


plt.plot(error_df.RTKx, error_df.Mean, '.k', label='Radar - mean w/ 95% CI')

#plt.plot(x1, f3(x1), '--k', label='Radar - spline', linewidth=1)

plt.errorbar(error_df.RTKx, error_df.Mean, 
             yerr = error_df.sdev.values*2, 
             fmt='none', 
             color='k',
             barsabove=True, 
             zorder=100,
             capsize=4,
             capthick=1.5,
             lw=2
             )

#for i in range(len(rtk1)):
#    point = rtk1.iloc[i].values
#    plt.text(point[0]+7,point[1],i+1)

plt.legend(loc='lower left')

#plt.title('Water surface elevation along Vejle Å')
plt.xlabel('River chainage [m]')
plt.ylabel('Water surface elevation \n[m.a.m.s.l.]')
plt.tight_layout()


plt.show()



#%%
df111 = radar.loc[radar_mask & (radar.Day==3), ['UTMx(meter',   'UTMy(meter','join_cngme', 'W.S.E. abo']]
df111 = df111[df111['W.S.E. abo'].between(df111['W.S.E. abo'].quantile(.05), df111['W.S.E. abo'].quantile(.95))]
df111.to_csv('radar_filtered.csv')


