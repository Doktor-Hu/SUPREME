#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File        :   Data_Preprocessing.py
@Time        :   2022/11/30 13:12:01
@Author      :   Zhongyang Hu
@Version     :   1.0.0
@Contact     :   z.hu@uu.nl
@Publication :   
@Desc        :   GEE output to NPY
'''
import os
import json
import geopandas as gpd
import glob
import copy
import numpy as np
import xarray as xarray


###------  User Config.

alb_dir = 'G:/My Drive/Albedo_ANT/Alb_Char'
shp_fn = 'C:/Users/zh_hu/Documents/ARSM/BATCH/GRID/5500_GRID.shp'
Param = 'Alb'
clip_prefix = 'ALB_Char_NA_replaced_'
res = 5500

###---------------------------------------------------------------------------------------------------------------------------------------------------------
###---------------------------------------------------------------------------------------------------------------------------------------------------------
###---------------------------------------------------------------------------------------------------------------------------------------------------------


parent_dir = alb_dir+'/Training/AP'
alb_ori = glob.glob(alb_dir + '\*.tif')
## Albeodo to AP (Training)
# ---- S1: Cropping to AP

for alb_fn  in alb_ori:
    outname = parent_dir + '/' + os.path.basename(alb_fn)[:-4] + '_AP.tif'
    code = 'gdalwarp -s_srs EPSG:3031 -t_srs EPSG:3031 -r average -te -3580843.6588 -151831.8217 -852843.6588 2383668.1783 -te_srs EPSG:3031 -of GTiff "{}" "{}'.format(alb_fn,outname)
    print(code)
    #os.system(code)

# ---- S2: Same as RACMO2 5.5


class data_prep:
     
    def __init__(self, grid_fn, tif_fn):
        self.grid = gpd.read_file(grid_fn)
        self.tif = xr.open_rasterio(tif_fn)
        
    def get_coor(self, index):
        gdf = self.grid
        gdf_sub = gdf[gdf['DN']==index]
        feature = [json.loads(gdf_sub.to_json())['features'][0]['geometry']]
        coors = feature [0]['coordinates'][0]
        x1 = (max([x[0] for x in coors])+min([x[0] for x in coors]))/2
        x2 = (max([x[1] for x in coors])+min([x[1] for x in coors]))/2
        
        return x1,x2    
    
    def get_xy(self, xs, ys):
        ds = self.tif 
        y=np.where(ds['y']==ds.sel(x=xs, y=ys, method="nearest")['y'])[0]
        x=np.where(ds['x']==ds.sel(x=xs, y=ys, method="nearest")['x'])[0]
        
        return int(x), int(y)
    

def latlon_to_xy(tif_in, locs):
    x0, y0 = tif_in.get_coor(locs['NW'])
    x1, y1 = tif_in.get_xy(x0,y0)
    
    x2, y2 = tif_in.get_coor(locs['SE'])
    x3, y3 = tif_in.get_xy(x2,y2)
    
    return tif_in.tif[:,y1:(y3+1),x1:(x3+1)]

def latlon_to_xy_loc(tif_in, locs):
    x0, y0 = tif_in.get_coor(locs['NW'])
    x1, y1 = tif_in.get_xy(x0,y0)
    
    x2, y2 = tif_in.get_coor(locs['SE'])
    x3, y3 = tif_in.get_xy(x2,y2)
    
    return y1, (y3+1), x1, (x3+1)


if not os.path.isdir(os.path.join(parent_dir,'Output')):
    os.mkdir(parent_dir+'/'+'Output')

if not os.path.isdir(os.path.join(parent_dir,'Output',Param)):
    os.mkdir(parent_dir+'/'+'Output'+'/'+Param)

os.chdir(os.path.join(parent_dir,'Output',Param))
print('Change to Param Directory: ' + os.path.join(parent_dir,'Output',Param))
print()

# Make subdir
for i in range(1,14):
    if os.path.isdir(clip_prefix+'AOI'+str(i))==False:
        os.mkdir(clip_prefix+'AOI'+str(i))
        
files = glob.glob(parent_dir+'/*.tif')    
#f=files[0]

for f in files:

    GRID_55=data_prep(shp_fn,f)
    AP55={'NW':130726,'NE': 130779, 'SW': 157014, 'SE':157067}
    x_s, x_e, y_s, y_e = latlon_to_xy_loc(GRID_55,AP55)

    AOI_1 = (x_s-54*3), (x_e-54*3), (y_s-54*2), (y_e-54*2)

    AOI_2 = (x_s-54*2), (x_e-54*2), (y_s-54*2), (y_e-54*2)
    AOI_3 = (x_s-54*2), (x_e-54*2), (y_s-54), (y_e-54)

    AOI_4 = (x_s-54), (x_e-54), (y_s-54*2), (y_e-54*2)

    AOI_5 = (x_s-54), (x_e-54), (y_s-54), (y_e-54)
    AOI_6 = (x_s-54), (x_e-54), y_s, y_e
    AOI_7 = (x_s-54), (x_e-54), (y_s+54), (y_e+54)

    AOI_8 = x_s, x_e, (y_s-54), (y_e-54)
    AOI_9 = x_s, x_e, y_s, y_e
    AOI_10 = x_s, x_e, (y_s+54), (y_e+54)

    AOI_11 = (x_s+54), (x_e+54), (y_s-54), (y_e-54)
    AOI_12 = (x_s+54), (x_e+54), y_s, y_e
    AOI_13 = (x_s+54), (x_e+54), (y_s+54), (y_e+54)
    AOIs = [AOI_1, AOI_2, AOI_3, AOI_4, AOI_5, AOI_6, AOI_7, AOI_8, AOI_9, AOI_10,  AOI_11, AOI_12, AOI_13]



    k=1

    for aoi in AOIs:
        if os.path.isfile(os.path.join(parent_dir+'/'+'Output'+'/'+Param+'/',clip_prefix+'AOI'+str(k),os.path.basename(f)[:-4]+'_AOI_'+str(k)+'.npy'))==False:
            AOI_IMG=GRID_55.tif.values

            mod = np.zeros(AOI_IMG.shape)

            for channel in range(5):
                temp = copy.deepcopy(AOI_IMG[:,:,channel])
                temp[np.where(np.isnan(temp))]=np.nanmedian(AOI_IMG[:,:,channel])
                mod[:,:,channel] = temp

                del temp
                 

            AOI_IMG_mod=mod[:,aoi[0]:aoi[1], aoi[2]: aoi[3]]
            clipped = AOI_IMG_mod[:,:,:]
            clip_out_fn=os.path.join(parent_dir+'/'+'Output'+'/'+Param+'/',clip_prefix+'AOI'+str(k),os.path.basename(f)[:-4]+'_AOI_'+str(k)+'.npy')
            print('[Prosessing Start]: Save Numpy Array '+clip_out_fn)
            if os.path.isfile(clip_out_fn)==False:
                np.save(clip_out_fn, clipped)
            k+=1
        else:
            print(os.path.join(parent_dir+'/'+'Output'+'/'+Param+'/',clip_prefix+'AOI'+str(k),os.path.basename(f)[:-4]+'_AOI_'+str(k)+'.npy') + ' already exists!')


## ---- FULL

#### ----- Alb ANT

if not os.path.isdir(os.path.join(alb_dir,'ANT_Extent')):
    os.mkdir(os.path.join(alb_dir,'ANT_Extent'))

for ori_fn in alb_ori:
    extend_fn = alb_dir +'/ANT_Extent/' + os.path.basename(ori_fn)[:-4]+'_Near_Entend.tif'
    code = 'gdalwarp -r near -te -3941905.7032 -4111955.9934 3915094.2968 4096044.0066 -of GTiff "{}" "{}"'.format(ori_fn,extend_fn)
    print(code)
    os.system(code)
    alb_data = xr.open_rasterio(extend_fn).data
    dem_out_fn = alb_dir +'/ANT_Extent/' + os.path.basename(ori_fn)[:-4]+'_Near_Entend.npy'
    np.save(dem_out_fn,alb_data)