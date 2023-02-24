import imp
import re
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import shapely as sp
import matplotlib.pyplot as plt
import geopandas as gpd
from glob import glob
from csv import reader

# %%
def mech(rake):
    indR = np.where((rake > 45.) & (rake < 135.))[0]
    indN = np.where((rake > 225.) & (rake < 315.))[0]
    indLL = np.where((rake >= 315.) | (rake <= 45.))[0]
    indRL = np.where((rake >= 135.) & (rake <= 225.))[0]
    res = np.empty(len(rake),dtype='<U3')
    if(len(indR) != 0): res[indR] = 'R'

    if(len(indN) != 0): res[indN] = 'N'

    if(len(indLL) != 0): res[indLL] = 'LL'

    if(len(indRL) != 0): res[indRL] = 'RL'

    return res

# %%
def col_mech(rake):
    indR = np.where((rake > 45.) & (rake < 135.))[0]
    indN = np.where((rake > 225.) & (rake < 315.))[0]
    indLL = np.where((rake >= 315.) | (rake <= 45.))[0]
    indRL = np.where((rake >= 135.) & (rake <= 225.))[0]
    col_mech = np.empty(len(rake),dtype='<U7')
    col_mech[indR] = "red"
    col_mech[indN] = "gold"
    col_mech[indLL] = "yellow"
    col_mech[indRL] = "orange"

    return col_mech

# %%
def dip_dir(geometry):

    dipdir = np.empty(geometry.shape[0])
    for i in range(geometry.shape[0]):
        coords = np.array(geometry.iloc[i].coords)
        dipdir[i] = -(coords[-1,0]-coords[0,0])/np.abs(coords[-1,0]-coords[0,0])

    return dipdir

# %%
# optimization version of function jump() with python package 'shapely'
def jump(source, init, region, Delta, round):

    'create buffer around Delta'
    'Delta = 5km in the theoretical case but depends in practice on fault database resolution'

    # Input  - source (GeoDataFrame): complete dataset of the strike-slip fault segments   
    #        - init (GeoDataFrame): the initial cascading sequence at present cascading round
    #        - region (ndarray): range of plots
    #        - Delta (float): the maximum distance that allows the jumping of rupture between
    #                         two independent fault segment
    #        - round (int): index of cascading round
    #  
    # Output - inddist (Series): indices/id of fault segments that satisfy the criteria


    # creat buffer around Delta, default buffer resolution = 16
    buffer = init['geometry'].to_crs(4479).iloc[0].buffer(Delta*1000)# shapely.Polygon

    # get faults in buffer
    inddist = source['id'].loc[source.to_crs(4479).intersects(buffer)]# indices of source fault segment that intersect with buffer
    inddist.reset_index(drop=True, inplace=True)
    if(round == 0):
        inddist = inddist.loc[inddist!=init['id'].iloc[0]]# remove init.id from inddist in the 1st round (round == 0)
        inddist.reset_index(drop=True, inplace=True)
    elif(round > 0):
        parts = init.iloc[0,3:-1]# cascade sequence(indexs of faults) in init
        # remove indexes in parts from inddist
        for values in parts:
            inddist = inddist.loc[inddist!=values]
    
    # remove parts (inital cascade sequence) with different mechanism
    njump = len(inddist)
    if(njump > 0):
        mechinit = mech(init['rake'])# mechanism of fault segement in the initial cascasde sequence
        mechjump = mech(source['rake'].iloc[inddist])# mechanism of fault segment that intersect with buffer
        inddist = inddist.loc[mechjump == mechinit]
        inddist.reset_index(drop=True, inplace=True)
        
        # remove parts (inital cascade sequence) with different dip direction
        njump = len(inddist)
        if(njump > 0):
            dipdir_init = dip_dir(init['geometry'])
            # WARNING: in the case of cascades, sign does not match dip direction,
            # but depends on the direction of propagation, eg. A+B or B+A
            # While originally a glitch, it permits to remove redundant cascades!
            # NB: the dip of cascades cannot be derived from sign
            dipdir_jump = dip_dir(source['geometry'].iloc[inddist])
            inddist = inddist.loc[dipdir_jump == dipdir_init]
            inddist.reset_index(drop=True, inplace=True)

    # plot results
    buffer = gpd.GeoSeries(buffer)# tranform for easy plot

    #fig, ax = plt.subplots(dpi=800, subplot_kw={'projection': ccrs.PlateCarree()})
    #source.plot(ax=ax, color='grey', linewidth=0.5, zorder=1)
    #if(inddist.shape[0]!=0): source.loc[inddist].plot(ax=ax, color='red', linewidth=0.5, zorder=2) 
    #buffer.plot(ax=ax, color=col_mech(init['rake']), alpha=0.5, zorder=3)
#
    #ax.set_aspect('equal')
    #ax.set_extent([region[0]-1, region[1], region[2]-1, region[3]+1], crs=ccrs.PlateCarree())
    #ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
    #plt.ioff()
    ##fig.savefig('./1_dist_seg'+str(init['id'].iloc[0])+'.pdf')
    #plt.close(fig)

    return inddist

# %%
def bendbranch(source, init, region, inddist, muD, delta):

    # Input  - source (GeoDataFrame): complete dataset of the strike-slip fault segments   
    #        - init (GeoDataFrame): the initial cascading sequence at present cascading round
    #        - region (ndarray): range of plots
    #        - inddist (Series): indices/id of fault segments that satisfy the jump() criteria
    #        - muD (float): dynamic friction coefficient
    #        - delta (float [degree]): range of preferred orientation
    #  
    # Output - indangle (Series): indices/id of fault segments that satisfy the bendbranch() criteria

    alpha = init['strike'].iloc[0]
    Psi = (init['rake'].iloc[0]/2.+45.)%90# angle between segment & Smax direction
    gamma = np.nan# cases when rake not SS or LL-RL combined
    if((init['rake'].iloc[0] >= 135) & (init['rake'].iloc[0] <= 225)): gamma = 1 # right-lateral
    if((init['rake'].iloc[0] >= 315) | (init['rake'].iloc[0] <= 45)): gamma = -1 # left-lateral
    # optimal angle for rupture, psi = np.nan for cases when rake not SS or LL-RL combined
    psi = gamma*(45-Psi-180*np.arctan(muD)/(2*np.pi))

    if(psi != np.nan):
        beta = source['strike'].loc[inddist]
        phi = -1*beta + alpha
        indangle = source['id'].loc[inddist].loc[(phi >= (psi-delta))&(phi <= (psi+delta))]
        indangle.reset_index(drop=True, inplace=True)
    
    # plot results
    #if(indangle.shape[0] > 0):
    #    
    #    fig = plt.figure(dpi=800)
    #    ax = plt.subplot(1,1,1,projection=ccrs.PlateCarree())
    #    source.loc[source['id']!=init['id'].iloc[0]].plot(ax=ax, color='grey', linewidth=0.5, zorder=1)
    #    source.loc[indangle].plot(ax=ax, color='red', linewidth=0.5, zorder=2)
    #    init.plot(ax=ax, color=col_mech(init['rake']), linewidth=0.5, alpha=0.7, zorder=3)
#
    #    ax.set_aspect('equal')
    #    ax.set_extent([region[0]-1, region[1], region[2]-1, region[3]+1], crs=ccrs.PlateCarree())
    #    ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
#
    #    plt.ioff()
    #    #plt.savefig('./2_angle_seg'+str(init['id'].iloc[0])+'.pdf')
    #    plt.close(fig)

    return indangle

# %%
def LS2P(ls):

    "Transform Shapely.LineString to GeoSeries that store vertices of 'LineString'"
    "in the form of Shapely.Point "

    # Input  - ls (Shapely.LineString):  
    #  
    # Output - p (GeoSeries): GeoSeries that store vertices of 'ls' in the form of Shapely.Point 

    p = gpd.GeoSeries(crs='EPSG:4326')

    for point in ls.coords:
        p = p.append(gpd.GeoSeries(sp.geometry.Point(point), crs='EPSG:4326'), ignore_index=True)
    
    return p

# %%
def propa(source, init, region, indangle, id_new, round):

    # Input  - source (GeoDataFrame): complete dataset of the strike-slip fault segments   
    #        - init (GeoDataFrame): the initial cascading sequence at present cascading round
    #        - region (ndarray): range of plots
    #        - indangle (Series): indices/id of fault segments that satisfy the bendbranch() criteria
    #        - id_new (int): index of each new event during each cacading round
    #        - round (int): index of cascading round
    #        - 
    # Output - id_new (int): index of each new event during each cacading round

    
    list_A = LS2P(init['geometry'].iloc[0])# source A (initiator)

    for index, row in source.loc[indangle].iterrows():
        list_B = LS2P(row['geometry'])# source B (propagator)

        # anchor points
        indanchor, d = list_B.sindex.nearest(list_A, return_distance=True)# indices & corresponding distance
        indanchor = indanchor[:,np.argmin(d)]
        #anchor_A = list_A.iloc[indanchor[0]]
        #anchor_B = list_B.iloc[indanchor[1]]

        # join segments (A1-B1 / A1-B2 / A2-B1 / A2-B2)
        splitA1 = np.arange(0,indanchor[0]+1,1)# [0:indanchor[0]]
        splitA2 = np.arange(list_A.shape[0]-1,indanchor[0]-1,-1)# [(shape[0]-1):indanchor[0]]
        splitB1 = np.arange(indanchor[1],-1,-1)# [indanchor[1]:0]
        splitB2 = np.arange(indanchor[1],list_B.shape[0],1)# [indanchor[1]:(list_B.shape[0]-1)]

        angleA1B1 = np.nan
        angleA1B2 = np.nan
        angleA2B1 = np.nan
        angleA2B2 = np.nan

        # identify if the angle is obtuse or acute by the dot product of two vectors
        # A1-B1
        if((len(splitA1)>1) & (len(splitB1)>1)):
            angleA1B1 = (list_A.iloc[0].x - list_A.iloc[indanchor[0]].x) * (list_B.iloc[indanchor[1]].x - list_B.iloc[0].x) \
                      + (list_A.iloc[0].y - list_A.iloc[indanchor[0]].y) * (list_B.iloc[indanchor[1]].y - list_B.iloc[0].y)
        # A1-B2
        if((len(splitA1)>1) & (len(splitB2)>1)):
            angleA1B2 = (list_A.iloc[0].x - list_A.iloc[indanchor[0]].x) * (list_B.iloc[indanchor[1]].x - list_B.iloc[-1].x) \
                      + (list_A.iloc[0].y - list_A.iloc[indanchor[0]].y) * (list_B.iloc[indanchor[1]].y - list_B.iloc[-1].y)
        # A2-B1
        if((len(splitA2)>1) & (len(splitB1)>1)):
            angleA2B1 = (list_A.iloc[-1].x - list_A.iloc[indanchor[0]].x) * (list_B.iloc[indanchor[1]].x - list_B.iloc[0].x) \
                      + (list_A.iloc[-1].y - list_A.iloc[indanchor[0]].y) * (list_B.iloc[indanchor[1]].y - list_B.iloc[0].y)
        # A2-B2
        if((len(splitA2)>1) & (len(splitB2)>1)):
            angleA2B2 = (list_A.iloc[-1].x - list_A.iloc[indanchor[0]].x) * (list_B.iloc[indanchor[1]].x - list_B.iloc[-1].x) \
                      + (list_A.iloc[-1].y - list_A.iloc[indanchor[0]].y) * (list_B.iloc[indanchor[1]].y - list_B.iloc[-1].y)
        
        path = np.array([angleA1B1, angleA1B2, angleA2B1, angleA2B2])
        path_correct = np.where((path != np.nan) & (path > 0))[0]

        if(len(path_correct) > 0):

            # delete small bits of segment association
            if(len(path_correct) > 1):
                tmp_length =np.array([len(splitA1)+len(splitB1), len(splitA1)+len(splitB2), len(splitA2)+len(splitB1), len(splitA2)+len(splitB2)])
                indmax = np.argmax(tmp_length[path_correct])
                path_correct = path_correct[indmax]
            
            # path (coordinates) of rupture porpagation
            if(path_correct == 0):
                list_AB = list_A.iloc[splitA1].append(list_B.iloc[splitB1], ignore_index=True)
            elif(path_correct == 1):
                list_AB = list_A.iloc[splitA1].append(list_B.iloc[splitB2], ignore_index=True)
            elif(path_correct == 2):
                list_AB = list_A.iloc[splitA2].append(list_B.iloc[splitB1], ignore_index=True)
            elif(path_correct == 3):
                list_AB = list_A.iloc[splitA2].append(list_B.iloc[splitB2], ignore_index=True)

            # create cascade source
            cascade = gpd.GeoDataFrame({'id': ['{0:05}'.format(id_new)],
                                        'strike': [(init['strike'].iloc[0] + row['strike'])/2.],
                                        'rake': [(init['rake'].iloc[0] + row['rake'])/2], 
                                        'geometry': [sp.geometry.LineString(list_AB.to_list())]},
                                       crs='EPSG:4326')
            if(round == 0):
                cascade['part-0'] = init['id'].iloc[0]
            elif(round > 0):
                cascade = cascade.merge(init.iloc[:,3:-1], left_index=True, right_index=True)

            cascade['part-'+str(round+1)] = row['id']# cascade['part-(round+1)']

            # save files: append GeoDataFrame to .shp & .csv files
            if(id_new == round*10000):
                cascade.to_file('./file_cascade_'+str(round)+'.shp')
            elif(id_new > round*10000):
                cascade.to_file('./file_cascade_'+str(round)+'.shp', mode='a')

            cascade.iloc[:,np.delete(np.arange(0,cascade.shape[1]),3)].to_csv('./file_cascade_'+str(round)+'.csv', mode='a', sep='\t', index=False, header=False)


            # plot results
            #fig = plt.figure(dpi=800)
            #ax = plt.subplot(1,1,1,projection=ccrs.PlateCarree())
            #source.plot(ax=ax, color='grey', linewidth=0.5, zorder=1)
            #cascade.plot(ax=ax, color='red', linewidth=0.5, zorder=2)
#
            #ax.set_aspect('equal')
            #ax.set_extent([region[0]-1, region[1], region[2]-1, region[3]+1], crs=ccrs.PlateCarree())
            #ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
#
            #plt.ioff()
            ##plt.savefig('./3_propa_seg'+str(init['id'].iloc[0])+'_'+str('{0:05}'.format(id_new))+'.pdf')
            #plt.close(fig)

            id_new = id_new + 1
            
    return id_new
        
# %%
def L2M(L,W):
    A = L*W
    # Well & Coppersmith (1994): Table 2A for Strike-Slip
    m_WC = 5.16 + 1.12*np.log10(L)
    # Mai & Beroza (2000)
    m_MB = 0.67*((np.log10(L) + 5.15)/0.36 + 7) - 10.7# Hanks & Kanamori (1979)
    # Hanks & Bakun (2002): Strik-Slip
    m_HB = np.log10(A) + 3.98
    m_HB[np.where(A > 537)[0]] = 4/3*np.log10(A[A > 537]) + 3.07
    # Leonard (2010): Table 6 for Strike-Slip
    m_L = 1.67*np.log10(L) + 4.17
    # Wesnousky (2008): Strike-Slip
    m_W = 5.56 + 0.87*np.log10(L)

    return m_WC, m_MB, m_HB, m_L, m_W
    
def L2M_CN(L):

    # Wesnousky (2008): Strike-Slip
    m_W = 5.56 + 0.87*np.log10(L)

    return m_W    

def azimuth_1(point1, point2):
    '''azimuth between 2 shapely points (interval 0 - 180)'''

    if point1[0]<point2[0]:
        angle = np.arctan2(point2[0] - point1[0], point2[1] - point1[1], dtype=np.float64)
    else:
        angle = np.arctan2(point1[0] - point2[0], point1[1] - point2[1], dtype=np.float64)

    return np.degrees(angle) if angle >= 0 else np.degrees(angle) + 360

def azimuth_2(point1, point2):
    """
    https://www.omnicalculator.com/other/azimuth#how-to-calculate-the-azimuth-from-latitude-and-longitude
    θ = atan2 [(sin Δλ ⋅ cos φ₂), (cos φ₁ ⋅ sin φ₂ − sin φ₁ ⋅ cos φ₂ ⋅ cos Δλ)]
    """

    if point1[0]<point2[0]:
        angle = np.arctan2(np.sin(np.pi/180.*(point2[0] - point1[0]))*np.cos(np.pi/180.*point2[1]), \
                np.cos(np.pi/180.*point1[1])*np.sin(np.pi/180.*point2[1])- \
                np.sin(np.pi/180.*point1[1])*np.cos(np.pi/180.*point2[1])*np.cos(np.pi/180.*(point2[0] - point1[0])))
    else:
        angle = np.arctan2(np.sin(np.pi/180.*(point1[0] - point2[0]))*np.cos(np.pi/180.*point1[1]), \
                np.cos(np.pi/180.*point2[1])*np.sin(np.pi/180.*point1[1])- \
                np.sin(np.pi/180.*point2[1])*np.cos(np.pi/180.*point1[1])*np.cos(np.pi/180.*(point1[0] - point2[0])))

    #angle = np.arctan2(np.sin(np.pi/180.*(point2[0] - point1[0]))*np.cos(np.pi/180.*point2[1]), \
    #        np.cos(np.pi/180.*point1[1])*np.sin(np.pi/180.*point2[1])- \
    #        np.sin(np.pi/180.*point1[1])*np.cos(np.pi/180.*point2[1])*np.cos(np.pi/180.*(point2[0] - point1[0])))

    return np.degrees(angle) 

def strike_geometry(data):
    """
    get the strike of each fault by calculating the arithmetic mean of azimuth of 
    each straight line segment belongs to each fault.
    """
    # initialization
    strike = pd.Series(dtype=np.float64)
    # iterate through geometries of faults
    for id, row in data.iterrows():
        strike_single = 0# strike angle for a single fault
        # iterate through each neighbour points pair in geometry
        for i_p in range(len(row['geometry'].coords)-1):
            strike_single += azimuth_2(row['geometry'].coords[i_p], row['geometry'].coords[i_p+1])
        strike_single /= len(row['geometry'].coords)# arithmetic mean
        strike.at[id] = strike_single

    data['strike'] = strike

def rake(data):
    """
    assign a 'average_ra' value (int) for each 'slip_type'
    """
    # Input  - data (GeoDataFrame): must contain column 'average_ra' and 'slip_type'
    
    for id, row in data.iterrows():
        if (row['slip_type']=='Dextral'):
            data['average_ra'].iloc[id] = 180.
        elif (row['slip_type']=='Sinistral'):
            data['average_ra'].iloc[id] = 0.
        elif (row['slip_type']=='Dextral-Normal'):
            data['average_ra'].iloc[id] = 202.5
        elif (row['slip_type']=='Dextral-Reverse'):
            data['average_ra'].iloc[id] = 157.5
        elif (row['slip_type']=='Sinistral-Reverse'):
            data['average_ra'].iloc[id] = 22.5
        elif (row['slip_type']=='Sinistral-Normal'):
            data['average_ra'].iloc[id] = 337.5

def nearest_fault(data, EPSG = 4479):
    """
    For each fault, find the its nearest fault and calculate minimum distance in EPSG:4479
    """
    # Input  - data (GeoDataFrame): 
    #        - EPSG (int): EPSG code that represent the map projection applied for distance calculation

    # initialization
    nearest_id = pd.Series(dtype=int)
    nearest_dis = pd.Series(dtype=float)
    geo = data['geometry'].to_crs(EPSG)

    for id, row in geo.iteritems():
        ind, dis = geo[~geo.index.isin([id])].sindex.nearest(row, return_distance=True)
        ind = geo[~geo.index.isin([id])].index[ind[1]][0]# index of nearest fault in the context of whole GeoDataFrame
        nearest_id.at[id] = ind# index of the nearest fault (int)
        nearest_dis.at[id] = dis[0]/1000.# distance (in km) to the nearest fault

    data['nearest_id'] = nearest_id
    data['nearest_dis'] = nearest_dis

def fault_txt2shp(path_txt, path_shp):
    """
    Transform fault data files in .txt format into file in .shp format
    """
    # read .txt files into shapely.LineString
    temp_list_shp = []
    for index, txt_name in enumerate(glob(path_txt+"/*.txt")):
        temp_list_txt = []
        with open(txt_name,'r') as f:
            lines = reader(f, delimiter='\t')
            for line in lines:
                temp_list_txt.append((float(line[0]), float(line[1])))# list in format [(lot, lat), ...]
        temp_list_shp.append(sp.geometry.LineString(temp_list_txt))# list of shapely.LineString
    
    fault_gdf = gpd.GeoDataFrame({'id': np.arange(index+1),
                                  'geometry': temp_list_shp}, 
                                crs='EPSG:4326')
    #fault_gdf.to_file(path_shp+'/dawanqv.shp')
    return fault_gdf


