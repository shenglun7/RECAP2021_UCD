# This contains source code that will be used in extract data from chamber model

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Rectangle


def extract_input_data(filepath):
    """
    extract data from existing input file (.txt) and convert to dataframe
    """

    df = pd.read_csv(filepath,skiprows=3,nrows=92,delim_whitespace=True,header=None)
    df.columns=['Number','Parameter','Value']
    #df['Date'] = 
    return df


def extract_model_10min_avg_outfile(filepath, nspec=616):
    """
    Extract data from results-YYYY-MM-DD-01.txt file from chamber model, get 10min average data for all species.
    input: filepath (str): path to .txt file
           nspec (int): number of species set in the model (default = 616)
    output: df_all (pd.DataFrame): df of 10min average data for all species
    """

    with open(filepath) as f:
        contents = f.readlines()

    date = filepath.split('/')[-1].split('.')[0][8:18]

    # get list of reaction time (every 10min)
    list_time = [s for s in contents if "Results at time" in s]
    list_time = [s.split(' ')[-1].split('\n')[0] for s in list_time]

    # create df to stote all 10min data
    df_all = pd.DataFrame(columns=['Number','Parameter','Value','Time','Date'])

    # get list of species with concentration in df (every 10min)
    for i in range(len(list_time)):
        num_skiprows = 1+1*(i+1)+616*i
        df_10min = pd.read_csv(filepath, skiprows=num_skiprows, nrows=616, header=None,
                               delim_whitespace=True)
        df_10min.columns = ['Number','Parameter','Value']
        df_10min['Time'] = float(list_time[i])
        df_10min['Date'] = pd.to_datetime(date)    
        df_all = pd.concat([df_all, df_10min], axis=0)
    
    return df_all


def extract_model_outfile(filepath, NOx_add):
    """
    Extract data from .out file from chamber model, 
    get initial concentration of NO, NOx, VOC, OFP, O3, and 
    final concentration of NO, NO2, O3

    input: filepath (str): path to .out file
           NOx_add (int): NOx_add set in the box model (determine the number of results per day)
    return: df, columns=['Date','O3_init','NO_init','NOx_init','VOC_init',
                         'OFP_init','O3_final','NO_final','NO2_final']
    """

    with open(filepath,'r') as f:
        contents = f.readlines()

        # list of initial concentrations
        list_date = [s for s in contents if "reading from file1" in s]
        list_init = [s for s in contents if 'Initial O3, NO, NO2(ppm)' in s]
        list_initvoc = [s for s in contents if 'Initial NOx, VOC(ppm)' in s]
        list_initOFP = [s for s in contents if 'Initial OFP(ppm)' in s]
        # list of final concentrations
        list_finalO3 = [s for s in contents if "Final O3(ppm)" in s]
        list_finalNOx = [s for s in contents if 'Final NO, NO2(ppm)' in s]

        df = pd.DataFrame(columns=['Date','O3_init','NO_init','NOx_init','VOC_init',
                                   'OFP_init','O3_final','NO_final','NO2_final'])
        
        for i in range(len(list_date)):
            df.loc[i,'Date'] = pd.to_datetime(list_date[i].split('_')[2], format='%Y%m%d')
            df.loc[i,'O3_init'] = float(list_init[i*NOx_add].split(' ')[-1].split(',')[0])
            df.loc[i,'NO_init'] = float(list_init[i*NOx_add].split(' ')[-1].split(',')[1])
            df.loc[i,'NOx_init'] = \
                float(list_initvoc[i*NOx_add].split(' ')[-1].split(',')[0])
            df.loc[i,'VOC_init'] = \
                float(list_initvoc[i*NOx_add].split(' ')[-1].split(',')[1])
            df.loc[i,'OFP_init'] = float(list_initOFP[i*NOx_add].split(' ')[-1])
            df.loc[i,'O3_final'] = float(list_finalO3[i*NOx_add].split(' ')[-1])
            df.loc[i,'NO_final'] = \
                float(list_finalNOx[i*NOx_add].split(' ')[-1].split(',')[0])
            df.loc[i,'NO2_final'] = \
                float(list_finalNOx[i*NOx_add].split(' ')[-1].split(',')[1])

        # convert concentration unit from ppm to ppb
        df.sort_values(by='Date', inplace=True)     
        df = df.set_index('Date')
        df = df.apply(lambda x: x.astype(float)*1000)
        df = df.reset_index(drop=False)

        return df


def get_model_O3_sensitivity(filepath_basecase, filepath_NOx_addition, NOx_add=5):
    """
    Extract basecase model results and modeled O3 sensitivity data, ready for time series plot
    input: filepath_basecase (str): filepath for basecase out file
           filepath_NOx_addition (str): filepath for sensitivity out file
    return: df
    """

    df_basecase = extract_model_outfile(filepath_basecase, NOx_add)
    df_NOx_addition = extract_model_outfile(filepath_NOx_addition, NOx_add)

    df = pd.merge(df_basecase, df_NOx_addition[['Date','O3_final','NOx_init']], 
                  on='Date', how = 'left', suffixes=('','_NOx_addition'))
    df['delta_O3'] = df['O3_final_NOx_addition'] - df['O3_final'] 

    return df

def merge_measure_model_data(df_chamber, df_temp, df_MDA8, df_model, loc):
    """
    Merge data from measurements (chamber measurement, ambient temperature, and MDA8 O3)
    with chamber model results
    input: 
    return: df
    """

    df_measure = \
        df_chamber[['Date','bag2_3hr','b1_b2_3hr','b3_b2_3hr']].\
            merge(df_temp[['Date','Value']], on='Date', how='right').\
                merge(df_MDA8[['Date','MDA8','Month','Week']], on='Date', how='left')
    df_measure.columns = ['Date','bag2_3hr','b1_b2_3hr','b3_b2_3hr',
                          'Temp','MDA8','Month','Week']

    df_measure_model = df_measure. \
        merge(df_model[['Date','O3_final','delta_O3']], on='Date', how='left')
    
    if loc.lower() == 'pasadena':
        df_measure_model = df_measure_model[(df_measure_model['Date']>='2021-07-16')&
                                            (df_measure_model['Date']<='2021-10-31')]
    elif loc.lower() == 'redlands':
        df_measure_model = df_measure_model[(df_measure_model['Date']>='2021-07-10')&
                                            (df_measure_model['Date']<='2021-10-31')]
    else:
        print('Please specify location as Pasadena or Redlands')
        return
    
    return df_measure_model


def get_voc_available_date_list(loc):
    """
    Get a list of dates that VOC measurement is available (for Pasadena: VOC measurement is from NOAA; for Redlands: VOC measurement is from CARB PTR-MS)
    """

    if loc.lower() == 'pasadena':
        date=['2021-08-07','2021-08-08','2021-08-09','2021-08-10','2021-08-11',
              '2021-08-12','2021-08-13','2021-08-14','2021-08-15','2021-08-16',
              '2021-08-17','2021-08-18','2021-08-19','2021-08-20','2021-08-21',
              '2021-08-22','2021-08-23','2021-08-24','2021-08-25','2021-08-26',
              '2021-08-27','2021-08-29','2021-09-01','2021-09-02','2021-09-03',
              '2021-09-04','2021-09-05','2021-09-06']  
    elif loc.lower() == 'redlands':
        date=['2021-08-14','2021-08-15','2021-08-16','2021-08-19','2021-09-02',
              '2021-09-03','2021-09-04','2021-09-05','2021-09-06','2021-09-07',
              '2021-09-08','2021-09-14','2021-09-15','2021-09-16','2021-09-18',
              '2021-09-19','2021-09-20','2021-09-21','2021-09-22','2021-09-23',
              '2021-09-24','2021-10-08','2021-10-09','2021-10-17']
    date = pd.to_datetime(date)
    return date


###################################################
# plot specific figures for moeasurement vs model comparison
###################################################
def plot_timeseries_measure_model(df_plot, loc):
    """
    Plot the time sereis of measured and modeled O3 for Pasadena
    input: df_plot (df): dataframe with measured and modeled O3
              loc (str): location of the chamber measurement
    return: fig
    """
    loc = loc.lower()
    fig, ax=plt.subplots(2,1, figsize=(6,5.5))

    ## plot modeled delta O3
    # measurement
    ax[0].scatter(df_plot['Date'], df_plot['b1_b2_3hr'], s=20, marker='o')
    ax[0].plot(df_plot['Date'], df_plot['b1_b2_3hr'], label='measurement')
    # model
    x = df_plot['Date']
    y = df_plot['delta_O3']
    ax[0].scatter(x,y, s=20, marker='o')
    ax[0].plot(x,y, label='model')
    ax[0].axhline(y=0, linestyle='--', color='k')

    ## plot modeled final O3
    # measurement
    ax[1].scatter(df_plot['Date'], df_plot['bag2_3hr'], s=20, marker='o')
    ax[1].plot(df_plot['Date'], df_plot['bag2_3hr'], label='measurement')
    # model
    x = df_plot['Date']
    y = df_plot['O3_final']
    ax[1].scatter(x,y, s=20, marker='o')
    ax[1].plot(x,y, label='model')

    ## plot shaded area (indicate VOC measurement available days)
    list_date = get_voc_available_date_list(loc)
    for i in range(len(list_date)):
        for j in range(2):
            ax[j].axvspan(list_date[i]-pd.offsets.DateOffset(hours=12),
                          list_date[i]+pd.offsets.DateOffset(hours=12),
                          alpha=0.4, facecolor='skyblue')

    ## appearance
    ax[0].set_ylim(-20,40)
    ax[1].set_ylim(0,180)
    ax[1].set_yticks([0, 30, 60, 90, 120, 150, 180])
    ax[0].set_ylabel('$\Delta O_3^{+NO_x}$ (ppbv)', labelpad=-1)
    ax[1].set_ylabel('Final O$_3$ (ppbv)')
    #ax[1].set_xlabel('Date')
    for i in range(2):
        ax[i].legend(loc='upper right')
        ax[i].set_xlim((pd.to_datetime('2021-07-05'),pd.to_datetime('2021-11-05')))
        ax[i].set_box_aspect(1/2.5)    
        for tick in ax[i].get_xticklabels():
            tick.set_rotation(30)
    
    # add title
    if loc == 'pasadena':
        ax[0].set_title('(a) $\Delta O_3^{+NO_x}$ - Pasadena', fontweight = 'bold')
        ax[1].set_title('(c) final O$_3$ - Pasadena', fontweight = 'bold')
    elif loc == 'redlands':
        ax[0].set_title('(a) $\Delta O_3^{+NO_x}$ - Redlands', fontweight = 'bold')
        ax[1].set_title('(c) final O$_3$ - Redlands', fontweight = 'bold')         
    
    plt.tight_layout()
    plt.show()
    return fig


def plot_scatter_measure_model_pasadena(df_plot, loc):
    """
    
    """

    fig, ax=plt.subplots(2,1, figsize=(3,5.5), sharey=False, sharex=False)

    x = df_plot['b1_b2_3hr']
    y = df_plot['delta_O3']
    mask = ~np.isnan(x) & ~np.isnan(y)
    r_value, _ = pearsonr(x[mask], y[mask])
    ax[0].text(-17,16, f'R = {r_value:.2f}', color='#1f77b4')
    ax[0].scatter(x,y, label='measured VOC available', s=20, marker='o')
    ax[0].plot([-20, 20], [-20, 20], transform=ax[0].transAxes, ls="--", c=".3")
    ax[0].set_xlim(-20,20)
    ax[0].set_ylim(-20,20)
    ax[0].set_yticks(np.arange(-20,20.1,10))
    ax[0].set_xticks(np.arange(-20,20.1,10))
    ax[0].set_xlabel('Measured $\Delta O_3^{+NO_x}$ (ppbv)')
    ax[0].set_ylabel('Modeled $\Delta O_3^{+NO_x}$ (ppbv)', labelpad=-1)
    ax[0].set_title('(b) $\Delta O_3^{+NO_x}$ - Pasadena', fontweight='bold')
    ax[0].set_box_aspect(1)

    x = df_plot['bag2_3hr']
    y = df_plot['O3_final']
    mask = ~np.isnan(x) & ~np.isnan(y)
    r_value, _ = pearsonr(x[mask], y[mask])
    ax[1].text(10,180,f'R = {r_value:.2f}', color='#1f77b4')
    ax[1].scatter(x,y, label='Model', s=20, marker='o')
    ax[1].plot([0, 180], [0, 180], transform=ax[1].transAxes, ls="--", c=".3")
    ax[1].set_xlim(0,200)
    ax[1].set_ylim(0,200)
    ax[1].set_yticks(np.arange(0,200.1,50))
    ax[1].set_xticks(np.arange(0,200.1,50))    
    ax[1].set_xlabel('Measured final O$_3$ (ppbv)')
    ax[1].set_ylabel('Modeled final O$_3$ (ppbv)')
    ax[1].set_title('(d) final O$_3$ - Pasadena', fontweight='bold')
    ax[1].set_box_aspect(1)
    
    plt.tight_layout()
    plt.show()
    return fig


def plot_scatter_measure_model_redlands(df_plot):
    """
    
    """

    list_date = get_voc_available_date_list('Redlands')
    fig, ax=plt.subplots(2,1, sharey=False, sharex=False, figsize=(4,5))

    # plot all days
    x = df_plot['b1_b2_3hr']
    y = df_plot['delta_O3']
    mask = ~np.isnan(x) & ~np.isnan(y)
    r_value, _ = pearsonr(x[mask], y[mask])
    ax[0].text(-28,24, f'R = {r_value:.2f}', color='#1f77b4')
    ax[0].scatter(x,y, label='all days', color='#1f77b4', s=20, marker='o')
    ax[0].plot([-30, 30], [-30, 30], transform=ax[0].transAxes, ls="--", c=".3")

    # plot blue shaded days
    x = df_plot[df_plot['Date'].isin(list_date)]['b1_b2_3hr']
    y = df_plot[df_plot['Date'].isin(list_date)]['delta_O3']
    mask = ~np.isnan(x) & ~np.isnan(y)
    r_value, _ = pearsonr(x[mask], y[mask])
    ax[0].text(-28,18, f'R = {r_value:.2f}', color='#ff7f0e')
    ax[0].scatter(x,y, label='days with \n PTR-MS', color='#ff7f0e', s=15, marker='o')

    ax[0].set_xlim(-30,30)
    ax[0].set_ylim(-30,30)
    ax[0].set_yticks(np.arange(-30,30.1,10))
    ax[0].set_xticks(np.arange(-30,30.1,10))    
    ax[0].set_xlabel('Measured $\Delta O_3^{+NO_x}$ (ppbv)')
    ax[0].set_ylabel('Modeled $\Delta O_3^{+NO_x}$ (ppbv)', labelpad=-1)
    ax[0].set_title('(b) $\Delta O_3^{+NO_x}$ - Redlands', fontweight='bold')
    ax[0].set_box_aspect(1)

    ## plot final O3
    # plot all days
    x2 = df_plot['bag2_3hr']
    y2 = df_plot['O3_final']
    mask = ~np.isnan(x2) & ~np.isnan(y2)
    r_value, _ = pearsonr(x2[mask], y2[mask])
    ax[1].text(10,200,f'R = {r_value:.2f}', color='#1f77b4')
    ax[1].scatter(x2,y2, label='all days', color='#1f77b4', s=15, marker='o')
    ax[1].plot([0, 120], [0, 120], transform=ax[1].transAxes, ls="--", c=".3")

    # plot blue shaded days
    x2 = df_plot[df_plot['Date'].isin(list_date)]['bag2_3hr']
    y2 = df_plot[df_plot['Date'].isin(list_date)]['O3_final']
    mask = ~np.isnan(x2) & ~np.isnan(y2)
    r_value, _ = pearsonr(x2[mask], y2[mask])
    ax[1].text(10,176,f'R = {r_value:.2f}', color='#ff7f0e')
    ax[1].scatter(x2,y2, label='days with \n PTR-MS', color='#ff7f0e', s=15, marker='o')
    ax[1].set_xlim(0,220)
    ax[1].set_ylim(0,220)
    ax[1].set_yticks(np.arange(0,200.1,50))
    ax[1].set_xticks(np.arange(0,200.1,50))  
    ax[1].set_xlabel('Measured final O$_3$ (ppbv)')
    ax[1].set_ylabel('Modeled final O$_3$ (ppbv)')
    ax[1].set_title('(d) final O$_3$ - Redlands', fontweight='bold')
    ax[1].set_box_aspect(1)

    for i in range(2):
        ax[i].legend(loc='lower right', fontsize=8, borderpad=0.1, labelspacing=0, bbox_to_anchor=(0.6, 0., 0.5, 0.5))
        
    plt.tight_layout()
    plt.show()

    return fig


###################################################
# Treat Isopleth files
###################################################

def get_initial_NOx_VOC_isoplethfile(filepath):
    """
    Get initial NOx and VOC concentration from output isopleth file
    input: filepath (str): path to isopleth output file
    return: NOx_init, VOC_init (float): initial NOx and VOC concentration, unit ppb
    """
    
    with open(filepath, 'r') as f:    
        contents = f.readlines()
    
    # get initial NOx, VOC in ppb (original unit is ppm)
    NOx_init = float(contents[0][37:48])*1e3
    VOC_init = float(contents[0][51:62])*1e3
    df_init = pd.DataFrame({'NOx':[NOx_init],'VOC':[VOC_init]}, index=[0])
    return df_init


def get_grid_cell_O3_isoplethfile(filepath):
    """
    Get df of grid cell O3 data from one output isopleth file (in one day)
    input: filepath (str): path to isopleth output file
    return: df (pd.DataFrame): df of grid cell O3 data, unit ppb
    """

    df_iso = pd.read_csv(filepath, skiprows=1, header=0)
    df_iso.columns = ['O3','ivoc','inox']
    df_iso['VOC'] = df_iso['ivoc']*1e3
    df_iso['NOx'] = df_iso['inox']*1e3
    df_iso = df_iso[['O3','NOx','VOC']]
    
    return df_iso


def extract_isopleth_output(folderpath, loc):
    """
    Extract data from a list of output file of drawing isopleth program
    Data includes measured initial NOx and VOC level, and NOx and grid cell (NOx and VOC) data of O3 isopleth
    
    input: filepath (str): folder that stores all isopleth_*.csv file
           loc (str): location of site, e.g. 'Redlands'
    return:   
    """
    # get list of isopleth_*.csv file
    loc = loc.lower()
    filelist = [f for f in os.listdir(folderpath) 
                if f.endswith(loc+'.csv') and f.startswith('isopleth')]
    
    # create blank df
    df_iso_all = pd.DataFrame()
    df_init_all = pd.DataFrame()

    # write data to df
    for i in range(len(filelist)):
        date = pd.to_datetime(filelist[i].split('_')[1])
        filepath = folderpath+filelist[i]

        # get initial NOx, VOC
        df_init = get_initial_NOx_VOC_isoplethfile(filepath)
        df_init['Date'] = date
        df_init_all = pd.concat([df_init_all, df_init], axis=0)
    
        # get grid data for O3 isopleth
        df_iso = get_grid_cell_O3_isoplethfile(filepath)
        df_iso['Date'] = date
        df_iso_all = pd.concat([df_iso_all, df_iso], axis=0)
    
    df_init_all = df_init_all.reset_index(drop=True)[['Date','NOx','VOC']]
    df_iso_all = df_iso_all.reset_index(drop=True)[['Date','O3','NOx','VOC']]
    
    return df_init_all, df_iso_all


def get_date_list_isopleth(df_init, df_measure_model):
    """
    Get list of date that has both model and measured data
    input: df_init (pd.DataFrame): initial NOx and VOC concentration
           df_measure_model (pd.DataFrame): measured and model data
    return: list_date (list): list of date that has both model and measured data
    """

    df_measure_model

    list_date = []
    for i in range(len(df_init)):
        date = df_init['Date'][i]

        if date in df_measure_model['Date'].unique():
            list_date.append(date)
    return list_date


#########################################################
# functions for drawing O3 isopleth
#########################################################

def draw_line_from_two_points(x1,y1,x2,y2):
    """
    This function draw a line from 2 points (x1,y1) and (x2,y2)
    input: x1,y1,x2,y2 (type: float)
    return: a, b (y = ax + b), (type: float) 
    """
    a=(y1-y2)/(x1-x2)
    b=y1-a*x1
    return a, b


def line_intersection(x1,y1,x2,y2,x3,y3,x4,y4):
    """
    This function get intersection point of 2 line generated 
    from 2 pairs of dots p1&p2 and p3&p4
    input: a1,b1,a2,b3 (type: float): coef of two lines (y=ax+b)
    return: x_int,y_int (type: float): X and Y of intersection
    """
    a1, b1 = draw_line_from_two_points(x1,y1,x2,y2)
    a2, b2 = draw_line_from_two_points(x3,y3,x4,y4)
    y_int = (b2*(a1/a2)-b1)/(a1/a2-1)
    x_int = (y_int-b1)/a1 

    return x_int,y_int


def get_CI_T_distribution(list_of_value, alpha=0.95):
    """
    Get confidence interval of a list of value (can contains NaN)
    input: list_of_value (list): list of value
           alpha (float): significance level (default = 0.95)
    return: CI (list): confidence interval    
    """

    # remove nan
    list_of_value = [x for x in list_of_value if ~np.isnan(x)]
    std = np.std(list_of_value)
    mean = np.mean(list_of_value)
    CI = stats.t.interval(alpha, len(list_of_value)-1, loc=mean, scale=std)
    CI = list(CI)

    return CI

    

#########################################################
# Only useful to analyze chamber model for ITS report
#########################################################

def plot_isopleth_in_mar_aug(df_init_NOx_VOC, df_O3_isopleth, loc):
# def to plot isopleth for March and Augest side by side
# input file df_iso_all is grid cell data, a output from def make_df_for_isopleth()
    
    # create date list
    date_list_1 = df_O3_isopleth[df_O3_isopleth['Date'].dt.month==3]['Date']
    date_list_2 = df_O3_isopleth[df_O3_isopleth['Date'].dt.month==8]['Date']
    date_list = [date_list_1, date_list_2]

    # set resolution
    VOC_max = 700
    NOx_max = 50
    step = 20
    grid_VOC = VOC_max/step
    grid_NOx = NOx_max/step

    fig,ax=plt.subplots(1,2, figsize=(8,4), subplot_kw={'aspect': VOC_max/NOx_max})

    for i in range(2):
        # plot contour line of O3 isopleth 
        # (X axis: VOC, Y axis: NOx, vg is a 2D matrix of O3 concentration at (x, y))
        df_contour = df_O3_isopleth[df_O3_isopleth['Date'].isin(date_list[i])].\
            groupby(['NOx','VOC']).mean().reset_index(drop=False)
        xg = np.arange(grid_VOC, VOC_max+grid_VOC, grid_VOC) 
        yg = np.arange(grid_NOx, NOx_max+grid_NOx, grid_NOx)  # resolution of NOx
        vg = df_contour.pivot(index='NOx', columns='VOC', values='O3').to_numpy()
        cax1 = ax[i].contour(xg, yg, vg, cmap='viridis', levels=8)  # contour line
        cax2 = ax[i].contourf(xg, yg, vg, cmap='viridis', levels=8)  # contour area
     
        # plot ridge line
        x_int=np.zeros(7)
        y_int=np.zeros(7)
        for j in np.arange(4,7,1):
            polys0 = cax1.allsegs[j]
            x00, y00 = polys0[0].T
            #ax[i].scatter(x00[0], y00[0])
            x_int[j-1],y_int[j-1] = \
                line_intersection(x00[1],y00[1],x00[4],y00[4],
                                  x00[-4],y00[-4], x00[-1],y00[-1])
        x_int=x_int[1:]
        y_int=y_int[1:]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_int,y_int)
        x_ridge = np.arange(0,VOC_max+200,10)
        line_ridge=slope*x_ridge+intercept+1
        ax[i].plot(x_ridge, line_ridge, c='k', 
                   linestyle='dashed', label='ridge line')
        #ax[i].scatter(x_int,y_int,c='k', linestyle='-')
        
        # add monthly-averaged point
        df_int_month = df_init_NOx_VOC[df_init_NOx_VOC['Date'].isin(date_list[i])]
        NOx_avg = stats.trim_mean(df_int_month['NOx'], 0.05)
        VOC_avg = stats.trim_mean(df_int_month['VOC'], 0.05)
        ax[i].scatter(VOC_avg, NOx_avg, c='k', s=10 ,label='Measured average')
        ax[i].axhline(y=NOx_avg, color='k', linestyle=':')
        ax[i].axvline(x=VOC_avg, color='k', linestyle=':')
        print('Averaged measured VOC and NOx (ppb) are: ',VOC_avg, NOx_avg)

        # add 95% confidence interval
        log_NOx = np.log(df_int_month['NOx'])
        log_VOC = np.log(df_int_month['VOC'])
        NOx_CI = get_CI_T_distribution(log_NOx, alpha=0.95)
        VOC_CI = get_CI_T_distribution(log_VOC, alpha=0.95)
        NOx_CI = np.exp(NOx_CI)
        VOC_CI = np.exp(VOC_CI)
        print('CI for NOx: ', NOx_CI[0], ', ', NOx_CI[1])
        print('CI for VOC: ', VOC_CI[0], ', ', VOC_CI[1])
        ax[i].add_patch(Rectangle((VOC_CI[0],NOx_CI[0]), VOC_CI[1]-VOC_CI[0], NOx_CI[1]-NOx_CI[0], linewidth=2, edgecolor='k', facecolor='none'))

        # add title to show NOx reduction
        NOx_control_avg = (NOx_avg-VOC_avg*slope-intercept)
        NOx_control_percentage_avg = (NOx_avg-VOC_avg*slope-intercept)/NOx_avg*100
        NOx_control_edge = (NOx_CI[1]-VOC_CI[1]*slope-intercept)
        NOx_control_percentage_edge = (NOx_CI[1]-VOC_CI[1]*slope-intercept)/NOx_CI[1]*100

        ax[i].annotate(f"$NO_x$ control on mean: {round(NOx_control_avg,2)} ppbv, {round(NOx_control_percentage_avg,1)}%", xy=(0.0, 1.08), xycoords='axes fraction', color='k', size=9) 
        ax[i].annotate(f"$NO_x$ control on edge: {round(NOx_control_edge,1)} ppbv, {round(NOx_control_percentage_edge,1)}%", xy=(0.0, 1.02), xycoords='axes fraction', color='k', size=9) 

        # appearance
        ax[i].set_ylim(0, NOx_max+grid_NOx)
        ax[i].set_xlim(0, VOC_max+grid_VOC)
        ax[i].set_xlabel('OFP (ppbv)')
        ax[i].set_ylabel('NO$_x$ concentration (ppbv)')
        title = ['(March)','(August)']
        ax[i].set_title(loc+' '+title[i], pad=30, fontweight='bold')
    
    # add one color bar
    ax[1].legend(loc='center left', bbox_to_anchor=(1.0, 1.0), facecolor="gray")
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.52])
    cbar = plt.colorbar(cax2, cax=cbar_ax)
    cbar.set_label('O$_3$ concentraion (ppbv)')

    plt.show()
    return fig 


######################################################
# extract source apportionment model results
######################################################
def extract_bgX3_outfile(filepath):
    """
    extract data from .out file of bgX3 model output
    input: filepath (str): path to .out file
    return: df_O3
    """

    # get date and O3 data to list
    list_date = []
    list_O3 = []
    with open(filepath, 'r') as f:
        save_next_line = False
        for line in f:
            if "date string" in line:
                list_date.append(line.strip())

            if "O3 initial concentration" in line:
                list_O3.append(line.strip())
            elif "final O3" in line:
                save_next_line = True
                list_O3.append(line.strip())
            elif save_next_line:
                save_next_line = False
                list_O3.append(line.strip())

    # fill date and O3 data in df             
    df_O3 = pd.DataFrame(columns=['Date','O3_init','O3_vanilla',
                                  'O3_X1','O3_X2','O3_X3'], 
                         index=range(len(list_date)))

    for i in range(len(list_date)):
        date = pd.to_datetime(list_date[i][26:34])
        df_O3.loc[i,'Date'] = date
        df_O3.loc[i,'O3_init'] = float(list_O3[i*3+0].split()[3])*1e3
        df_O3.loc[i,'O3_vanilla'] = float(list_O3[i*3+1].split()[2])*1e3
        df_O3.loc[i,'O3_X1'] = float(list_O3[i*3+1].split()[3])*1e3
        df_O3.loc[i,'O3_X2'] = float(list_O3[i*3+1].split()[4])*1e3
        df_O3.loc[i,'O3_X3'] = float(list_O3[i*3+2].split()[0])*1e3    
    df_O3['O3_final'] = df_O3['O3_vanilla']+df_O3['O3_X1']+df_O3['O3_X2']+df_O3['O3_X3']

    # clean df
    df_O3 = df_O3[['Date','O3_init','O3_final','O3_vanilla','O3_X1','O3_X2','O3_X3']]
    df_O3.iloc[:,1:] = df_O3.iloc[:,1:].astype(float)
    
    return df_O3


def merge_all_bgX3_outfile(list_filepath, list_source):
    """
    Merge all bgX3 model output files in a folder
    input: folderpath (str): path to folder that stores all bgX3 model output files
    return: df_O3
    """

    df_O3 = pd.DataFrame()
    for i in range(len(list_filepath)):
        filepath = list_filepath[i]
        df_O3 = extract_bgX3_outfile(filepath)
        df_O3.columns = ['Date','O3_init','O3_final','O3_vanilla'] + list_source[i]
        if i == 0:
            df_O3_merge = df_O3
        else:
            df_O3_merge = pd.merge(df_O3_merge, df_O3, on='Date', how='right')

    list_source_flatten = [item for sublist in list_source for item in sublist]        
    list_column = ['Date','O3_init','O3_final'] + list_source_flatten
    df_O3_merge = df_O3_merge[list_column]
    df_O3_merge.loc[:,'Background'] = \
        df_O3_merge.loc[:,'O3_final'] - df_O3_merge.iloc[:,3:].sum(axis=1) + \
            df_O3_merge.loc[:,'H2O2']  # not show H2O2 as source as it's too low

    return df_O3_merge


def plot_timeseries_source_O3(df_O3, list_of_source, loc):
    """
    Plot time series of source apportionment results
    input: df_O3 (pd.DataFrame): df of all source apportionment results
           list_of_source (list): list of source names shown in the plot
           loc (str): location of the chamber measurement e.g. 'Redlands'
    return: fig
    """

    columns_to_plot = ['Date'] + list_of_source
    df_plot = df_O3[columns_to_plot].copy()

    fig,ax=plt.subplots(1, figsize=(6,6))
    colors = plt.cm.tab20.colors[:15] 
    df_plot.plot.area(ax=ax, x='Date', stacked=True, color=colors)
    plt.legend(ncol=1, loc='upper left', fontsize=6, bbox_to_anchor=(1.05, 1))
    ax.set_box_aspect(1/2)
    plt.title(loc, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('$O_3$ concentration (ppbv)')
    plt.ylim(0,200)
    plt.yticks([0, 30, 60, 90, 120, 150, 180])
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.show()
    return fig


def plot_pie_source_O3(df_O3, list_of_source, loc, with_bg=True):
    """
    
    
    """

    # Create a pie chart using the mean values
    fig, ax=plt.subplots(1, figsize=(6,3))

    mean_value = df_O3[list_of_source].mean()
    x = mean_value.index.tolist()
    y = mean_value.values.tolist()
    porcent = [s/sum(y)*100 for s in y]
    if with_bg:
        colors = plt.cm.tab20.colors[0:15]
    else:
        colors = plt.cm.tab20.colors[1:15]
    
    patches, texts = ax.pie(mean_value, startangle=90, colors=colors, radius=1.2)
    # add label in legend with source name and precentage
    labels = ['{0} - {1:1.1f} %'.format(i,j) for i,j in zip(x, porcent)]
    plt.legend(patches, labels, loc='center right', bbox_to_anchor=(0, 0.5),
               fontsize=8)

    ax.set_box_aspect(1)
    plt.title(loc, fontweight='bold')
    fig.set_facecolor('white')

    plt.show()
    return fig

