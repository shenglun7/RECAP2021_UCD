# This contains source code that will be used in analyze the raw chamber data (NOx,
# NOy, O3, temperature, pressure, flow rate, valve status etc.)

import pandas as pd
import numpy as np
from scipy import stats
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def list_of_measurement_day(loc, test_type='test'):
    """
    Get a list of measurement days for a given location.
    input: loc (str): 'Redlands' or 'Pasadena'
           test_type (str): 'test': chamber measurement day
                            'no test': no measurement day
                            'qa/qc': QA/QC test day
                            'wall loss': wall loss test day 
    return: date_list (list): list of measurement days (object)    
    """
    
    if loc == 'Redlands':
        log_sheet = pd.read_csv('../../data/raw/Chamber/log_sheet_redlands.csv')
    elif loc == 'Pasadena':
        log_sheet = pd.read_csv('../../data/raw/Chamber/log_sheet_pasadena.csv')
    else:
        print('Location not found!')
        return
    
    if test_type == 'test' or test_type == 'no test':
        list_date = pd.to_datetime(log_sheet.loc[log_sheet['Test']==test_type, 'Date']).\
                                   dt.strftime("%Y-%m-%d").to_list()
        return list_date
    elif test_type == 'qa/qc' or test_type == 'wall loss':
        list_date = log_sheet.loc[log_sheet['Test']==test_type, 'Date'].to_list()
        return list_date
    else:
        print('test_type not found!')
        return


def get_raw_lvm(filepath_lvm):
    """
    Get chamber-system data from .lvm file.
    input: filename_lvm (str): path to .lvm file
    output: df_lvm
    """
    
    # open .lvm file
    df_lvm = pd.read_csv(filepath_lvm, skiprows=21, sep='\t')
    loc = filepath_lvm.split('/')[5]

    # clean df_lvm
    col_name = df_lvm.columns[df_lvm.columns.str.contains('Untitled')].tolist()+['Comment']
    df_lvm = df_lvm.loc[:,col_name]
    df_lvm.columns = ['NO','dif','NOy','NOx','NO_NOx','NO2','bag1','bag2','bag3',
                      'chamber','amb','Time']
    df_lvm = df_lvm[['NO','dif','NOy','bag1','bag2','bag3','chamber','amb','Time']]
    df_lvm['Time'] = pd.to_datetime(df_lvm['Time'])

    # calibration equation for NOy monitor (done by CARB in Sacramento)
    if loc == 'Redlands':      
        df_lvm['NOy'] = (df_lvm['NOy']+0.0069)/0.9746
        df_lvm['NO'] = (df_lvm['NO']+0.0055)/0.9182
    elif loc == 'Pasadena':
        df_lvm['NOy'] = (df_lvm['NOy']+0.0036)/0.7702
        df_lvm['NO'] = (df_lvm['NO']+0.0032)/0.7783
    else:
        print('Location not found!')
        return
       
    return df_lvm


def get_raw_NOx(filepath_NOx, correction=True):
    """
    Get chamber-system measured NOx data from raw NOx data file.
    input: filepath_NOx (str): file path of raw NOx data file
           correction (bool): whether to apply additional correction for NOx in 
                              Redlands (default: True)
    return: df_NOx
    """

    # get folder path
    folder_path = filepath_NOx[:36]
    file_name = filepath_NOx[36:]

    # if file exists, read it, else create a blank df
    if file_name in os.listdir(folder_path): 
        df_NOx = pd.read_csv(filepath_NOx, skiprows=12, sep='\t')
        df_NOx = df_NOx.iloc[:,:-1]
        df_NOx.columns = ['Time','NO','NOx','NO2']
        df_NOx['Time'] = pd.to_datetime(df_NOx['Time'], format='%d/%m/%Y %H:%M:%S.%f')
        
        # calibration equation for NOx monitor (done by CARB in Sacramento)
        loc = filepath_NOx.split('/')[5]
        if loc == 'Redlands':
            df_NOx['NO'] = (df_NOx['NO']+0.0057)/0.9325
            df_NOx['NOx'] = (df_NOx['NOx']+0.0055)/0.8651
        elif loc == 'Pasadena':
            df_NOx['NO'] = (df_NOx['NO']+0.0022)/0.8162
            df_NOx['NOx'] = (df_NOx['NOx']+0.0034)/0.8380
        else:
            print('Location not found!')
            return
        # apply additional correction for NOx in Redlands 
        # (got from side-by-side comparison test in Pasadena)
        if correction:
             if loc == 'Redlands':
                df_NOx['NOx'] = df_NOx['NOx']*0.82 - 2.90

        df_NOx['NO2'] = df_NOx['NOx'] - df_NOx['NO']
    else:
        print('No NOx file!')
        df_NOx = pd.DataFrame({'Time':np.nan,'NO':np.nan,
                               'NOx':np.nan,'NO2':np.nan}, index=[0])
    return df_NOx


def get_raw_O3(filepath_O3):
    """
    Get chamber-system measured O3 data from raw O3 data file.
    input: filepath_O3 (str): file path of raw O3 data file
    return: df_O3    
    """
    
    df_O3 = pd.read_csv(filepath_O3, skiprows=4)
    df_O3 = df_O3.iloc[:,1:]
    df_O3.columns = ['O3','Temp','Pressure','Flow rate','Date','Time']
    
    # filter out data from other dates
    if '2021' in df_O3.loc[0,'Date']:
        df_O3['Date'] = pd.to_datetime(df_O3['Date'], format='%d/%m/%Y').\
                            dt.strftime("%Y-%m-%d")
    else:
        df_O3['Date'] = pd.to_datetime(df_O3['Date'], format='%d/%m/%y').\
                            dt.strftime("%Y-%m-%d")
    date = pd.to_datetime(str(filepath_O3.split('/')[-1][:10]), 
               format='%Y_%m_%d').strftime("%Y-%m-%d")
    df_O3 = df_O3.loc[df_O3['Date']==date,:].reset_index(drop=True)    
    
    # combine date and time (note: year format could be 2021 or 21)
    df_O3['Time'] = pd.to_datetime(df_O3['Date']+' '+df_O3['Time'], 
                                   format='%Y-%m-%d %H:%M:%S')
    return df_O3

def get_NOxNOyO3_filepath(date, para , loc):
    """
    Get file path of raw data file for a given date and parameter
    input: date (str): date of interest
           para (str): parameter of interest (NOx, NOy, or O3)
           loc (str): location of interest (Pasadena or Redlands)
    return: file path of raw data file
    """

    # format date for different parameters' files
    date = pd.to_datetime(date)
    date_str = date.strftime(format='%m%d%y')
    date_str_O3 = date.strftime(format='%Y_%m_%d')    
    
    if para == 'NOy':
        if loc == 'Pasadena':
            filename = '../../data/raw/Chamber/'+loc+'/' + para + '/CT_' + date_str + '.lvm'
        elif loc == 'Redlands':
            filename = '../../data/raw/Chamber/'+loc+'/' + para + '/RL_' + date_str + '.lvm'
    elif para == 'NOx':
        filename = '../../data/raw/Chamber/'+loc+'/' + para + '/Measurement_' + date_str + '.txt'
    elif para == 'O3':
        filename = '../../data/raw/Chamber/'+loc+'/' + para + '/' + date_str_O3 + '.csv'
    else:
        print('Location or Parameter not found')
        return
    return filename


def get_combined_initial_data(date, loc):
    """
    Get ambient air NOx, NOy, O3 before chamber experiment as initial condition of basecase chamber.
    input: date (object): only one date, format: '2021-01-01' 
           loc (str): 'Redlands' or 'Riverside'
    return: df: columns = ['Time','NO_x', 'NO2', 'NOx', 'NO_y', 'NOy', 'O3', 
                           'Temp','Pressure', 'Flow rate', 'amb', 'chamber', 'bag1', 
                           'bag2', 'bag3']
    """

    filename_lvm = get_NOxNOyO3_filepath(date, 'NOy', loc)
    filename_NOx = get_NOxNOyO3_filepath(date, 'NOx', loc)
    filename_O3 = get_NOxNOyO3_filepath(date, 'O3', loc)
    df_lvm = get_raw_lvm(filename_lvm)
    df_NOx = get_raw_NOx(filename_NOx)
    df_O3 = get_raw_O3(filename_O3)

    # get start time of chamber experiment and minor 5 minutes (aviod valve change effect)
    time_start_chamber = df_lvm[df_lvm['chamber']==1].iloc[0,:]['Time'] - \
                         pd.Timedelta(minutes=5)

    # get start time of ambient measurement assume start 2hr before chamber experiment (2hr is enough for fill chambers)
    time_start_ambient = time_start_chamber - pd.Timedelta(hours=2)    
 
    # filter data only measured in ambient
    df_lvm_ambient = df_lvm[(df_lvm['Time']>=time_start_ambient)&
                            (df_lvm['Time']<=time_start_chamber)].reset_index(drop=True)
    df_lvm_ambient.drop_duplicates(subset=['Time'])  # delete duplicated Time
    
    # remove data when monitors are warming up (5min or 300sec)
    df_lvm_ambient = df_lvm_ambient.drop(index=np.arange(300)).reset_index(drop=True)

    # combine all the data, missing NOx data will be filled with corrected NOy data
    if df_NOx.shape[0] == 1:
        df_NOx = df_lvm_ambient[['Time','NO','NOy','dif']]
        df_NOx.columns = ['Time','NO','NOx','NO2'] 
        #df_NOx['NOx'] = df_NOx['NOx']*0.74-0.18
        df_NOx['NO2'] = df_NOx['NOx'] - df_NOx['NO']
    df=df_lvm_ambient.merge(df_NOx, on='Time', how='left').\
        merge(df_O3, on='Time', how='left')

    # rename columns (NO_y: NO from NOy monitor; NO_x: NO from NOx monitor; chamber: 1 = measure chamber air, 0 = measure ambient air; bagX: 1 = measure bag X air)
    df.columns = ['NO_y','Dif','NOy','bag1','bag2','bag3','chamber','amb','Time','NO_x','NOx','NO2','O3','Temp','Pressure','Flow rate','Date']
    df = df[['Time', 'NO_x', 'NO2', 'NOx', 'NO_y', 'NOy', 'O3', 'Temp', 'Pressure', 'Flow rate', 'amb', 'chamber', 'bag1', 'bag2', 'bag3']]
    
    return df


def get_combined_ambient_data(date, loc):
    """
    Get ambient air NOx, NOy, O3 concentration from chamber system (all data in ambient measurement mode)
    input: date (object): only one date, format: '2021-01-01' 
           loc (str): 'Redlands' or 'Riverside'
    return: df: columns = ['Time', 'NO_x', 'NO2', 'NOx', 'NO_y', 'NOy', 'O3', 'Temp',
                           'Pressure', 'Flow rate', 'amb', 'chamber', 'bag1', 'bag2', 'bag3']
    """

    filename_lvm = get_NOxNOyO3_filepath(date, 'NOy', loc)
    filename_NOx = get_NOxNOyO3_filepath(date, 'NOx', loc)
    filename_O3 = get_NOxNOyO3_filepath(date, 'O3', loc)
    df_lvm = get_raw_lvm(filename_lvm)
    df_NOx = get_raw_NOx(filename_NOx)
    df_O3 = get_raw_O3(filename_O3)

    # get data measured in chamber ('chamber' == 1 means monitors are measuring chamber air)
    df_lvm_ambient = df_lvm[df_lvm['chamber']==0].reset_index(drop=True)
 
    # combine all the data
    # combine all the data, missing NOx data will be filled with corrected NOy data
    if df_NOx.shape[0] == 1:
        df_NOx = df_lvm_ambient[['Time','NO','NOy','dif']]
        df_NOx.columns = ['Time','NO','NOx','NO2'] 
        df_NOx['NOx'] = np.nan
        df_NOx['NO2'] = np.nan
        df_NOx['NO'] = np.nan
    df=df_lvm_ambient.merge(df_NOx, on='Time', how='left').\
        merge(df_O3, on='Time', how='left')

    # rename columns (NO_y: NO from NOy monitor; NO_x: NO from NOx monitor; amb: 1 = measure ambient air; chamber: 1 = measure chamber air; bagX: 1 = measure bag X air)
    df.columns = ['NO_y','Dif','NOy','bag1','bag2','bag3','chamber','amb','Time','NO_x','NOx','NO2','O3','Temp','Pressure','Flow rate','Date']
    df = df[['Time', 'NO_x', 'NO2', 'NOx', 'NO_y', 'NOy', 'O3', 'Temp', 'Pressure', 'Flow rate', 'amb', 'chamber', 'bag1', 'bag2', 'bag3']]
    
    return df


def get_combined_chamber_data(date, loc):
    """
    Get chamber air NOx, NOy, O3 concentration
    input: date (object): only one date, format: '2021-01-01' 
           loc (str): 'Redlands' or 'Riverside'
    return: df: columns = ['Time', 'Second' ,'NO_x', 'NO2', 'NOx', 'NO_y', 'NOy', 'O3', 
                           'Temp','Pressure', 'Flow rate', 'amb', 'chamber', 'bag1', 
                           'bag2', 'bag3']
                Note: (NO_y: NO from NOy monitor; NO_x: NO from NOx monitor; chamber: 1 = measure chamber air, 0 = measure ambient air; bagX: 1 = measure bag X air)
    """

    filename_lvm = get_NOxNOyO3_filepath(date, 'NOy', loc)
    filename_NOx = get_NOxNOyO3_filepath(date, 'NOx', loc)
    filename_O3 = get_NOxNOyO3_filepath(date, 'O3', loc)
    df_lvm = get_raw_lvm(filename_lvm)
    df_NOx = get_raw_NOx(filename_NOx)
    df_O3 = get_raw_O3(filename_O3)

    # get data measured in chamber ('chamber' == 1 means monitors are measuring chamber air)
    df_lvm_chamber = df_lvm[df_lvm['chamber']==1].reset_index(drop=True)
 
    # combine all the data
    # combine all the data, missing NOx data will be filled with corrected NOy data
    if df_NOx.shape[0] == 1:
        df_NOx = df_lvm_chamber[['Time','NO','NOy','dif']]
        df_NOx.columns = ['Time','NO','NOx','NO2'] 
        df_NOx['NOx'] = df_NOx['NOx']*0.74-0.18
        df_NOx['NO2'] = df_NOx['NOx'] - df_NOx['NO']
    df=df_lvm_chamber.merge(df_NOx, on='Time', how='left').\
        merge(df_O3, on='Time', how='left')
    
    # calculate second since time_start
    time_start = df_lvm.loc[df_lvm['chamber']==1, 'Time'].iloc[0]
    df['Second'] = (df['Time'] - time_start).dt.seconds

    # rename columns (NO_y: NO from NOy monitor; NO_x: NO from NOx monitor; amb: 1 = measure ambient air; chamber: 1 = measure chamber air; bagX: 1 = measure bag X air)
    df.columns = ['NO_y','Dif','NOy','bag1','bag2','bag3','chamber','amb','Time','NO_x','NOx','NO2','O3','Temp','Pressure','Flow rate','Date','Second']
    df = df[['Time', 'Second', 'NO_x', 'NO2', 'NOx', 'NO_y', 'NOy', 'O3', 'Temp', 'Pressure', 'Flow rate', 'amb', 'chamber', 'bag1', 'bag2', 'bag3']]
    
    return df


def get_chamber_averaged_data(df):
    """
    Averaging NOx, NOy, O3 data in each chamber at every 10min step, use trim_mean at 0.1 to remove outliers. 
    input: df: contains chamber data, geenrated from get_combined_chamber_data()
    return: df_chamber_avg: 10min averaged NOx, NOy, O3. (Time: time measuring chambers, 
            UV_time: time of UV lights on)
    """

    # create df to store bag_avg data
    df_bag_avg = pd.DataFrame(columns=['Time','UV_time','Bag',
                                       'NO_y','NOy','NO_x','NOx','O3'])
    df_bag_avg['Time'] = np.arange(10,250,10) # time in chamber
    df_bag_avg['UV_time'] = [0,0,0] + np.arange(10,220,10).tolist() # time with UV
    df_bag_avg['Bag'] = ['bag1','bag2','bag3']*8

    # calculate 10min averaged value (trimmed by 0.1) with wall loss correction (5% per hour)
    for i in range(24):
        a = 10*60*i+3*60
        b = 10*60*i+9*60-15
        wall_loss = (i+1)*10/60*0.05+1 # calibrate wall loss
        df_bag_avg.loc[i,['O3']] = \
            df.loc[(df['Second']>=a)&(df['Second']<=b), ['O3']].\
                apply(lambda x: stats.trim_mean(x[x>0].dropna(), 0.15), axis=0)*wall_loss
        df_bag_avg.loc[i,['NO_x','NOx','NO_y', 'NOy']] = \
            df.loc[(df['Second']>=a)&(df['Second']<=b), ['NO_x','NOx','NO_y','NOy']].\
                apply(lambda x: stats.trim_mean(x[x>0].dropna(), 0.15), axis=0)
    
    return df_bag_avg

def linearRegssion_withNaN(x, y, fit_intercept=True):
    """
    Linear regression to X and Y with NaN values
    input: x, y: 1D array
           fit_intercept: True or False
    return: coef (float), intercept (float), r_value (float)
    """
    
    from sklearn.linear_model import LinearRegression

    # remove nan
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]    
    r_value = stats.pearsonr(x[mask], y[mask])[0]
    p_value = stats.pearsonr(x[mask], y[mask])[1]

    # Reshape x to be a 2D array required by LinearRegression
    x = [[i] for i in x] 

    if fit_intercept:
        model = LinearRegression(fit_intercept=True)
        model.fit(x,y)
        coef = model.coef_[0]
        intercept = model.intercept_
        return coef, intercept, r_value

    else:
        model = LinearRegression(fit_intercept=False)
        model.fit(x, y)
        coef = model.coef_[0]
        return coef, r_value


def get_o3_sensitivity(date, loc, plot=False):
    """
    Calculate O3 sensitivity results of daily chamber experiment. Linear regression applied
      to 10min averaged O3 in each bag to calculate 3hr projected O3 concentration.       
      3hr O3[bag1] - 3hr O3[bag2] = ΔO3[+NOx], 3hr O3[bag3] - 3hr O3[bag2] = ΔO3[+VOC],
    input: date (object): only one date, format: '2021-01-01'
           loc (str): 'Redlands' or 'Riverside'
           daily_plot (bool): True: plot daily O3 sensitivity results to check data quality
                              False: do not plot (default)
    return: df_linear : O3 sensitivity results (X_t0: initial concentration, X_sl:
                        slope, X_int:interception, X_r: r value)
            fig: plot of O3 sensitivity results
    """
    if loc != 'Redlands' and loc != 'Pasadena':
        print('No location found!')
        return

    df_chamber = get_combined_chamber_data(date, loc)
    df_bag_avg = get_chamber_averaged_data(df_chamber)

    # create df to store O3 sensitivity results (X_t0: initial concentration, X_sl: slope, X_int:interception, X_r: r value)
    df_linear = pd.DataFrame(columns=['Date','bag1_t0','bag2_t0','bag3_t0',
                                      'bag1_sl','bag1_int','bag1_r',
                                      'bag2_sl','bag2_int','bag2_r',
                                      'bag3_sl','bag3_int','bag3_r',
                                      'bag1_3hr','bag2_3hr','bag3_3hr',
                                      'b1_b2_3hr','b3_b2_3hr'], index=[0])
    df_linear['Date'] = date  
    df_linear['bag1_t0'] = df_bag_avg.loc[df_bag_avg['Time']==10,'O3'].iloc[0]
    df_linear['bag2_t0'] = df_bag_avg.loc[df_bag_avg['Time']==20,'O3'].iloc[0]
    df_linear['bag3_t0'] = df_bag_avg.loc[df_bag_avg['Time']==30,'O3'].iloc[0]

    # apply linear regression
    for i in ['bag1','bag2','bag3']:     
        df = df_bag_avg[(df_bag_avg['UV_time']>0)&(df_bag_avg['UV_time']<=180)&
                        (df_bag_avg['Bag']==i)].reset_index(drop=True)
        model = LinearRegression(fit_intercept=True)
        x = df['UV_time'].astype(float)
        y = df['O3'].astype(float)
        slope, intercept, r_value = linearRegssion_withNaN(x, y, fit_intercept=True)
    
        # customize column names
        i_sl = i+'_sl'
        i_int = i+'_int'
        i_r = i+'_r'
        i_3hr = i+'_3hr'
    
        # calculate 3hr projected O3 concentration and O3 sensitivity
        df_linear.loc[0,i_sl]=slope
        df_linear.loc[0,i_int]=intercept
        df_linear.loc[0,i_r]=r_value
        df_linear.loc[0,i_3hr] = slope*180 + intercept
        df_linear['b1_b2_3hr'] = df_linear['bag1_3hr'] - df_linear['bag2_3hr']
        df_linear['b3_b2_3hr'] = df_linear['bag3_3hr'] - df_linear['bag2_3hr']
    
    if plot:
        fig, ax=plt.subplots(1)
        for i in ['bag1','bag2','bag3']: 
            df = df_bag_avg[(df_bag_avg['UV_time']>0)&(df_bag_avg['UV_time']<=180)&
                            (df_bag_avg['Bag']==i)].reset_index(drop=True)
            ax.scatter(df['UV_time'], df['O3'])
            i_sl = i+'_sl'
            i_int = i+'_int'
            pred_x = np.arange(0,210,30)
            pred_O3 = df_linear.loc[0,i_sl]*pred_x+df_linear.loc[0,i_int]
            ax.plot(pred_x, pred_O3, label=i)
            ax.set_title(date)
        
        ax.set_xticks(np.arange(0,210,30))
        ax.legend(loc='lower right')
        ax.set_xlabel('UV time (min)')
        ax.set_ylabel('Ozone (ppb)')
        fig.show()

    return df_linear
