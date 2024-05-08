# This contains source code that will be used in extract data from AQMD monitoring station

import pandas as pd
import numpy as np
from scipy import stats
import datetime as dt
import os
from sklearn.linear_model import LinearRegression


def get_hourly_aqmd(filepath):
    """
    Get hourly AQMD data for a given year and location
    data source: https://www.arb.ca.gov/aqmis2/aqmis2.php
    input: filepath
    return: df with hourly data
    """

    df = pd.read_csv(filepath)
    df = df[~(df['start_hour'].isna())]
    df = df[['date','start_hour','value','variable','units','quality','name ']]
    df.columns = ['Date','Hour','Value','Variable','Units','Quality','Site']
    df['Date'] = pd.to_datetime(df['Date'])

    return df

def convert_F_to_C(temp_F):
    """
    convert temperature from F to C
    input: temp_F (float): temperature in F
    return temp_C (float): temperature in C
    """

    temp_C = (temp_F-32)*5/9
    return temp_C


def get_MDA8_O3(filepath):
    """
    Get daily MDA8 O3 data from AQMD data file
    input: filepath (str): path to AQMD data file
    return: df
    """
    site = filepath.split('/')[-1].split('_')[-1].split('.')[0]
    if site == 'RL':
        site = 'Redlands'
    elif site == 'PA':
        site = 'Pasadena'

    df=pd.read_csv(filepath)
    df=df[df['name'].notnull()]
    df.loc[:,'summary_date']=pd.to_datetime(df.loc[:,'summary_date'], format='%Y-%m-%d', exact=False)
    df=df[['summary_date','ozone_dmol8n','name']]
    df.loc[:,'ozone_dmol8n']=df.loc[:,'ozone_dmol8n'].astype(float)*1000  # unit to ppbv
    df=df.rename(columns={'summary_date':'Date','ozone_dmol8n':'MDA8','name':'Site'})
    df.loc[:,'Site']=site

    return df


def get_morning_daily_avg_aqmd(filepath):
    """
    Get daily averaged aqmd data from morning hours (10AM ~ 12PM). 
    This time correspond to the sampling time of smog chamber system.
    input: filepath
    return: df
    """

    df = get_hourly_aqmd(filepath)
    df = df[df['Hour'].isin([10,11,12])].groupby('Date').\
            agg({'Value':'mean','Variable':'first','Units':'first',
                 'Quality':'first','Site':'first'}).reset_index()
    return df


###############################################
# treat .ict files
###############################################
def extract_ict_data(filepath):
    """
    extract data from .ict files collected by NOAA and CARB during RECAP-2021 campaign in Pasadena, CA.
    input: filepath (str): path to .ict file
    return: df
    """

    # open as df and skip the information lines
    with open(filepath) as f:
        lines = f.readlines()
    nline = int(lines[0].split(',')[0])
    df = pd.read_csv(filepath, skiprows=nline-1)
    df.columns = df.columns.str.replace(" ","") # delete space in column names
    
    # get Date and start time from information lines
    year = int(lines[6].split(',')[0])  
    month = int(lines[6].split(',')[1])
    day = int(lines[6].split(',')[2])
    start_date=dt.datetime(year, month, day).date()

    if 'PTR' in filepath:
        start_time=dt.datetime.strptime(lines[8].split(',')[2].split(' ')[5],"%H:%M").time()
    else:
        start_time=dt.datetime.strptime("00:00","%H:%M").time()
     
    # UTC to PDT 
    hours_to_pdt = dt.timedelta(hours = 7) # 7 hours minors from UTC to PDT
    df['Date_pdt'] = pd.to_timedelta(df.iloc[:,0], unit='s') + \
                        pd.to_datetime(dt.datetime.combine(start_date, start_time) - hours_to_pdt)
    
    # reshape df
    df = df.iloc[:,3:].melt(id_vars=['Date_pdt'], var_name='Parameter', value_name="Value")
    df = df[~(df['Parameter'].str.contains('PST'))]

    return df


###############################################
# ambient data for chamber model
###############################################
def generate_ambient_model_input(df_O3, df_NO, df_NO2, dates, loc):
    """
    Generate model input for ambient data for certain location and range of dates
    input: df_O3 (df): daily averaged O3 data
           df_NO (df): daily averaged NO data
           df_NO2 (df): daily averaged NO2 data
           dates (list): list of dates in pd.datetime format
           loc (str): location of the data, e.g. 'Redlands', 'Sacramento'
    return: txt file with model input saved to data folder
    """
    
    df_merge = df_O3[['Date','Value']].\
        merge(df_NO[['Date','Value']], on='Date', how='left').\
            merge(df_NO2[['Date','Value']], on='Date', how='left')
    df_merge.columns = ['Date','O3','NO','NO2']
    parameter = ['O3','NO','NO2']

    for i in dates:
        df_daily=pd.DataFrame(index=[1,2,3], columns=['value'])
        date=i.strftime('%Y%m%d')

        for j in range(3):
            conc = df_merge[df_merge['Date']==i][parameter[j]].values[0]
            if conc <= 0:
                print('negative value for '+parameter[j]+' at '+date)
                conc = 0
            elif conc == np.nan:
                print('NaN value for '+parameter[j]+' at '+date)
                conc = 0
            df_daily.iloc[j,0] = format(conc, '.2e')  # change format of float

        # save to txt file
        filename = '../../data/intermediate/Model/Input/Ambient/' + loc.lower() +\
                        '_ambient_' + date + '_11.txt'
        df_daily.to_csv(filename, index=True, header=False, sep=' ')
    


def calculate_month_avg_CO_Bio(CO, Temp, RH):
    """
    Calculate monthly average CO*Bio (Wu et al. 2021) equation from Guenther et al. 1991
    input: CO (float): CO concentration in ppm
           Temp (float): temperature in C
           RH (float): relative humidity in %
    return CO*Bio (float): CO-Bio in ppm
    """
    Temp = Temp + 273.15

    # Calculate Biogenix factor
    T1=95100
    T2=231000
    T3=311.83
    Ts=301
    R=8.314
    H1=0.00236
    H2=0.8495

    factor_T = (np.exp(T1*(Temp-Ts)/R/Temp/Ts))/(1+np.exp(T2*(Temp-T3)/R/Temp/Ts))
    factor_RH = RH*H1 + H2
    factor_RH = factor_RH/(40*H1+H2)
    factor_Bio = factor_T*factor_RH
    CO_Bio = CO*factor_T*factor_RH

    return CO_Bio