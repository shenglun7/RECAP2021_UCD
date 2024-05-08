"""
This file contains source code that will be used in generating input file for chamber model
"""

import pandas as pd
import numpy as np


def replace_CTM_by_measurement(df_CTM, df_measure, loc):
    """
    Replace CTM results of SARPC11 data by measurement data during the campaign
    input: df_CTM (df): daily CTM result data file, e.g. redlands_4km_20200801_11.txt
           df_measure (df): daily measured SAPRC data
    return: df_saprc (df): updated daily SAPRC11 data file with measurement data
    """

    list_saprc_measure = df_measure['S11'].dropna().unique()

    for i in list_saprc_measure:
        i_tag = i + '_X'
        measure_conc = df_measure[df_measure['S11']==i]['Value'].values[0]
        measure_conc = measure_conc/1000  # convert to ppm
        
        # NGOR, NaN, CO2 are not included in SAPRC11
        if loc.lower() == 'pasadena':
            if i != 'NROG' and i != 'CO2':
                df_CTM.loc[df_CTM['target']==i,'grid'] = measure_conc
                df_CTM.loc[df_CTM['target'].str.contains(i_tag), 'grid']=0
        
        elif loc.lower() == 'redlands':
            if measure_conc>0 and measure_conc<500 and i !='NROG' and \
                  i !='CO' and i !='ALK1' and i !='ALK2':
                # correct HCHO and CCHO (supplementory meterial)
                if(i=='HCHO'):
                    measure_conc = measure_conc*4.165*0.9
                if(i=='CCHO'):
                    measure_conc = measure_conc*6.00*0.66
                df_CTM.loc[df_CTM['target']==i,'grid'] = measure_conc/1000 # convert to ppm
                df_CTM.loc[df_CTM['target'].str.contains(i_tag), 'grid']=0

        else:
            print('location not found')
    
    return df_CTM


def change_format_saprc(df_saprc):
    """
    Change the format of string and float in saprc df (read from saprc11 .txt file)
    'found' contains index of SAPRC species (need to be a 3 character string), 
    'target' contains name of SAPRC species (need to be a 16 character string)), 
    'grid' contains concentration (need to be in scientific notation, with 2 decimal places)

    input: df_saprc (df): daily SAPRC11 data file
    return: df_saprc (df): updated daily SAPRC11 data file
    """

    for i in df_saprc.index:
        df_saprc.loc[i,'found'] = "{:>3}".format(df_saprc.loc[i,'found'])
        df_saprc.loc[i,'target'] = "{:<16}".format(df_saprc.loc[i,'target'])
    df_saprc.grid = df_saprc.grid.apply(lambda x: '%.2e' % x)

    return df_saprc


def save_saprc_txt(df_saprc, date, loc):
    """
    Save daily SAPRC11 data to txt file
    input: df_saprc (df): daily SAPRC11 data file, formatted
           date (date format): date of the data, e.g. '20200801'
           loc (str): location of the data, e.g. 'Redlands', 'pasadena'
           heading (str): heading of the txt file, e.g. 'redlands_4km_20200801_11.txt'
    return: txt file with SAPRC11 data saved to data folder
    """
    
    # get heading (first 3 lines) from CTM file and use it for new SAPRC file
    date_2020 = date + pd.offsets.DateOffset(days=1)
    filepath_CTM = '../../data/raw/CTM_2020/' + loc.lower() + '_4km_2020' + \
        date_2020.strftime('%m%d') + '_11.txt'
    with open(filepath_CTM, 'r') as f_old:
        contents = f_old.readlines()
    f_old.close()

    # save SAPRC data to txt file
    filepath_saprc = '../../data/final/Chamber_model_input/'+ \
        loc.lower() +'_4km_2021' + date.strftime('%m%d') + '_11_cor.txt'
    
    with open(filepath_saprc, 'w') as f:
        # write heading
        for j in range(3):
            f.write(contents[j])
        # write new SAPRC data to file
        dfAsString = df_saprc[['found','target','grid']].\
            to_string(header=False, index=False)
        f.write(dfAsString)
    f.close()