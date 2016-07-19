# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a python script for pre-process of data.
"""

import numpy as np
import pandas as pd


#%%
def select_data(dataframe=None, csv_file=None, col_dict=None):
    """
    select certain data with given col_dict from dataframe of csv file.
    @params:
    dataframe: pandas.DataFrame object
    csv_file: string, csv filename; dataframe and csv_file cannot be both None.
    col_dict: dict object, with column name as the key.
    """
    if dataframe is not None:
        df = dataframe.copy()
    elif csv_file is not None:
        df = pd.read_csv(csv_file)
    else:
        print('dataframe and csv_file should not be None at the same time!')
        
    column = df.columns.tolist()
    if isinstance(col_dict, dict):
        for col, value in col_dict:
            if col in column:
                df = df[df[col] == value]
                if not df:
                    print('selected dataframe is empty')
                    return None
            else:
                print('wrong column name: {}, please check col_dict!'.format(col))
                return None
    else:
        print('col_dict should be a dict!' )
        return None    
    return df    
