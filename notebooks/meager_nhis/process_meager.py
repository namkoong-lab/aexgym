import pandas as pd 
import numpy as np 
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

def get_meager_cluster_dict(df):
    cluster_dict = {}
    for i, cluster in enumerate(df['cluster'].unique()):
        temp_dict = {}
        temp_df = df[df['cluster'] == cluster].drop(columns=['cluster'])

        response_columns = ['profit', 'expenditures', 'revenues', 'consumption'] 

        temp_dict['response'] = temp_df[response_columns]
        temp_dict['features'] = temp_df.drop(columns=response_columns)

        cluster_dict[i] = temp_dict
    return cluster_dict


def get_meager_cluster_df(DATAPATH = 'Meager_data.csv', cluster_type = 'cluster_id', filter_both_treat=False):
    
    #read in csv 
    df = pd.read_csv(DATAPATH)

    #check if cluster_type is valid 
    if cluster_type not in df.columns:
        raise ValueError(f'{cluster_type} not in columns of dataframe. Must be one of [site, district, cluster_id]')
    
    #prepare columns to drop from dataframe
    clusters_to_drop = ['site', 'district', 'cluster_id']
    clusters_to_drop.remove(cluster_type)
    columns_to_drop = ['Unnamed: 0', 'consumerdurables', 'married', 'primary', 'hhsize', 'individual_rand', 'temptation']
    
    #drop columns and then values with nulls
    df = df.drop(columns = [*clusters_to_drop, *columns_to_drop], axis=1)
    df = df.dropna()

    #add indicators for countries and convert boolean columns to binary
    df = pd.get_dummies(df, columns = ['country'])
    df = df.replace({True: 1, False: 0})
    if filter_both_treat:
        id_list = []
        for id in df[cluster_type].unique():
            if len(df[df[cluster_type] == id]['treatment'].unique()) > 1:
                id_list.append(id)
        df = df[df[cluster_type].isin(id_list)]
    #standardize columns with numbers that are not binary 
    numeric_columns = df.select_dtypes(include=['number']).columns
    binary_columns = [col for col in numeric_columns if df[col].isin([0, 1]).all()]
    binary_columns = binary_columns + ['profit', 'revenues', 'expenditures', 'consumption']
    columns_to_standardize = [col for col in numeric_columns if col not in binary_columns]

    # Standardize 
    scaler = StandardScaler()
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
    #df[['profit', 'expenditures', 'revenues', 'consumption']] = df[['profit', 'expenditures', 'revenues', 'consumption']].apply(lambda x: x - x.mean())

    #filter out clusters that do not have both treatment and control
    

    #convert to float 
    numeric_columns = df.select_dtypes(include=['number']).columns
    df[numeric_columns] = df[numeric_columns].astype(float)
    df = sm.add_constant(df, has_constant='add')
    df = df.rename(columns={cluster_type: 'cluster'})
    return df