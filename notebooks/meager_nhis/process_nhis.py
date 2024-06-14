import numpy as np 
import pandas as pd 

def preprocess_nhis(df, 
                    clusters = ['born_us',
                                'born_mexico',
                                'born_southamerica',
                                'born_europe',
                                'born_russia',
                                'born_africa',
                                'born_middle_east',
                                'born_indian',
                                'born_asia',
                                'born_se_asia']):
    for column in df.columns:
        if df[column].isnull().mean() > 0.01:
            df.drop(column, axis=1, inplace=True)
    df = df.drop(columns=['Unnamed: 0', 'year'])
    df.dropna()
    df = df.rename(columns={'medicaid_': 'treatment'})
    df = df.applymap(lambda x: int(x) if isinstance(x, bool) else x)

    array = df[clusters].values
    cluster_list = []
    for value in np.argmax(array, axis=1):
        cluster_list.append(clusters[value])

    df['cluster'] = cluster_list
    df = df.drop(clusters, axis=1)
    exclude_column = 'cluster'
    df[df.columns.difference([exclude_column])] = df[df.columns.difference([exclude_column])].astype(float)
    return df

def get_nhis_cluster_dict(df):
    cluster_dict = {}
    for i, cluster in enumerate(df['cluster'].unique()):
        temp_dict = {}
        temp_df = df[df['cluster'] == cluster].drop(columns=['cluster'])

        response_columns = ['care_office_2wks'] 

        temp_dict['response'] = temp_df[response_columns]
        temp_dict['features'] = temp_df.drop(columns=response_columns)

        cluster_dict[i] = temp_dict
    return cluster_dict