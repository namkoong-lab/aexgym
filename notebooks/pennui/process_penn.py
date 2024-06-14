import pandas as pd 
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm 
from sklearn.linear_model import Ridge

columns_to_remove = ["flagwk0",
"flagwk1",
"flagwk2",
"flagwk3",
"flagwk4",
"flagwk5",
"flagwk6",
"flagwk7",
"flagwk8",
"flagwag0",
"flagwag1",
"flagwag2",
"flagwag3",
"flagwag4",
"flagwag5",
"flagwag6",
"flagwag7",
"flagwag8",
"wrweek0",
"wrweek1",
"wrweek2",
"wrweek3",
"wrweek4",
"wrweek5",
"wrweek6",
"wrweek7",
"wrweek8",
"wrwage0",
"wrwage1",
"wrwage2",
"wrwage3",
"wrwage4",
"wrwage5",
"wrwage6",
"wrwage7",
"wrwage8", 
'ab_dt', 
'newab_dt',
'areacode', 
'Unnamed: 0']

def process_penn(PATH, outcome=1, linear_impute = True): 
    """
    Code for preprocessing the PennUI dataset. 

    args:
    - PATH: path to the PennUI dataset
    - outcome: 1 or 2 for the two outcomes in the dataset 
    - linear_impute: whether to impute missing values using linear regression 

    returns:
    - df: preprocessed dataframe
    """
    df = pd.read_csv(PATH)
    df = df.drop(columns=columns_to_remove)
    for column in df.columns:
        if df[column].isnull().mean() > 0.2:
            df = df.drop(column, axis=1)

    numeric_columns = df.select_dtypes(include=['number']).columns
    df = df[numeric_columns]
    df = df.dropna()
    binary_columns = [col for col in numeric_columns if df[col].isin([0, 1]).all()] + ['inuidur1', 'inuidur2']
    columns_to_standardize = [col for col in numeric_columns if col not in binary_columns]
    scaler = StandardScaler()
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
    
    new_df = df.copy()
    if linear_impute:
        treatments = ['t0', 't1', 't2', 't3', 't4', 't5', 't6']

        #for each treatment, fit a model to the treatment group and predict for all observations
        for treatment in treatments:

            # exclude to treatment group 
            temp_df = df[df[treatment] == 1.0].drop(columns=treatments)

            #fit model to treatment group 
            X = temp_df.drop(columns = ['inuidur1', 'inuidur2'])
            y = - temp_df[f'inuidur{outcome}']
            X = sm.add_constant(X)
            model = Ridge(alpha=1.0)
            model.fit(X, y)

            # predict for all observations 
            new_X = df.drop(columns = ['inuidur1', 'inuidur2']+treatments)
            new_X = sm.add_constant(new_X)
            new_Y = model.predict(new_X)
            new_df[f'{treatment}_predict'] = new_Y
        for i, row in enumerate(new_df.iterrows()):
            for treatment in treatments:
                if row[1][treatment] == 1.0:
                    new_df.iloc[i, new_df.columns.get_loc(f'{treatment}_predict')] = - row[1][f'inuidur{outcome}']
                    break

        new_df = new_df.drop(columns = treatments + ['inuidur1', 'inuidur2'])
        new_df = sm.add_constant(new_df, has_constant='add')
    return new_df.astype('float32')