import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def load_data(path):
    data=pd.read_csv(path)
    return data

def clean_data(df):
    drop_cols=['Day_Mins','Night_Mins','Eve_Mins','Intl_Mins','Phone']
    df['Total_cals']=df['Day_Calls']+df['Eve_Calls']+df['Night_Calls']+df['Intl_Calls']
    drop_cols+=['Day_Calls','Eve_Calls','Night_Calls','Intl_Calls']

    df['total_charges']=df['Day_Charge']+df['Eve_Charge']+df['Night_Charge']+df['Intl_Charge']
    drop_cols+=['Day_Charge','Eve_Charge','Night_Charge','Intl_Charge']

    df.drop(columns=drop_cols,axis=1,inplace=True)

    return df
def split_feature(df):
    numarical_features=df.select_dtypes(include=['number']).columns.tolist()
    categorical_featfures=df.select_dtypes(exclude=['number']).columns.tolist()

    return numarical_features,categorical_featfures

def preprocess_data(numarical_data,categorical_data):
    numarical_pipeline=Pipeline([
        ['imputer',SimpleImputer(strategy='mean')],
        ['scaler',StandardScaler()]

    ])
    categorical_pipiline=Pipeline([
        ['imputer',SimpleImputer(strategy='most_frequent')],
        ['encoder',OneHotEncoder(handle_unknown='ignore')]
    ])
    processor=ColumnTransformer([
        ['num',numarical_pipeline,numarical_data],
        ['cat',categorical_pipiline,categorical_data]
    ])

    return processor


def convert_processor_to_df(processor, processed_data):

    if hasattr(processed_data, "toarray"):
        processed_data = processed_data.toarray()

    data = pd.DataFrame(
        processed_data,
        columns=processor.get_feature_names_out()
    )

    return data