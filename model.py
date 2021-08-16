import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

dataset=pd.read_csv("E:\python\Price Prediction project -regression\Price Prediction project -regression\data1.csv")

dataset.head()

df = dataset.copy()

df=df.drop(['Unnamed: 0'],axis =1)

df.dtypes
df.isnull().sum()

df['price']=df['price'].fillna(0)
df.dropna(subset=["train_class","fare"], inplace=True)
df=df.drop(['insert_date'],axis=1)

df = df.reset_index()
import datetime
datetimeFormat = '%Y-%m-%d %H:%M:%S'
def fun(a,b):
    diff = datetime.datetime.strptime(b, datetimeFormat)- datetime.datetime.strptime(a, datetimeFormat)
    return(diff.seconds/3600.0)         


df['travel_time_in_hrs'] = df.apply(lambda x:fun(x['start_date'],x['end_date']),axis=1) 

drop_features = ['start_date','end_date']
df.drop(drop_features,axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder

lab_en = LabelEncoder()
df.iloc[:,1] = lab_en.fit_transform(df.iloc[:,1])
df.iloc[:,2] = lab_en.fit_transform(df.iloc[:,2])
df.iloc[:,3] = lab_en.fit_transform(df.iloc[:,3])
df.iloc[:,5] = lab_en.fit_transform(df.iloc[:,5])
df.iloc[:,6] = lab_en.fit_transform(df.iloc[:,6])

X = df.drop(['price'], axis=1)
Y = df[['price']]

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X,Y)

pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
