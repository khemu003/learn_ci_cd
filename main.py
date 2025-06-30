import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("D:\\DataSet\\insurance.csv")
print(df.head(2))

numerical = df.select_dtypes(include=["int64", "float64"])
categorical = df.select_dtypes(include=["object"])


print(df.isnull().sum())
if df.empty==True:
    si = SimpleImputer()
    for i in df.columns:
        if df[i].isnull().sum() > 0:
            df[i] = si.fit_transform(df[i])

print('-'*20)
print("Now DataFrame have not Null value.")
print('-'*20)

le = LabelEncoder()
for col in categorical:
    df[col] = le.fit_transform(df[col])

print("Now DataFrame have not Categorical columns.")
print('-'*20)

x = df.drop(columns=["charges"])
y = df["charges"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score = r2_score(y_test, y_pred)
print(f"score of model : {score}")