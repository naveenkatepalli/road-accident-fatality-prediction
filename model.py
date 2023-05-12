#this is machine learning model used in the website
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
ss = StandardScaler()
lor = LogisticRegression(random_state=0)
df = pd.read_csv(
    r"C:\Users\Naveen\OneDrive\Desktop\road accident\road accident data.csv")


def algo(df):
    x = df.iloc[:, 0:5].values
    y = df.iloc[:, 5].values
    x = ss.fit_transform(x)
    lor.fit(x, y)


def predict(persons, day_week, hour, drunk_drive, month):
    a = lor.predict(ss.transform(
        [[persons, day_week, hour, drunk_drive, month]]))
    print(a)


model = algo(df)
result = predict(1, 5, 8, 0, 5)
print(result)
