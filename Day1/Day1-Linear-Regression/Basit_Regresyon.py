import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# Y = AX+B
df = pd.read_csv('satislar.csv')
print(df.head())

#Veri Onisleme islemi 
aylar = df[['Aylar']]
print(aylar)

satislar = df[['Satislar']]
print(satislar)

satislar2 = df.iloc[:,:1].values
print(satislar2)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(
    aylar,satislar,test_size=.25,random_state=42
    )
'''
from sklearn.preprocessing import StandardScaler

# %%
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
'''
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train,y_train)


tahmin = model.predict(x_test)


x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test,model.predict(x_test))
