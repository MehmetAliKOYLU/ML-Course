import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('veriler.csv')
print(df)
boykilo = df[['boy','kilo']]
print(boykilo)


eksik = pd.read_csv('eksikveriler.csv')


## Eksik verileri tamamlamak icin kullandik

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
Yas = eksik.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)

## Veri Tipini degistirmek icin bunu kullaniyoruz.


## Label Encoder   KATEGORIK - > NUMERIC

ulke =  eksik.iloc[:,:1].values
print(ulke)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(eksik.iloc[:,0])
print(ulke)

## one hot encoding
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

##SONUC

sonuc = pd.DataFrame(data=ulke,index = range(22),columns=['fr','tr','us'])
print(sonuc)
sonuc2 = pd.DataFrame(data=Yas,index=range(22),columns=['Boy','kilo','yas'])
print(sonuc2)
cinsiyet = eksik.iloc[:,-1].values
print(cinsiyet)
sonuc3 = pd.DataFrame(data=cinsiyet,index= range (22), columns=['cinsiyet'])
print(sonuc3)

## Veri BIRLESTIRME 

ANA_VERI_CINSIYET_OLMADAN = pd.concat([sonuc,sonuc2],axis=1) ## AXIS 1 YAPIP YAN YANA EKLEME YAPTIRDIK
print(ANA_VERI_CINSIYET_OLMADAN)
ANA_VERI_  = pd.concat([ANA_VERI_CINSIYET_OLMADAN,sonuc3],axis=1)

## Veri kumesini test train bolme islemi

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(ANA_VERI_CINSIYET_OLMADAN,sonuc3,test_size=.2,random_state=42)


## StandarScaler
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)