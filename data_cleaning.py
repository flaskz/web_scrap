# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 10:30:37 2018

@author: l.ikeda
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

with open('apts.pkl', 'rb') as f:
    data = pickle.load(f)
    
meu_teste = dict(data[0])

meu_teste['suites'] = [1]
meu_teste['rentalPrices'] = ['2100']
meu_teste['bedrooms'] = [1]
meu_teste['areas'] = [90]
meu_teste['parkingSpaces'] = [1]
meu_teste['address'][3] = 'ALTO DA RUA XV'

data.append(meu_teste)
data[-1]

    
set([x['listingType'] for x in data])
set([len(x['rentalPrices']) for x in data])
set([len(x['bedrooms']) for x in data])
set([x['bedrooms'][0] for x in data if len(x['bedrooms']) > 0])
set([int(x['areas'][0]) for x in data])
set([len(x['parkingSpaces']) for x in data])
set([len(x['suites']) for x in data])
set([x['parkingSpaces'][0] for x in data if len(x['parkingSpaces']) > 0])
len(set([x['address'][3] for x in data]))
len(set([y for x in data for y in x['amenities']]))

df = pd.DataFrame()

df['aluguel'] = [int(''.join(x['rentalPrices'][0].split('.'))) for x in data]
df['iptu'] = [int(''.join(x['iptuPrices'][0].split('.'))) for x in data]
df['condominio'] = [int(''.join(x['condoFee'].split('.'))) for x in data]
df['area'] = [int(x['areas'][0]) for x in data]
df['quartos'] = [int(x['bedrooms'][0]) if len(x['bedrooms']) > 0 else 0 for x in data]
df['suites'] = [int(x['suites'][0]) if len(x['suites']) > 0 else 0 for x in data]
df['vagas'] = [int(x['parkingSpaces'][0]) if len(x['parkingSpaces']) > 0 else 0 for x in data]
df['listingId'] = [x['listingId'] for x in data]
df['bairro'] = [x['address'][3] for x in data]
df['url'] = [x['url'] for x in data]

df['preco'] = df['aluguel'] + df['condominio']

df.info()



#for d in data:
#    df['aluguel'] = int(''.join(d['rentalPrices'][0].split('.')))
#    df['iptu'] = int(''.join(d['iptuPrices'][0].split('.')))
#    df['condominio'] = int(''.join(d['condoFee'].split('.')))

print('aluguel:', np.mean(df['aluguel']))
print('iptu:', np.mean(df['iptu']))
print('condominio:', np.mean(df['condominio']))

df.head()

# por bairro
teste = pd.DataFrame()
teste = df.join(pd.get_dummies(df['bairro']).iloc[:,:-1])

teste = teste.drop(columns = ['listingId', 'aluguel', 'iptu', 'condominio'], axis=1)
teste = teste.loc[(teste['preco'] < 3000) & (teste['area'] < 200)]


aux = {}
for bairro in set(df['bairro']):
    print(bairro, np.mean(df['preco'].loc[df['bairro']==bairro]))
    aux[bairro] = np.mean(df['preco'].loc[df['bairro']==bairro])

   
    
sorted_by_value = sorted(aux.items(), key=lambda kv: kv[1])

for i in range(10):
    try:
        hue = sorted_by_value[i*10:(i+1)*10]
        plt.bar([x[0] for x in hue], [x[1] for x in hue])
        plt.xticks(np.arange(len(hue)), rotation=65)
        plt.show()
    except Exception as e:
        print(e)
        break
    
#for i in range(len(set(df['bairro']))):
    

from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(df[['bairro_encoded', 'area', 'quartos', 'vagas']], df['preco'], test_size = 0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(teste.drop(columns = ['bairro', 'preco', 'url'], axis=1), teste['preco'], test_size = 0.3, random_state=0, shuffle=False)

from sklearn.ensemble import RandomForestRegressor
rfg = RandomForestRegressor(n_estimators=50, random_state=0)

rfg.fit(X_train, y_train)
preds_rfg = rfg.predict(X_test)

#rfg.fit(np.array(X_train).reshape(-1,1), y_train)
#preds_rfg = rfg.predict(np.array(X_test).reshape(-1,1))

rfg.feature_importances_

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(np.array(X_train).reshape(-1,1), y_train)
preds_lr = lr.predict(np.array(X_test).reshape(-1,1))
lr.coef_

from sklearn.metrics import mean_squared_error, r2_score
print(np.sqrt(mean_squared_error(y_test, preds_lr)))
print(r2_score(y_test, preds_lr))

print(np.sqrt(mean_squared_error(y_test, preds_rfg)))
print(r2_score(y_test, preds_rfg))

plt.scatter(X_test, y_test)
# plt.plot(X_test.area, X_test.quartos)
plt.plot(X_test, preds_lr, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Initialising the ANN
regressor = Sequential()

# Adding the input layer and the first hidden layer
regressor.add(Dense(units = 60, kernel_initializer = 'uniform', activation = 'relu', input_dim = 79))

# Adding the second hidden layer
regressor.add(Dense(units = 120, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the second hidden layer
regressor.add(Dense(units = 200, kernel_initializer = 'uniform', activation = 'relu'))

regressor.add(Dropout(0.5))

# Adding the second hidden layer
regressor.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the second hidden layer
regressor.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))

regressor.add(Dropout(0.5))

# Adding the second hidden layer
regressor.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))

# Compiling the ANN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae'])

# Fitting the ANN to the Training set
regressor.fit(X_train, y_train, batch_size = 60, epochs = 100)

reg_preds = regressor.predict(sc.transform(teste.drop(columns = ['bairro', 'preco', 'url'], axis=1)))

lst = []
lst2 = []
for i in range(len(reg_preds)):
    if ((teste['preco'].iloc[i] < (reg_preds[i] - 700)) & (teste['preco'].iloc[i] < 2000) & (teste['preco'].iloc[i] > 1500) & (teste['bairro'].iloc[i] == 'ALTO DA RUA XV')):
       if teste.iloc[i] not in lst:
           lst2.append(teste.iloc[i]) 

           
       


